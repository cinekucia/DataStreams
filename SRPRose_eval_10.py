import os
import pandas as pd
from capymoa.stream import ARFFStream
from SRPRoseRiver import SRPRoseRiver
from river import tree
from river.drift import ADWIN
from RiverWrapperClassifier import RiverWrapperClassifier
from Prequential_evaluation_proba import prequential_evaluation_proba
from river.metrics import Accuracy as AccMetric

def run_evaluation(arff_file_path: str, output_dir: str, \
                   window_size: int = 1000, max_instances: int = 1_000_000_000,
                   optimise: bool = False, progress_bar: bool = True) -> pd.DataFrame:
    """
    Run a probability-aware prequential evaluation on a single ARFF file using SRPRoseRiver,
    save windowed metrics, and return cumulative metrics as a DataFrame.

    Parameters:
    - arff_file_path: Path to the .arff data file.
    - output_dir: Directory where window CSV results will be written.
    - window_size: Size of the sliding window for windowed metrics.
    - max_instances: Maximum number of instances to process.
    - optimise: Whether to optimise parameters (unused for wrapper).
    - progress_bar: Show progress bar if True.

    Returns:
    - DataFrame containing cumulative metrics for this stream.
    """
    print(f"Processing stream: {arff_file_path}")

    # 1. Build the stream
    stream = ARFFStream(arff_file_path)

    # 2. Extract schema and nominal attributes
    schema = stream.get_schema()
    print("Schema extracted from stream.")
    nominal_attrs: list[str] = []
    try:
        moa_header = schema.get_moa_header()
        num_features = schema.get_num_attributes()
        print(f"Iterating through {num_features} features for nominal types...")
        for i in range(num_features):
            attr = moa_header.attribute(i)
            if attr.isNominal():
                nominal_attrs.append(attr.name())
        print(f"Nominal attributes: {nominal_attrs}")
    except AttributeError:
        print("No MOA header available; proceeding without nominal attributes.")

    # 3. Instantiate SRPRoseRiver model
    base_ht_model = tree.HoeffdingTreeClassifier(
        grace_period=50,
        delta=0.01,
        nominal_attributes=nominal_attrs if nominal_attrs else None # CRITICAL LINE
    )
    print(f"Base Hoeffding Tree configured with nominal_attributes: {base_ht_model.nominal_attributes}")


    srp_wise_rose_model = SRPRoseRiver(
    model=base_ht_model,
    n_models=10,
    lam=6.0,
    theta_class_decay=0.99,
    rose_window_size=200,
    drift_detector=ADWIN(delta=1e-5),
    warning_detector=ADWIN(delta=1e-4),
    seed=1
)

    # 4. Wrap for River compatibility
    model = RiverWrapperClassifier(river_model=srp_wise_rose_model, schema=schema)

    # 5. Run prequential evaluation
    results = prequential_evaluation_proba(
        stream=stream,
        learner=model,
        window_size=window_size,
        store_predictions=False,
        store_y=False,
        optimise=optimise,
        progress_bar=progress_bar
    )

    # 6. Collect cumulative metrics into DataFrame
    base_name = os.path.splitext(os.path.basename(arff_file_path))[0]
    model_name = "SRPRoseRiver_10_6_200"
    cumulative = results.cumulative
    metrics = {
        "Learner": model_name,
        "Stream": base_name,
        "Max Instances": results.max_instances(),
        "Wallclock Time (s)": results.wallclock(),
        "Cumulative Accuracy": cumulative.accuracy(),
        "Cumulative Precision": cumulative.precision(),
        "Cumulative Recall": cumulative.recall(),
        "Cumulative Kappa M": cumulative.kappa_m(),
        "Cumulative Kappa T": cumulative.kappa_t(),
    }
    df_metrics = pd.DataFrame([metrics])

    # 7. Save windowed metrics
    df_windows = results.metrics_per_window()
    windows_csv = os.path.join(output_dir, f"windows_{model_name}_{base_name}.csv")
    df_windows.to_csv(windows_csv, index=False)
    print(f"✅ Saved windowed metrics to {windows_csv}")

    return df_metrics


if __name__ == "__main__":
    data_dir = "./data"
    output_dir = "./FINAL_results"
    os.makedirs(output_dir, exist_ok=True)

    dataset_files = [
        "AGR_a.arff", "AGR_g.arff", "HYPER.arff", "LED_a.arff", "LED_g.arff",
        "RBF_f.arff", "RBF_m.arff", "RTG.arff", "SEA_a.arff", "SEA_g.arff",
        "covtype.arff","electricity.arff","airlines.arff","internet_ads.arff","kdd99.arff"
    ]

    all_metrics: list[pd.DataFrame] = []
    for fname in dataset_files:
        file_path = os.path.join(data_dir, fname)
        if not os.path.isfile(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue
        df = run_evaluation(file_path, output_dir)
        all_metrics.append(df)

    # Combine all cumulative metrics into one CSV
    if all_metrics:
        all_df = pd.concat(all_metrics, ignore_index=True)
        combined_csv = os.path.join(output_dir, f"metrics_SRPROSE_10_all_streams.csv")
        all_df.to_csv(combined_csv, index=False)
        print(f"✅ Saved combined cumulative metrics to {combined_csv}")
    else:
        print("No metrics to combine.")
