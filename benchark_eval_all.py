import os
import pandas as pd
from capymoa.stream import ARFFStream
from capymoa.classifier import AdaptiveRandomForestClassifier
from river.drift import ADWIN
from RiverWrapperClassifier import RiverWrapperClassifier
from Prequential_evaluation_proba import prequential_evaluation_proba
from river import forest

def run_evaluation(arff_file_path: str, output_dir: str, \
                   window_size: int = 1000, max_instances: int = 1_000_000_000,
                   optimise: bool = False, progress_bar: bool = True):
    """
    Run a probability-aware prequential evaluation on a single ARFF file and save metrics.

    Parameters:
    - arff_file_path: Path to the .arff data file.
    - output_dir: Directory where CSV results will be written.
    - window_size: Size of the sliding window for windowed metrics.
    - max_instances: Maximum number of instances to process.
    - optimise: Whether to optimise parameters (unused for wrapper).
    - progress_bar: Show progress bar if True.
    """
    print(f"Processing stream: {arff_file_path}")

    # 1. Build the stream
    stream = ARFFStream(arff_file_path)

    # 2. Extract schema and nominal attributes
    schema = stream.get_schema()
    print("Schema extracted from stream.")
    nominal_attrs = []
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
        nominal_attrs = []

    # 3. Instantiate the ARF model
    arf = forest.ARFClassifier(
        n_models=10,
        seed=42,
        grace_period=50,
        delta=0.01,
        nominal_attributes=nominal_attrs or None,
        leaf_prediction="nba",
        drift_detector=ADWIN(delta=0.001),
        warning_detector=ADWIN(delta=0.01)
    )

    # 4. Wrap for River compatibility
    model = RiverWrapperClassifier(river_model=arf, schema=schema)

    # 5. Run prequential evaluation
    results = prequential_evaluation_proba(
        stream=stream,
        learner=model,
        window_size=window_size,
        max_instances=max_instances,
        store_predictions=False,
        store_y=False,
        optimise=optimise,
        progress_bar=progress_bar
    )

    # 6. Collect and save metrics
    base_name = os.path.splitext(os.path.basename(arff_file_path))[0]
    model_name = type(arf).__name__

    # cumulative metrics
    metrics = {
        "Learner": str(arf),
        "Stream": base_name,
        "Max Instances": results.max_instances(),
        "Wallclock Time (s)": results.wallclock(),
        "Cumulative Accuracy": results.cumulative.accuracy(),
        "Cumulative Precision": results.cumulative.precision(),
        "Cumulative Recall": results.cumulative.recall(),
        "Cumulative Kappa M": results.cumulative.kappa_m(),
        "Cumulative Kappa T": results.cumulative.kappa_t(),
    }
    df_metrics = pd.DataFrame([metrics])
    metrics_csv = os.path.join(output_dir, f"metrics_{model_name}_{base_name}.csv")
    df_metrics.to_csv(metrics_csv, index=False)

    # windowed metrics
    df_windows = results.metrics_per_window()
    windows_csv = os.path.join(output_dir, f"windows_{model_name}_{base_name}.csv")
    df_windows.to_csv(windows_csv, index=False)

    print(f"✅ Saved metrics to {metrics_csv} and {windows_csv}")


if __name__ == "__main__":
    data_dir = "./data"
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)

    dataset_files = [
        "AGR_a.arff", "AGR_g.arff", "HYPER.arff", "LED_a.arff", "LED_g.arff",
        "RBF_f.arff", "RBF_m.arff", "RTG.arff", "SEA_a.arff", "SEA_g.arff"
    ]

    for fname in dataset_files:
        file_path = os.path.join(data_dir, fname)
        if not os.path.isfile(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue
        run_evaluation(file_path, output_dir)
