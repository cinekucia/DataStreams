import os
import pandas as pd
from capymoa.stream import ARFFStream
from river.drift import ADWIN
from RiverWrapperClassifier import RiverWrapperClassifier
from Prequential_evaluation_proba import prequential_evaluation_proba
from river import tree, ensemble, metrics

from Selective_SRP import SelectiveSRPClassifier

def run_evaluation(arff_file_path: str, output_dir: str, \
                   window_size: int = 1000, max_instances: int = 1_000_000_00000,
                   optimise: bool = False, progress_bar: bool = True) -> pd.DataFrame:
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
        
    base_model = tree.HoeffdingTreeClassifier(
        grace_period=50,
        delta=0.01,
        nominal_attributes=nominal_attrs or None
    )

    river_model = SelectiveSRPClassifier(
        model=base_model,  # Pass the created base model here
        n_models=10,
        seed=42,
        prediction_confidence_threshold=0.1
    )
    print(f"Instantiated River model: {river_model}")

    model = RiverWrapperClassifier(river_model=river_model, schema=schema)

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
    
    model_name = "SelectiveSRP_t0.1"
    
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
    
    # <<< CHANGE 4: Update output directory to keep results separate
    output_dir = "./SelectiveSRP_t0.1_FINAL"
    
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
        # Update filename for the combined results
        combined_csv = os.path.join(output_dir, f"metrics_SelectiveSRP_t0.1_all_streams.csv")
        all_df.to_csv(combined_csv, index=False)
        print(f"✅ Saved combined cumulative metrics to {combined_csv}")
    else:
        print("No metrics to combine.")