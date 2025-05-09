import os
import csv
from itertools import tee
import matplotlib.pyplot as plt

from river import metrics, preprocessing, forest
from river.drift import ADWIN
from river.datasets import synth

csv_path = "regression_results_smart.csv"
fieldnames = ["dataset", "model", "MAE", "RMSE", "R2", "MAPE"]
if not os.path.exists(csv_path):
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

# --------------------------
# ARFF Export
# --------------------------
def save_stream_to_arff(stream, relation_name, output_file):
    stream = list(stream)
    if not stream:
        raise ValueError("Stream is empty")

    x_example, _ = stream[0]
    features = list(x_example.keys())
    labels = sorted(set(y for _, y in stream))

    with open(output_file, "w") as f:
        f.write(f"@RELATION {relation_name}\n\n")
        for feat in features:
            f.write(f"@ATTRIBUTE {feat} NUMERIC\n")
        f.write(f"@ATTRIBUTE class {{{', '.join(map(str, labels))}}}\n\n")
        f.write("@DATA\n")
        for x, y in stream:
            row = [str(x[feat]) for feat in features] + [str(y)]
            f.write(",".join(row) + "\n")
    print(f"✅ ARFF saved: {output_file}")

# --------------------------
# Evaluation Function
# --------------------------
def evaluate_smart_vs_standard(name, stream):
    stream_std, stream_smart = tee(stream, 2)

    # --- Models ---
    model_std = (
        preprocessing.StandardScaler() |
        forest.ARFRegressor(
            n_models=25,
            seed=42,
            lambda_value=6,
            grace_period=50,
            leaf_prediction="adaptive"
        )
    )

    model_smart = SmartARFRegressor(
        n_models=10,
        max_models=24,
        grace_period=50,
        lambda_value=6,
        seed=42,
        leaf_prediction="adaptive",
        metric=metrics.MAE(),
        drift_detector=ADWIN(delta=0.001),
        warning_detector=ADWIN(delta=0.01),
        disable_weighted_vote=False,  # Enable smart voting
        monitor_window=200,
        accuracy_drop_threshold=0.6,
        regression_pruning_error_threshold=0.2
    )

    # --- Metrics ---
    std_metrics = {
        "MAE": metrics.MAE(),
        "RMSE": metrics.RMSE(),
        "R2": metrics.R2(),
        "MAPE": metrics.MAPE()
    }
    smart_metrics = {
        "MAE": metrics.MAE(),
        "RMSE": metrics.RMSE(),
        "R2": metrics.R2(),
        "MAPE": metrics.MAPE()
    }

    # --- MAE curves ---
    std_mae, smart_mae = [], []

    for x, y in stream_std:
        y_pred = model_std.predict_one(x)
        if y_pred is not None:
            for m in std_metrics.values():
                m.update(y, y_pred)
        model_std.learn_one(x, y)
        std_mae.append(std_metrics["MAE"].get())

    for x, y in stream_smart:
        y_pred = model_smart.predict_one(x)
        if y_pred is not None:
            for m in smart_metrics.values():
                m.update(y, y_pred)
        model_smart.learn_one(x, y)
        smart_mae.append(smart_metrics["MAE"].get())

    # --- Output to console ---
    print(f"\n📊 [{name.upper()}]")
    print(f"Standard ARF: MAE={std_metrics['MAE'].get():.4f}, R2={std_metrics['R2'].get():.4f}")
    print(f"Smart    ARF: MAE={smart_metrics['MAE'].get():.4f}, R2={smart_metrics['R2'].get():.4f}")

    # --- Save to CSV ---
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({
            "dataset": name,
            "model": "Standard ARF",
            **{k: f"{m.get():.4f}" for k, m in std_metrics.items()}
        })
        writer.writerow({
            "dataset": name,
            "model": "Smart ARF",
            **{k: f"{m.get():.4f}" for k, m in smart_metrics.items()}
        })

    # --- Plot MAE ---
    plt.figure(figsize=(10, 5))
    plt.plot(std_mae, label="Standard ARF")
    plt.plot(smart_mae, label="Smart ARF")
    plt.title(f"MAE Comparison - {name}")
    plt.xlabel("Instances Seen")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Plot Model Count (Smart only) ---
    if hasattr(model_smart, "model_count_history"):
        model_smart.plot_model_count()

# --------------------------
# Dataset Setup
# --------------------------
regression_datasets = [
    ("friedman", synth.Friedman(seed=1).take(100000)),
    ("friedman_drift", synth.FriedmanDrift(seed=1).take(100000)),
    ("planes2d", synth.Planes2D(seed=1).take(100000))
]

output_dir = "arff_exports"
os.makedirs(output_dir, exist_ok=True)

# --------------------------
# Run Evaluation
# --------------------------
for name, stream in regression_datasets:
    stream_eval, stream_arff = tee(stream, 2)
    evaluate_smart_vs_standard(name, stream_eval)
    save_stream_to_arff(
        stream_arff,
        relation_name=name,
        output_file=os.path.join(output_dir, f"{name}_smart.arff")
    )

print(f"\n✅ All results saved to {csv_path}")
