from typing import Iterator, Tuple, Dict, Any, Union
from scipy.io.arff import loadarff
import numpy as np

def arff_to_river_stream(
    filepath: str,
    task: str = "classification",
    target: Union[int, str] = -1
) -> Iterator[Tuple[Dict[str, Any], Any]]:
    """
    Reads an ARFF file and yields data rows in the format expected by River:
    (feature_dict, target_label_or_value).

    Args:
        filepath: Path to the ARFF file.
        task: "classification" (default) or "regression".
        target: Which attribute to use as y:
          - int index into the ARFF’s attribute list (default -1, last attribute)
          - str name of the attribute

    Yields:
        A tuple (x, y) where
          x: dict of feature_name -> Python-native value
          y: float/int for regression, or str/int for classification

    Raises:
        FileNotFoundError: If the filepath does not exist.
        ValueError: If `task` is not "classification" or "regression",
                    or if `target` name is invalid.
        Exception: Propagates errors from loadarff.
    """
    if task not in ("classification", "regression"):
        raise ValueError(f"Invalid task '{task}'. Use 'classification' or 'regression'.")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data, meta = loadarff(f)
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        raise
    except Exception as e:
        print(f"Error loading ARFF file '{filepath}': {e}")
        raise

    feature_names = meta.names()

    # Resolve target attribute name
    if isinstance(target, int):
        target_idx = target
        # allow negative indexing
        try:
            target_name = feature_names[target_idx]
        except IndexError:
            raise ValueError(f"Target index {target_idx} out of range for attributes {feature_names}")
    else:
        if target not in feature_names:
            raise ValueError(f"Target name '{target}' not found in attributes {feature_names}")
        target_name = target

    input_feature_names = [n for n in feature_names if n != target_name]

    print(f"ARFF Reader: task='{task}', target='{target_name}'. "
          f"Features={input_feature_names}")

    for row in data:
        x: Dict[str, Any] = {}
        # ─── Process inputs ─────────────────────────────────────────────────────
        for name in input_feature_names:
            v = row[name]
            if isinstance(v, bytes):
                try:
                    x[name] = v.decode('utf-8')
                except UnicodeDecodeError:
                    print(f"Warning: Could not decode bytes for feature '{name}'.")
                    x[name] = v
            else:
                if isinstance(v, np.integer):
                    x[name] = int(v)
                elif isinstance(v, np.floating):
                    x[name] = None if np.isnan(v) else float(v)
                else:
                    x[name] = v

        # ─── Process target ─────────────────────────────────────────────────────
        y_raw = row[target_name]
        if task == "regression":
            if isinstance(y_raw, bytes):
                try:
                    y = float(y_raw.decode('utf-8'))
                except Exception:
                    y = None
            elif isinstance(y_raw, np.integer):
                y = int(y_raw)
            elif isinstance(y_raw, np.floating):
                y = None if np.isnan(y_raw) else float(y_raw)
            else:
                try:
                    y = float(y_raw)
                except Exception:
                    y = None
        else:  # classification
            if isinstance(y_raw, bytes):
                try:
                    y = y_raw.decode('utf-8')
                except UnicodeDecodeError:
                    print(f"Warning: Could not decode bytes for target '{target_name}'.")
                    y = y_raw
            elif isinstance(y_raw, (np.integer, np.floating)):
                if isinstance(y_raw, np.floating) and np.isnan(y_raw):
                    y = None
                else:
                    y = int(y_raw) if float(y_raw).is_integer() else float(y_raw)
            else:
                y = y_raw

        # ─── Yield or skip ──────────────────────────────────────────────────────
        if y is not None:
            yield x, y
        else:
            print(f"Skipping row due to missing target: {row}")
