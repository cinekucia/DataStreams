import numpy as np
import time # Add this if not already imported
from tqdm.auto import tqdm # Add this if not already imported
from capymoa.evaluation.evaluation import ( # Add these specific imports
    _setup_progress_bar,
    _is_fast_mode_compilable,
    _prequential_evaluation_fast,
    start_time_measuring,
    stop_time_measuring,
    ClassificationEvaluator,
    RegressionEvaluator,
    PredictionIntervalEvaluator,
    ClassificationWindowedEvaluator,
    RegressionWindowedEvaluator,
    PredictionIntervalWindowedEvaluator,
    PrequentialResults,
    MOAPredictionIntervalLearner # If you ever use prediction intervals
)
from capymoa.stream import Stream # Add Stream type import
from capymoa.base import Classifier, Regressor # Add base learner types
from typing import Union, Optional # Add typing imports

# --- Modified Evaluation Function ---

def prequential_evaluation_proba( # Renamed function
    stream: Stream,
    learner: Union[Classifier, Regressor],
    max_instances: Optional[int] = None,
    window_size: int = 1000,
    store_predictions: bool = False,
    store_y: bool = False,
    optimise: bool = True, # Keep the signature, but we know optimise=False for wrapper
    restart_stream: bool = True,
    progress_bar: Union[bool, tqdm] = False,
) -> PrequentialResults:
    """
    Run and evaluate a learner on a stream using prequential evaluation.
    MODIFIED to handle learners returning probability arrays from predict().
    """
    if restart_stream:
        stream.restart()

    # --- Optimization check remains, but will be false for our wrapper ---
    if optimise and _is_fast_mode_compilable(stream, learner, optimise):
        print("Warning: optimise=True requested, but wrapper likely incompatible. Using Python loop.")
        # Optionally fall through to Python loop or call the fast version if you know it *might* work
        # For safety with wrapper, stick to Python loop:
        # return _prequential_evaluation_fast(...)
        pass # Force Python loop below

    # --- Storage Initialization ---
    predictions_to_store = None
    if store_predictions:
        predictions_to_store = [] # Store the raw prediction (array or index)

    ground_truth_y_to_store = None
    if store_y:
        ground_truth_y_to_store = [] # Store the ground truth (index or value)

    # --- Start measuring time ---
    start_wallclock_time, start_cpu_time = start_time_measuring()

    # --- Evaluator Initialization ---
    evaluator_cumulative = None
    evaluator_windowed = None
    is_classification = stream.get_schema().is_classification()

    if is_classification:
        evaluator_cumulative = ClassificationEvaluator(
            schema=stream.get_schema() # No window_size needed for cumulative
        )
        if window_size is not None and window_size > 0 : # Check window_size > 0
            evaluator_windowed = ClassificationWindowedEvaluator(
                schema=stream.get_schema(), window_size=window_size
            )
    else: # Regression or PI
       # (Keep the original Regression/PI evaluator logic here if needed)
        if not isinstance(learner, MOAPredictionIntervalLearner):
             evaluator_cumulative = RegressionEvaluator(
                 schema=stream.get_schema()
             )
             if window_size is not None and window_size > 0:
                 evaluator_windowed = RegressionWindowedEvaluator(
                     schema=stream.get_schema(), window_size=window_size
                 )
        else:
             evaluator_cumulative = PredictionIntervalEvaluator(
                 schema=stream.get_schema()
             )
             if window_size is not None and window_size > 0:
                 evaluator_windowed = PredictionIntervalWindowedEvaluator(
                     schema=stream.get_schema(), window_size=window_size
                 )

    # --- Progress Bar Setup ---
    # Note: optimize=True is incompatible with progress bar in original code
    actual_progress_bar = _setup_progress_bar(
        "Eval", progress_bar, stream, learner, max_instances
    )

    # --- Main Evaluation Loop ---
    instances_processed = 0
    while stream.has_more_instances() and (max_instances is None or instances_processed < max_instances):
        instance = stream.next_instance()
        instances_processed += 1

        # --- Core Logic: Predict -> Evaluate -> Train ---
        # 1. Predict (Get probability array from our wrapper)
        prediction_output = learner.predict(instance) # This is the np.ndarray

        # 2. Get Ground Truth
        if is_classification:
            y_true = instance.y_index # Get the true class index
        else:
            y_true = instance.y_value # Get the true regression value

        # 3. *** MODIFICATION: Convert prediction for evaluator ***
        if is_classification:
            if isinstance(prediction_output, np.ndarray):
                # If it's an array, find the index with the highest probability
                y_pred_eval = np.argmax(prediction_output)
            elif isinstance(prediction_output, (int, np.integer)):
                 # If learner already returned index (unlikely for our wrapper)
                 y_pred_eval = prediction_output
            else:
                 # Handle unexpected prediction type if necessary
                 print(f"Warning: Unexpected prediction type {type(prediction_output)}. Setting prediction to None for evaluation.")
                 y_pred_eval = None
        else: # Regression
            y_pred_eval = prediction_output # Pass regression value directly


        # 4. Update Evaluators (using the converted prediction index for classification)
        if evaluator_cumulative:
             evaluator_cumulative.update(y_true, y_pred_eval)
        if evaluator_windowed:
             evaluator_windowed.update(y_true, y_pred_eval)

        # 5. Train the learner
        learner.train(instance)
        # --- End Core Logic ---

        # --- Store results if requested ---
        if predictions_to_store is not None:
            # Store the raw output of predict (the probability array)
            predictions_to_store.append(prediction_output)

        if ground_truth_y_to_store is not None:
            # Store the ground truth index/value
            ground_truth_y_to_store.append(y_true)

        # --- Update progress bar ---
        if actual_progress_bar is not None:
            actual_progress_bar.update(1)

    # --- Cleanup ---
    if actual_progress_bar is not None:
        actual_progress_bar.close()

    # --- Stop measuring time ---
    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(
        start_wallclock_time, start_cpu_time
    )

    # --- Finalize windowed results ---
    if (
        evaluator_windowed is not None
        and evaluator_windowed.get_instances_seen() > 0 # Check if any instances were seen
        and window_size is not None # Ensure window_size is valid
        and evaluator_windowed.get_instances_seen() % window_size != 0
    ):
        evaluator_windowed.result_windows.append(evaluator_windowed.metrics())

    # --- Package Results ---
    results = PrequentialResults(
        learner=str(learner), # Use the learner's __str__
        stream=str(stream),  # Use the stream's __str__
        wallclock=elapsed_wallclock_time,
        cpu_time=elapsed_cpu_time,
        max_instances=max_instances,
        cumulative_evaluator=evaluator_cumulative,
        windowed_evaluator=evaluator_windowed,
        ground_truth_y=ground_truth_y_to_store,
        predictions=predictions_to_store,
    )

    return results
