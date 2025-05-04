import math
import matplotlib.pyplot as plt # Keep if you plan to use model_count_history later

# River imports
from river import base, compose, forest, tree, metrics, drift, stats
from river import datasets, preprocessing # For the example usage
from river.utils.random import poisson # Correct location for poisson

class SmartARFv2(forest.ARFClassifier): # Renamed to avoid confusion
    """
    Smart Adaptive Random Forest Classifier (v2 - Add on Warning, Prune Periodically).

    Adaptation Strategy:
    1.  Immediate Addition on Warning: Adds a new base learner to the ensemble
        whenever an existing learner's warning detector triggers (up to max_models).
        If at max_models, replaces the warning learner instead.
    2.  Periodic Removal: Periodically removes learners whose recent performance
        falls below a threshold (respecting min_models).
    3.  Weighted Voting: Ensemble members vote based on their recent accuracy.

    Removed Features from previous/base ARF:
    - Standard drift detectors (_drift_detectors)
    - Background learners (_background)
    - Accuracy-drop pruning mechanism

    Parameters
    ----------
    n_models
        Initial number of models. Defaults to 10.
    max_models
        Maximum number of models allowed in the ensemble.
    min_models
        Minimum number of models to keep during periodic removal.
        If None, defaults to 1 (allowing shrinkage). It's recommended to set > 1.
    warning_detector
        Drift detector instance used to signal warnings within each base learner.
        Required for the primary adaptation mechanism.
    performance_threshold
        Minimum recent accuracy required for a model to survive periodic removal.
    monitor_window
        Size of the sliding window (number of instances) used to calculate recent accuracy.
    prune_period
        How often (in terms of number of learned instances) to perform periodic removal.
    kwargs
        Other parameters passed down to `forest.ARFClassifier` (e.g., seed,
        model_selector, leaf_prediction, etc.). Note: `drift_detector` is ignored.
    """
    def __init__(self,
                 n_models=10,
                 max_models=30,
                 min_models=1,          # Default min_models to 1
                 warning_detector: base.DriftDetector = None, # Make warning detector essential
                 performance_threshold=0.6,
                 monitor_window=100,
                 prune_period=500,
                 **kwargs):

        # Remove drift_detector from kwargs if present, as it's not used
        kwargs.pop('drift_detector', None)
        # Ensure warning_detector is provided
        if warning_detector is None:
            raise ValueError("SmartARFv2 requires a 'warning_detector'.")

        # Call the parent constructor
        super().__init__(n_models=n_models, warning_detector=warning_detector, drift_detector=None, **kwargs) # Explicitly set drift_detector=None

        # --- Store SmartARFv2 specific parameters ---
        resolved_min_models = 1 if min_models is None else min_models
        self.max_models = max(max_models, n_models)
        self.min_models = min(resolved_min_models, n_models)
        if self.min_models < 1: self.min_models = 1 # Ensure at least 1 model
        if self.min_models > self.max_models:
             raise ValueError(f"min_models ({self.min_models}) cannot be greater than max_models ({self.max_models})")

        self.performance_threshold = performance_threshold
        self.monitor_window = monitor_window
        self.prune_period = prune_period

        # --- Initialize SmartARFv2 specific state ---
        self.model_count_history = []
        # Accuracy window will be populated during _init_ensemble
        self._accuracy_window = []

        # Periodic pruning state
        self._steps_since_last_prune_check = 0
        self._learn_step = 0


    # Override _init_ensemble to add our custom state initialization
    def _init_ensemble(self, features):
        """ Initialize the ensemble, including parent logic and custom state. """
        super()._init_ensemble(features)
        if not hasattr(self, '_warning_detectors') or len(self._warning_detectors) != len(self.data):
             print("Warning: _warning_detectors list mismatch after super()._init_ensemble. Reinitializing.")
             self._warning_detectors = [self.warning_detector.clone() for _ in range(len(self.data))]
        num_models = len(self.data)
        self._accuracy_window = [[] for _ in range(num_models)]
        self._background = []
        self._drift_detectors = []


    def learn_one(self, x, y, **kwargs):
        """ Update the ensemble with a single observation (x, y). """

        if not self:
             features = list(x.keys()) if isinstance(x, dict) else None
             self._init_ensemble(features)

        current_step = self._learn_step
        self.model_count_history.append(len(self.data))

        models_to_add_warning = []
        indices_to_remove_warning = [] # Stores ORIGINAL indices to remove

        current_ensemble_indices = list(range(len(self.data)))

        for i in current_ensemble_indices:
            # Check if index 'i' is still valid relative to the CURRENT state of self.data
            # This check might be insufficient if removals happen mid-loop in a complex way.
            # Processing additions/removals AFTER the loop is safer.
            if i >= len(self.data): continue

            # --- Start of CORRECTED Indentation Block ---
            # Defensive checks for state list synchronization (INDENTED CORRECTLY)
            if i >= len(self._accuracy_window) or i >= len(self._metrics) or i >= len(self._warning_detectors):
                print(f"Warning: State list mismatch detected at index {i} in learn_one. Attempting sync.")
                self._sync_state_lists()
                if i >= len(self.data) or i >= len(self._accuracy_window): # Check again after sync
                     print(f"Error: Still mismatch after sync for index {i}. Skipping model.")
                     continue # Skip this model for this iteration
            # --- End of CORRECTED Indentation Block ---

            model = self.data[i] # Access model only after index check

            # 1. Get prediction, update metrics, update accuracy window
            y_pred = model.predict_one(x)
            if y_pred is not None:
                 self._metrics[i].update(y_true=y, y_pred=y_pred)
                 is_correct = int(y_pred == y)
            else:
                 is_correct = 0

            # Ensure accuracy window list is long enough before accessing index i
            self._ensure_accuracy_window_exists(i)
            self._accuracy_window[i].append(is_correct)
            if len(self._accuracy_window[i]) > self.monitor_window:
                self._accuracy_window[i].pop(0)

            # 2. Train the model
            k = poisson(rate=self.lambda_value, rng=self._rng)
            if k > 0:
                model.learn_one(x=x, y=y, w=k * kwargs.get('w', 1.0))

            # 3. Handle Warning Detector -> Add/Replace Immediately
            if y_pred is not None and i < len(self._warning_detectors) and self._warning_detectors[i] is not None: # Check index validity for detector
                 warn_detector = self._warning_detectors[i]
                 warn_detector.update(int(y_pred != y))
                 if warn_detector.drift_detected:
                     print(f"⚠️ Warning detected for model {i}. Triggering immediate action.")
                     self._warning_detectors[i] = self.warning_detector.clone() # Reset detector

                     if len(self.data) < self.max_models:
                         print(f"   -> Scheduling addition of new model (ensemble size {len(self.data)+1}/{self.max_models}).")
                         models_to_add_warning.append(self._new_base_model())
                     else:
                         print(f"   -> Scheduling replacement of model {i} (ensemble at max size {self.max_models}).")
                         # Mark ORIGINAL index 'i' for removal
                         if i not in indices_to_remove_warning: # Avoid duplicates if triggered twice somehow
                             indices_to_remove_warning.append(i)
                         # Schedule a new model to take its place
                         models_to_add_warning.append(self._new_base_model())


        # --- Process Additions/Removals from Warning Triggers (AFTER loop) ---
        indices_to_remove_warning.sort(reverse=True) # Highest index first
        num_removed_warn = 0
        for index in indices_to_remove_warning:
             if index < len(self.data): # Check validity based on current state
                 if self._remove_model_at_index(index):
                     num_removed_warn += 1
             else:
                 print(f"Warning: Index {index} for warning-based removal became invalid before processing.")

        num_added_warn = 0
        for new_model in models_to_add_warning:
             # Check max_models again before adding
             if len(self.data) < self.max_models:
                 self._add_new_model(new_model)
                 num_added_warn += 1
             else:
                 print(f"Warning: Could not add model from warning trigger, ensemble still full ({len(self.data)}/{self.max_models}). Replacement might have failed.")
                 break


        # --- Periodic REMOVAL Only ---
        self._steps_since_last_prune_check += 1
        if self._steps_since_last_prune_check >= self.prune_period:
             self._perform_periodic_removal()
             self._steps_since_last_prune_check = 0

        self._learn_step += 1
        return self

    # --- _get_recent_accuracy, _find_worst_model remain the same ---
    def _get_recent_accuracy(self, i):
        # Added bounds check within the function itself
        if i < 0 or i >= len(self._accuracy_window): return 0.0
        accs = self._accuracy_window[i]
        if not accs: return 1.0
        count = len(accs)
        total_correct = sum(accs)
        return total_correct / count if count > 0 else 1.0

    def _find_worst_model(self):
        if not self.data: return -1
        if len(self._accuracy_window) != len(self.data): self._sync_state_lists()
        if len(self._accuracy_window) != len(self.data): return -1 # Sync failed

        recent_accuracies = [self._get_recent_accuracy(i) for i in range(len(self.data))]
        if not recent_accuracies: return -1

        min_acc = min(recent_accuracies)
        worst_indices = [i for i, acc in enumerate(recent_accuracies) if acc == min_acc]
        # Return first worst model found
        return worst_indices[0] if worst_indices else -1

    # --- Periodic REMOVAL Only ---
    def _perform_periodic_removal(self):
        """ Periodically removes models below the performance threshold """
        if len(self.data) <= self.min_models: return

        indices_to_remove = []
        eligible_indices = list(range(len(self.data)))
        try:
            # Sort by accuracy, worst first
            eligible_indices.sort(key=lambda i: self._get_recent_accuracy(i))
        except IndexError:
            print("Warning: Index error during periodic removal sort.")
            self._sync_state_lists(); return

        for i in eligible_indices:
            if len(self.data) - len(indices_to_remove) <= self.min_models: break
            acc = self._get_recent_accuracy(i)
            if acc < self.performance_threshold:
                print(f"📉 Periodic Removal: Tree {i} below threshold ({acc:.2f} < {self.performance_threshold:.2f}). Scheduling removal.")
                indices_to_remove.append(i)
            else: break # Rest are better or equal

        num_removed = 0
        indices_to_remove.sort(reverse=True) # Highest index first
        if indices_to_remove:
             print(f"Periodic Removal: Attempting removal of {len(indices_to_remove)} models.")
             for index in indices_to_remove:
                 if index < len(self.data): # Check validity
                     if self._remove_model_at_index(index): num_removed += 1
                 else: print(f"Warning: Index {index} for periodic removal became invalid.")
             if num_removed > 0: print(f"Periodic Removal: Successfully removed {num_removed} models.")

    # --- Helper for Adding Models (Simplified) ---
    def _add_new_model(self, model_instance):
        """ Adds a new model instance and its associated state lists. """
        self.data.append(model_instance)
        self._metrics.append(self.metric.clone())
        self._warning_detectors.append(self.warning_detector.clone() if self.warning_detector else None)
        # Ensure accuracy window list exists and add an empty list for the new model
        self._ensure_accuracy_window_exists(len(self.data) - 1)


    # --- Helper for Removing Models (Simplified) ---
    def _remove_model_at_index(self, index):
        """ Removes a model and its associated state at a specific index. Returns True on success. """
        if index < 0 or index >= len(self.data): return False

        print(f"✂️ Removing tree at index {index} (Recent Acc: {self._get_recent_accuracy(index):.4f})")
        try:
            del self.data[index]
            del self._metrics[index]
            if index < len(self._warning_detectors): del self._warning_detectors[index]
            if index < len(self._accuracy_window): del self._accuracy_window[index]
            else: print(f"Warning: Accuracy window missing for index {index} during removal.")

        except IndexError:
            print(f"Warning: IndexError during removal of state for index {index}.")
            self._sync_state_lists(); return False # Indicate failure

        return True

    # --- _ensure_accuracy_window_exists, _sync_state_lists (Simplified) ---
    def _ensure_accuracy_window_exists(self, i):
        """ Ensures accuracy window list is long enough. """
        while len(self._accuracy_window) <= i:
            self._accuracy_window.append([])

    def _sync_state_lists(self):
        """ Ensures internal state lists match the number of models in self.data. """
        num_models = len(self.data)
        # Accuracy Window
        while len(self._accuracy_window) < num_models: self._accuracy_window.append([])
        self._accuracy_window = self._accuracy_window[:num_models]
        # Sync core lists
        core_lists = [self._metrics, self._warning_detectors]
        list_names = ["_metrics", "_warning_detectors"]
        for i, lst in enumerate(core_lists):
             current_len = len(lst)
             if current_len != num_models:
                 print(f"Warning: Syncing list {list_names[i]}. Length was {current_len}, expected {num_models}.")
                 # Simple truncation. Padding requires cloning detectors/metrics.
                 core_lists[i] = lst[:num_models]
                 # If list was too short, state is lost for newer models after sync.
                 # Re-initialization might be needed in severe cases.


    # --- Prediction methods remain the same (Weighted Voting) ---
    def predict_proba_one(self, x):
        """ Predict probabilities using weighted voting based on recent accuracy. """
        if not self.data: return {}
        y_pred_probas = {}
        total_weight = 0.0
        n_active_predictors = 0
        for i, model in enumerate(self.data):
            # Check index validity before accessing state
            if i >= len(self._accuracy_window):
                print(f"Warning: Skipping model {i} in predict_proba_one due to state mismatch.")
                continue
            weight = self._get_recent_accuracy(i)
            if weight > 0:
                predictions = model.predict_proba_one(x)
                if predictions:
                    n_active_predictors += 1
                    for label, proba in predictions.items():
                        y_pred_probas[label] = y_pred_probas.get(label, 0.0) + weight * proba
                    total_weight += weight

        if total_weight > 0:
            for label in y_pred_probas: y_pred_probas[label] /= total_weight
        elif n_active_predictors > 0:
            # Fallback logic (same as before)
            print("Warning: Fallback to equal weights in prediction (total weight was 0).")
            fallback_weight = 1.0 / n_active_predictors
            y_pred_probas = {}
            active_preds = 0
            for i, model in enumerate(self.data):
                 if i >= len(self._accuracy_window): continue # Check index again
                 predictions = model.predict_proba_one(x)
                 if predictions:
                     active_preds += 1
                     for label, proba in predictions.items():
                         y_pred_probas[label] = y_pred_probas.get(label, 0.0) + fallback_weight * proba
            if active_preds > 0:
                norm_sum = sum(y_pred_probas.values())
                if norm_sum > 0 :
                    for label in y_pred_probas: y_pred_probas[label] /= norm_sum
                else: return {}
            else: return {} # No models predicted in fallback
        elif not self.data: return {} # No models
        else: return {} # Models exist but none predicted or had weight
        return y_pred_probas

    def predict_one(self, x):
        """ Predict the class label using weighted voting. """
        probas = self.predict_proba_one(x)
        if not probas: return None
        return max(probas, key=probas.get)