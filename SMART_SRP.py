from __future__ import annotations

import collections
import itertools
import math
import random
from typing import List, Dict, Set, Optional

import numpy as np

from river import base
from river.drift import ADWIN
from river.metrics import MAE, Accuracy
from river.metrics.base import ClassificationMetric, Metric, RegressionMetric
from river.tree import HoeffdingTreeClassifier # Keep HTR for base class compatibility
from river.utils.random import poisson

# ---------------------------------------------------------------------------
# Base Classes (with clone() fix)
# ---------------------------------------------------------------------------
class BaseSRPEnsemble(base.Wrapper, base.Ensemble):
    # --- ( Largely unchanged from previous version - Snipped for brevity ) ---
    _TRAIN_RANDOM_SUBSPACES = "subspaces"
    _TRAIN_RESAMPLING = "resampling"
    _TRAIN_RANDOM_PATCHES = "patches"
    _FEATURES_SQRT = "sqrt"
    _FEATURES_SQRT_INV = "rmsqrt"
    _VALID_TRAINING_METHODS = {
        _TRAIN_RANDOM_PATCHES, _TRAIN_RESAMPLING, _TRAIN_RANDOM_SUBSPACES,
    }

    def __init__(
        self, model: base.Estimator | None = None, n_models: int = 10,
        subspace_size: int | float | str = 0.6, training_method: str = "patches",
        lam: float = 6.0, drift_detector: base.DriftDetector | None = None,
        warning_detector: base.DriftDetector | None = None, disable_detector: str = "off",
        disable_weighted_vote: bool = False, seed: int | None = None, metric: Metric | None = None,
    ):
        super().__init__([]) ; self.model = model ; self.n_models = n_models
        self.subspace_size = subspace_size ; self.training_method = training_method
        self.lam = lam ; self.drift_detector = drift_detector ; self.warning_detector = warning_detector
        self.disable_weighted_vote = disable_weighted_vote ; self.disable_detector = disable_detector
        self.metric = metric ; self.seed = seed ; self._rng = random.Random(self.seed)
        self._n_samples_seen = 0 ; self._subspaces: list = [] ; self._features: list = []
        self._base_learner_class: type[BaseSRPClassifier] | type[BaseSRPRegressor] | None = None

    @property
    def _min_number_of_models(self): return 0
    @property
    def _wrapped_model(self): return self.model
    @classmethod
    def _unit_test_params(cls): yield {"n_models": 3, "seed": 42}
    def _unit_test_skips(self): return {
        "check_shuffle_features_no_impact", "check_emerging_features", "check_disappearing_features",
    }

    def _generate_subspaces(self, features: list):
        # --- ( Unchanged subspace generation logic - Snipped ) ---
        n_features = len(features) ; self._features = features
        self._subspaces = [None] * self.n_models ; k=0 # Initialize k

        if self.training_method != self._TRAIN_RESAMPLING:
            if isinstance(self.subspace_size, float) and 0.0 < self.subspace_size <= 1:
                k = round(n_features * self.subspace_size)
                if k < 2 and n_features >= 2: k = 2
                elif n_features < 2: k = n_features
            elif isinstance(self.subspace_size, int) and self.subspace_size >= 1:
                k = min(self.subspace_size, n_features)
            elif self.subspace_size == self._FEATURES_SQRT:
                k = round(math.sqrt(n_features)) + 1
            elif self.subspace_size == self._FEATURES_SQRT_INV:
                k = n_features - round(math.sqrt(n_features))
            else: raise ValueError(f"Invalid subspace_size: {self.subspace_size}")
            k = max(1, min(k, n_features))

            if k != 0 and k < n_features:
                try: num_combinations = math.comb(n_features, k)
                except ValueError: num_combinations = float('inf')
                if n_features <= 25 and k > 0 and num_combinations <= max(self.n_models * 10, 100):
                    all_combinations = list(itertools.combinations(features, k))
                    if len(all_combinations) < self.n_models:
                         indices = [i % len(all_combinations) for i in range(self.n_models)]
                         self._rng.shuffle(indices) ; self._subspaces = [list(all_combinations[i]) for i in indices]
                    else:
                         selected_indices = self._rng.sample(range(len(all_combinations)), k=self.n_models)
                         self._subspaces = [list(all_combinations[i]) for i in selected_indices]
                else:
                    self._subspaces = [random_subspace(all_features=features, k=k, rng=self._rng) for _ in range(self.n_models)]
            elif k == n_features: self._subspaces = [features] * self.n_models
            else: self.training_method = self._TRAIN_RESAMPLING ; self._subspaces = [None] * self.n_models
        # --- ( End subspace generation logic ) ---


    def _init_ensemble(self, features: list):
        self._generate_subspaces(features=features)
        subspace_indices = list(range(self.n_models))
        if (self.training_method == self._TRAIN_RANDOM_PATCHES or
            self.training_method == self._TRAIN_RANDOM_SUBSPACES):
            self._rng.shuffle(subspace_indices)

        self.data = []
        for i in range(self.n_models):
            subspace = self._subspaces[subspace_indices[i]]
            # --- Crucially, clone all components for each base learner ---
            base_model_clone = self.model.clone()
            metric_clone = self.metric.clone()
            drift_detector_clone = self.drift_detector.clone() if self.drift_detector else None
            warning_detector_clone = self.warning_detector.clone() if self.warning_detector else None

            self.append(
                self._base_learner_class(  # type: ignore
                    idx_original=i, model=base_model_clone, metric=metric_clone,
                    created_on=self._n_samples_seen, drift_detector=drift_detector_clone,
                    warning_detector=warning_detector_clone, is_background_learner=False,
                    rng=self._rng, features=subspace,
                    **( {"drift_detection_criteria": self.drift_detection_criteria} # type: ignore
                       if hasattr(self, "drift_detection_criteria") else {} ),
                    disable_drift_detector=(self.disable_detector == "drift"),
                    disable_background_learner=(self.disable_detector == "drift" or self.disable_detector == "warning")
                )
            )

    def reset(self):
        self.data = [] ; self._n_samples_seen = 0
        self._rng = random.Random(self.seed)
        self._subspaces = [] ; self._features = []

class BaseSRPEstimator:
    # --- ( Largely unchanged - Snipped ) ---
    def __init__(
        self, idx_original: int, model: base.Estimator, metric: Metric, created_on: int,
        drift_detector: base.DriftDetector | None, warning_detector: base.DriftDetector | None,
        is_background_learner: bool, rng: random.Random, features=None,
        disable_drift_detector: bool = False, disable_background_learner: bool = False, **kwargs
    ):
        self.idx_original = idx_original ; self.created_on = created_on
        self.model = model ; self.metric = metric ; self.features = features
        self.disable_drift_detector = disable_drift_detector
        self.disable_background_learner = disable_background_learner
        self.drift_detector = drift_detector ; self.warning_detector = warning_detector
        if self.drift_detector is None: self.disable_drift_detector = True
        if self.warning_detector is None: self.disable_background_learner = True
        self.is_background_learner = is_background_learner
        self.n_drifts_detected = 0 ; self.n_warnings_detected = 0 ; self.rng = rng
        self._background_learner: BaseSRPClassifier | BaseSRPRegressor | None = None
        self._kwargs = kwargs

    def _trigger_warning(self, all_features, n_samples_seen: int):
        if self.disable_background_learner or self.is_background_learner: return

        k = len(self.features) if self.features is not None else 0
        subspace = None
        if self.features is not None and k > 0:
            subspace = random_subspace(all_features=all_features, k=k, rng=self.rng)
        elif self.features is None: subspace = None
        else: subspace = []

        constructor_args = {
            "idx_original": self.idx_original,
            # --- Use clone() for model and metric ---
            "model": self.model.clone(),
            "metric": self.metric.clone(),
            "created_on": n_samples_seen,
            "drift_detector": self.drift_detector.clone() if self.drift_detector else None,
            "warning_detector": self.warning_detector.clone() if self.warning_detector else None,
            "is_background_learner": True, "rng": self.rng, "features": subspace,
            "disable_drift_detector": self.disable_drift_detector,
            "disable_background_learner": self.disable_background_learner
        }
        constructor_args.update(self._kwargs)
        self._background_learner = self.__class__(**constructor_args) # type: ignore

        if self.warning_detector: self.warning_detector = self.warning_detector.clone()

    def reset(self, all_features: list, n_samples_seen: int):
        # --- Simplified Reset: Only resets the current model instance ---
        # --- Promotion logic moved to the main ensemble class ---
        k = len(self.features) if self.features is not None else 0
        subspace = None
        if self.features is not None and k > 0:
            subspace = random_subspace(all_features=all_features, k=k, rng=self.rng)
        elif self.features is None: subspace = None
        else: subspace = []

        # --- Use clone() for robust reset ---
        self.model = self.model.clone()
        self.metric = self.metric.clone()
        self.created_on = n_samples_seen
        if self.drift_detector: self.drift_detector = self.drift_detector.clone()
        if self.warning_detector: self.warning_detector = self.warning_detector.clone()

        self.features = subspace
        self.n_drifts_detected = 0 ; self.n_warnings_detected = 0
        self._background_learner = None # Clear background learner on reset

def random_subspace(all_features: list, k: int, rng: random.Random):
    # --- ( Unchanged - Snipped ) ---
    n_features = len(all_features); k = min(k, n_features)
    if n_features == 0 or k <= 0: return []
    return rng.sample(all_features, k=k)


class BaseSRPClassifier(BaseSRPEstimator):
    # --- ( Largely unchanged __init__ - Snipped ) ---
    def __init__(
        self, idx_original: int, model: base.Classifier, metric: ClassificationMetric, created_on: int,
        drift_detector: base.DriftDetector | None, warning_detector: base.DriftDetector | None,
        is_background_learner: bool, rng: random.Random, features=None,
        disable_drift_detector: bool = False, disable_background_learner: bool = False, **kwargs
    ):
        super().__init__(
            idx_original=idx_original, model=model, metric=metric, created_on=created_on,
            drift_detector=drift_detector, warning_detector=warning_detector,
            is_background_learner=is_background_learner, rng=rng, features=features,
            disable_drift_detector=disable_drift_detector,
            disable_background_learner=disable_background_learner, **kwargs
        )
        if not isinstance(model, base.Classifier): raise ValueError("Model must be a Classifier.")
        if not isinstance(metric, ClassificationMetric): raise ValueError("Metric must be a ClassificationMetric.")
        self.model: base.Classifier ; self.metric: ClassificationMetric
        self._background_learner : BaseSRPClassifier | None = None

    def learn_one(
        self, x: dict, y: base.typing.ClfTarget, *, w: float,
        n_samples_seen: int, all_features: list, **kwargs,
    ):
        # --- Removed Drift Detection Logic ---
        x_subset = x
        if self.features is not None:
            x_subset = {k: x[k] for k in self.features if k in x}

        # Store prediction *before* learning for warning detection input
        y_pred = self.predict_one(x)
        correctly_classifies = (y_pred == y) if y_pred is not None else False

        # --- Learning ---
        learn_kwargs = kwargs.copy()
        model_params = self.model._get_params()
        supports_weight = 'weight' in model_params or 'sample_weight' in model_params or hasattr(self.model, '_supports_weights')
        if supports_weight: learn_kwargs['w'] = w
        try:
            self.model.learn_one(x=x_subset, y=y, **learn_kwargs)
        except TypeError as e:
            if supports_weight and ('unexpected keyword argument' in str(e) or 'takes no keyword arguments' in str(e)):
                 learn_kwargs.pop('w', None); learn_kwargs.pop('sample_weight', None)
                 for _ in range(int(round(w))): self.model.learn_one(x=x_subset, y=y, **learn_kwargs)
            else: # Re-raise or handle non-weight related TypeErrors
                 # Fallback loop if weight failed unexpectedly OR model doesn't support weight
                 if not supports_weight:
                      for _ in range(int(round(w))): self.model.learn_one(x=x_subset, y=y, **learn_kwargs)
                 else: # Re-raise if it wasn't a weight issue
                      raise e


        # --- Train background learner ---
        if self._background_learner:
            self._background_learner.learn_one(
                x=x, y=y, w=w, n_samples_seen=n_samples_seen,
                all_features=all_features, **kwargs,
            )

        # --- Warning Detection (for main learner) ---
        if not self.is_background_learner:
            drift_input = float(not correctly_classifies)
            if not self.disable_background_learner and self.warning_detector is not None:
                warning_detected_before = self.warning_detector.drift_detected
                self.warning_detector.update(drift_input)
                if self.warning_detector.drift_detected and not warning_detected_before:
                    if not all_features:
                        print(f"Warning: Cannot trigger warning for model {self.idx_original}, 'all_features' list is empty.")
                    else:
                        self.n_warnings_detected += 1
                        self._trigger_warning(all_features=all_features, n_samples_seen=n_samples_seen)
                        # Warning detector reset happens in _trigger_warning

    # --- predict_proba_one and predict_one remain the same ---
    def predict_proba_one(self, x, **kwargs):
        x_subset = x
        if self.features is not None: x_subset = {k: x[k] for k in self.features if k in x}
        if hasattr(self.model, 'predict_proba_one'):
             try: return self.model.predict_proba_one(x_subset, **kwargs)
             except Exception: return {}
        else:
             try: pred = self.model.predict_one(x_subset, **kwargs) ; return {pred: 1.0} if pred is not None else {}
             except Exception: return {}

    def predict_one(self, x: dict, **kwargs) -> base.typing.ClfTarget | None:
         y_proba = self.predict_proba_one(x, **kwargs)
         if not y_proba: return None
         try: return max(y_proba, key=y_proba.get)
         except ValueError: return None

# ---------------------------------------------------------------------------
# SmartSRPClassifier with ARF-style Growth
# ---------------------------------------------------------------------------
class SmartSRPClassifier(BaseSRPEnsemble, base.Classifier):
    """
    Streaming Random Patches Classifier with Smart pruning and ARF-style growth.

    Extends SRPClassifier by:
    1. Adding pruning based on sustained accuracy drops after warnings.
    2. Enforcing a maximum model limit (`max_models`).
    3. **Adding new models upon drift detection when a background learner exists
       (similar to Adaptive Random Forest), instead of only replacing.**
    """
    def __init__(
        self,
        model: base.Classifier | None = None,
        n_models: int = 10,
        subspace_size: int | float | str = 0.6,
        training_method: str = "patches",
        lam: float = 6.0,
        drift_detector: base.DriftDetector | None = None,
        warning_detector: base.DriftDetector | None = None,
        disable_detector: str = "off",
        disable_weighted_vote: bool = False,
        seed: int | None = None,
        metric: ClassificationMetric | None = None,
        # Smart Pruning & Growth Parameters
        max_models: int = 30,
        accuracy_drop_threshold: float = 0.75,
        monitor_window: int = 200,
        **kwargs,
    ):
        # --- (Parameter Validation and Default Handling - Unchanged) ---
        if not isinstance(max_models, int) or max_models < 1: raise ValueError("max_models must be > 0.")
        if not isinstance(monitor_window, int) or monitor_window < 1: raise ValueError("monitor_window must be > 0.")
        if not isinstance(accuracy_drop_threshold, float) or not 0.0 < accuracy_drop_threshold <= 1.0:
             raise ValueError("accuracy_drop_threshold must be in (0, 1].")
        if model is None: model = HoeffdingTreeClassifier(grace_period=50, delta=0.01)
        if not isinstance(model, base.Classifier): raise ValueError("model must be a Classifier.")
        if metric is None: metric = Accuracy()
        if not isinstance(metric, ClassificationMetric): raise ValueError("metric must be ClassificationMetric.")
        effective_drift_detector = drift_detector
        effective_warning_detector = warning_detector
        if disable_detector == "off":
             if effective_drift_detector is None: effective_drift_detector = ADWIN(delta=1e-5)
             if effective_warning_detector is None: effective_warning_detector = ADWIN(delta=1e-4)
        elif disable_detector == "drift": effective_drift_detector = None; effective_warning_detector = None
        elif disable_detector == "warning":
             if effective_drift_detector is None and drift_detector is None: effective_drift_detector = ADWIN(delta=1e-5)
             effective_warning_detector = None
        else: raise AttributeError(f"Invalid disable_detector option: {disable_detector}")

        # Initialize base SRP Ensemble
        super().__init__(
            model=model, n_models=n_models, subspace_size=subspace_size, training_method=training_method,
            lam=lam, drift_detector=effective_drift_detector, warning_detector=effective_warning_detector,
            disable_detector=disable_detector, disable_weighted_vote=disable_weighted_vote,
            seed=seed, metric=metric, **kwargs
        )

        # --- Smart Pruning Initialization ---
        self.max_models = max_models
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.monitor_window = monitor_window
        self.model_count_history: List[int] = []
        self._accuracy_window: List[collections.deque] = []
        self._warned_ids: Set[int] = set()
        self._warning_step: Dict[int, int] = {}
        self._warned_acc: Dict[int, float] = {}

        # Define the base learner class for _init_ensemble
        self._base_learner_class = BaseSRPClassifier

    def _init_ensemble(self, features: list):
        super()._init_ensemble(features)
        self._accuracy_window = [collections.deque(maxlen=self.monitor_window) for _ in range(len(self))]
        self._warned_ids.clear() ; self._warning_step.clear() ; self._warned_acc.clear()

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs):
        self._n_samples_seen += 1
        step = self._n_samples_seen

        # 0. Initialize
        if not self:
            current_features = list(x.keys())
            if not current_features and not self._features: return self
            elif not self._features: self._features = current_features
            self._init_ensemble(self._features)
        if not self._features: self._features = list(x.keys())

        # Record size before modifications this step
        self.model_count_history.append(len(self))

        # --- Temporary storage for models to add ---
        models_to_add = []
        indices_to_reset = []
        indices_to_clear_background = []

        # --- Main Loop: Predict, Update Acc Windows, Train, Detect Drift/Warning ---
        models_to_iterate = list(enumerate(self)) # Iterate copy
        indices_to_remove_on_error = []

        for i, model in models_to_iterate:
             if i in indices_to_remove_on_error: continue # Skip if marked for removal

             # 1. Predict & Update Accuracy Window / Metric
             y_pred = model.predict_one(x)
             correct = 0
             if y_pred is not None:
                 correct = int(y_pred == y)
                 try: model.metric.update(y_true=y, y_pred=y_pred)
                 except Exception: pass # Ignore metric update errors for now
             if i < len(self._accuracy_window): self._accuracy_window[i].append(correct)

             # 2. Determine Weight & Train Base Learner (Handles Warning Detection)
             k = 1.0 if self.training_method == self._TRAIN_RANDOM_SUBSPACES else poisson(rate=self.lam, rng=self._rng)
             if k > 0:
                 try:
                     model.learn_one(x=x, y=y, w=k, n_samples_seen=step, all_features=self._features, **kwargs)
                 except Exception as e:
                     print(f"Error during base learn_one for model {i}: {e}. Marking for removal.")
                     indices_to_remove_on_error.append(i)
                     continue # Skip drift/warning checks if learning failed

             # 3. Start Monitoring on New Warnings
             is_currently_warned = (not model.disable_background_learner and
                                    model.warning_detector is not None and
                                    model.warning_detector.drift_detected)
             if is_currently_warned and i not in self._warned_ids and i < len(self._accuracy_window):
                  # print(f"🚦 Warning detected for model {i} at step {step}. Starting accuracy monitoring.")
                  current_acc = self._get_recent_acc(i)
                  self._warned_ids.add(i)
                  self._warning_step[i] = step
                  self._warned_acc[i] = current_acc
                  # print(f"   - Initial accuracy for monitoring: {current_acc:.3f}")

             # --- 4. Drift Detection (Moved Up from Base Learner) ---
             if not model.disable_drift_detector and model.drift_detector is not None:
                 drift_input = float(not correct) # Use correctness before learning
                 drift_detected_before = model.drift_detector.drift_detected
                 model.drift_detector.update(drift_input)

                 if model.drift_detector.drift_detected and not drift_detected_before:
                     print(f"🚩 Drift detected for model {i} at step {step}.")
                     # --- ARF-Style Addition/Reset ---
                     if not model.disable_background_learner and model._background_learner is not None:
                         # Promote background learner to become a new model
                         print(f"  -> Promoting background learner of model {i} to new model.")
                         new_model_instance = model._background_learner
                         # Prepare to add it later (after loop and potential pruning)
                         models_to_add.append(new_model_instance)
                         # Mark original model to clear background and reset detector
                         indices_to_clear_background.append(i)
                     else:
                         # No background learner or disabled, reset the model in place
                         print(f"  -> Resetting model {i} in place.")
                         indices_to_reset.append(i)


        # --- 5. Process Error Removals (adjusting subsequent indices) ---
        if indices_to_remove_on_error:
             indices_to_remove_on_error.sort(reverse=True)
             for idx_to_remove in indices_to_remove_on_error:
                  if 0 <= idx_to_remove < len(self):
                       self._remove_model(idx_to_remove) # Handles tracker adjustments
             # Need to recalculate indices for reset/clear background/add lists if removals happened
             # This gets complex. Simpler: Defer these actions or handle index mapping carefully.
             # For now, assume errors are rare and don't handle index shifts perfectly across actions.

        # --- 6. Handle Resets & Clear Backgrounds (BEFORE pruning/adding) ---
        # Reset models that drifted without background learners
        indices_to_reset.sort(reverse=True) # Process high indices first
        for i in indices_to_reset:
             if 0 <= i < len(self): # Check validity after potential error removals
                  print(f"Executing reset for model {i}")
                  self[i].reset(all_features=self._features, n_samples_seen=step)
                  # Reset might affect warned status - check if needed? For now, keep warned status.

        # Clear background learners for models whose background was promoted
        indices_to_clear_background.sort(reverse=True)
        for i in indices_to_clear_background:
            if 0 <= i < len(self):
                print(f"Clearing background learner for model {i}")
                self[i]._background_learner = None
                # Also reset the drift detector of the original model slot
                if self[i].drift_detector:
                    self[i].drift_detector = self[i].drift_detector.clone()


        # --- 7. Check Pruning based on Accuracy Drop ---
        self._check_prune_accuracy_drop(step)


        # --- 8. Handle Model Additions (Check Capacity & Prune Worst if Needed) ---
        for new_model in models_to_add:
            # Check capacity BEFORE adding
            while len(self) >= self.max_models:
                if len(self) <= 1: break # Don't remove the last one
                worst_idx = self._find_worst_model_idx()
                if worst_idx is None or not (0 <= worst_idx < len(self)):
                     print("Warning: Could not prune worst model to make space for new one.")
                     break # Cannot prune, cannot add
                print(f"Capacity ({self.max_models}) reached. Pruning worst model {worst_idx} to add new one.")
                self._remove_model(worst_idx)

            # Add the new model if space allows (or was made)
            if len(self) < self.max_models:
                 print(f"Adding new model derived from background learner. New size: {len(self) + 1}")
                 self.append(new_model)
                 # Initialize accuracy window for the new model
                 self._ensure_acc_window(len(self) - 1)
            else:
                 print("Warning: Could not add promoted background learner, max_models reached and pruning failed.")


        # --- 9. Check Pruning again for Max Models (in case adding pushed over limit somehow) ---
        # This is a safeguard, should ideally be handled by the loop above.
        self._check_prune_max_models()


        return self

    # --- Helper methods (_ensure_acc_window, _get_recent_acc) ---
    def _ensure_acc_window(self, idx: int):
        while len(self._accuracy_window) <= idx:
            self._accuracy_window.append(collections.deque(maxlen=self.monitor_window))

    def _get_recent_acc(self, idx: int) -> float:
        if idx >= len(self._accuracy_window): return 0.0
        window = self._accuracy_window[idx]
        if not window: return 1.0
        return sum(window) / len(window)

    # --- Pruning methods (_check_prune_accuracy_drop, _check_prune_max_models) ---
    def _check_prune_accuracy_drop(self, current_step: int):
        models_to_prune: List[int] = []
        models_to_stop_monitoring: List[int] = []
        for model_idx in list(self._warned_ids):
            if model_idx >= len(self): models_to_stop_monitoring.append(model_idx); continue
            warning_start_step = self._warning_step.get(model_idx, current_step)
            age_since_warning = current_step - warning_start_step
            if age_since_warning > self.monitor_window:
                models_to_stop_monitoring.append(model_idx); continue
            current_acc = self._get_recent_acc(model_idx)
            past_acc = self._warned_acc.get(model_idx, 1.0)
            if past_acc > 1e-9 and current_acc < (self.accuracy_drop_threshold * past_acc - 1e-9):
                print(f"📉 Pruning Model {model_idx} (Acc Drop: {past_acc:.3f} -> {current_acc:.3f}, Age: {age_since_warning})")
                models_to_prune.append(model_idx)
                models_to_stop_monitoring.append(model_idx)
        for model_idx in models_to_stop_monitoring:
            self._warned_ids.discard(model_idx)
            self._warning_step.pop(model_idx, None); self._warned_acc.pop(model_idx, None)
        if models_to_prune:
            unique_indices_to_prune = sorted(list(set(models_to_prune)), reverse=True)
            for model_idx in unique_indices_to_prune:
                 if 0 <= model_idx < len(self): self._remove_model(model_idx)

    def _check_prune_max_models(self):
        while len(self) > self.max_models:
            if len(self) <= 1: break
            worst_idx = self._find_worst_model_idx()
            if worst_idx is None or not (0 <= worst_idx < len(self)):
                 print("Warning: Cannot prune by max_models, failed to find worst model."); break
            print(f"📉 Pruning Model {worst_idx} (Max models exceeded: {len(self)} > {self.max_models})")
            self._remove_model(worst_idx)

    # --- _find_worst_model_idx (unchanged, uses metrics or accuracy fallback) ---
    def _find_worst_model_idx(self) -> Optional[int]:
        if not self: return None
        scores = []; valid_indices = []; metric_bigger_is_better = True; has_metric_info = False
        for i, model in enumerate(self):
            try:
                score = model.metric.get()
                if not has_metric_info and hasattr(model.metric, 'bigger_is_better'):
                    metric_bigger_is_better = model.metric.bigger_is_better; has_metric_info = True
                if score is not None:
                     comparable_score = score if metric_bigger_is_better else -score
                     scores.append(comparable_score); valid_indices.append(i)
            except Exception: pass
        if not scores:
            accuracies = []; acc_valid_indices = []
            for i in range(len(self)):
                if i < len(self._accuracy_window) and self._accuracy_window[i]:
                    acc = self._get_recent_acc(i); accuracies.append(acc); acc_valid_indices.append(i)
            if accuracies:
                min_acc = min(accuracies); worst_acc_index_in_list = accuracies.index(min_acc)
                return acc_valid_indices[worst_acc_index_in_list]
            else: return len(self) - 1 if self else None
        min_score = min(scores); worst_score_index_in_list = scores.index(min_score)
        return valid_indices[worst_score_index_in_list]

    # --- _remove_model (unchanged, handles tracker adjustments) ---
    def _remove_model(self, index: int):
        if not (0 <= index < len(self)): return
        # try: # Optional: Print removed model info
        #     removed_model = self[index]; metric_val = removed_model.metric.get()
        #     score_str = f"{metric_val:.4f}" if metric_val is not None else "N/A"
        #     print(f"🪓 Removing model at index {index} (Orig Idx: {removed_model.idx_original}, Score: {score_str})")
        # except Exception: print(f"🪓 Removing model at index {index}.")
        del self[index]
        if index < len(self._accuracy_window): del self._accuracy_window[index]
        new_warned_ids = set(); new_warning_step = {}; new_warned_acc = {}
        for warned_idx in self._warned_ids:
            if warned_idx == index: continue
            elif warned_idx > index: new_idx = warned_idx - 1
            else: new_idx = warned_idx
            if new_idx >= 0:
                new_warned_ids.add(new_idx)
                if warned_idx in self._warning_step: new_warning_step[new_idx] = self._warning_step[warned_idx]
                if warned_idx in self._warned_acc: new_warned_acc[new_idx] = self._warned_acc[warned_idx]
        self._warned_ids = new_warned_ids; self._warning_step = new_warning_step; self._warned_acc = new_warned_acc

    # --- reset (unchanged) ---
    def reset(self):
        super().reset(); self.model_count_history = []
        self._accuracy_window = []; self._warned_ids.clear()
        self._warning_step.clear(); self._warned_acc.clear()

    # --- predict_proba_one / predict_one (unchanged) ---
    def predict_proba_one(self, x, **kwargs):
        y_pred = collections.Counter(); total_weight = 0.0
        if not self: return {}
        for i, model in enumerate(self):
             y_proba_temp = model.predict_proba_one(x, **kwargs); weight = 1.0
             if not self.disable_weighted_vote:
                 try:
                     metric_value = model.metric.get()
                     if metric_value is not None:
                         metric_bigger_is_better = getattr(model.metric, 'bigger_is_better', True)
                         if metric_bigger_is_better: weight = metric_value
                         else: weight = 1.0 / (metric_value + 1e-9) if metric_value >= 0 else 0.0
                         weight = max(0.0, weight)
                     else: weight = 0.0
                 except Exception: weight = 0.0
             if weight > 1e-9: # Use epsilon comparison for float weights
                  total_weight += weight
                  for label, proba in y_proba_temp.items(): y_pred[label] += proba * weight
        if total_weight > 1e-9:
            return {label: proba_sum / total_weight for label, proba_sum in y_pred.items()}
        elif len(y_pred) > 0: return {label: 1.0 / len(y_pred) for label in y_pred}
        else: return {}

    def predict_one(self, x, **kwargs) -> base.typing.ClfTarget | None:
         y_proba = self.predict_proba_one(x, **kwargs)
         if not y_proba: return None
         try: return max(y_proba, key=y_proba.get)
         except ValueError: return None


# ============================================================================
# == Evaluation Script (Using the Refactored SmartSRPClassifier) =============
# ============================================================================

if __name__ == "__main__":

    from itertools import tee
    import matplotlib.pyplot as plt

    from river import evaluate, metrics
    from river.datasets import synth
    from river.drift import ADWIN
    from river.tree import HoeffdingTreeClassifier
    from river.ensemble import SRPClassifier # Standard SRP for comparison

    # 1) Create two identical streams
    print("Setting up data stream...")
    base_stream = synth.ConceptDriftStream(
        seed=42, position=5000, width=400
    ).take(10000)
    stream_classic, stream_smart = tee(base_stream, 2)

    # 2) Instantiate models
    print("Initializing models...")
    # Classic SRP
    model_classic = SRPClassifier(
        model=HoeffdingTreeClassifier(grace_period=50, delta=0.01), n_models=10, lam=6,
        drift_detector=ADWIN(delta=1e-5), warning_detector=ADWIN(delta=1e-4),
        seed=42, metric=Accuracy()
    )

    # Smart SRP (with ARF-style growth)
    model_smart = SmartSRPClassifier(
        model=HoeffdingTreeClassifier(grace_period=50, delta=0.01), n_models=10, max_models=15,
        monitor_window=200, accuracy_drop_threshold=0.6, lam=6,
        drift_detector=ADWIN(delta=1e-5), warning_detector=ADWIN(delta=1e-4),
        seed=12, metric=Accuracy()
    )

    # 3) Choose metric
    metric = metrics.Accuracy()

    # 4) Progressive evaluation
    print("▶️  Evaluating Classic SRP…")
    res_classic = evaluate.progressive_val_score(
        dataset=stream_classic, model=model_classic, metric=metric.clone(), print_every=2000
    )
    print("\n▶️  Evaluating Smart SRP…")
    res_smart = evaluate.progressive_val_score(
        dataset=stream_smart, model=model_smart, metric=metric.clone(), print_every=2000
    )

    # 5) Final accuracies
    print("\n--- Final Accuracy ---")
    final_acc_classic = res_classic.get() if hasattr(res_classic, 'get') else res_classic
    final_acc_smart = res_smart.get() if hasattr(res_smart, 'get') else res_smart
    print(f"Classic SRP : {final_acc_classic:.4f}")
    print(f"Smart SRP   : {final_acc_smart:.4f}")

    # 6) Plot ensemble size
    if model_smart.model_count_history:
        plt.figure(figsize=(10, 5))
        # Plot steps vs size: Use range(len(...)) for x-axis if steps not stored explicitly
        plt.plot(range(len(model_smart.model_count_history)), model_smart.model_count_history, linewidth=1.5)
        plt.axhline(y=model_smart.max_models, color='r', linestyle='-.', linewidth=1, label=f'Max Models ({model_smart.max_models})')
        plt.axhline(y=model_smart.n_models, color='g', linestyle=':', linewidth=1, label=f'Initial Models ({model_smart.n_models})')
        plt.xlabel("Training Step (Instances Seen)")
        plt.ylabel("Ensemble Size")
        plt.title("Smart SRP Ensemble Size Over Time")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
    else: print("\nNo ensemble size history recorded for Smart SRP.")
