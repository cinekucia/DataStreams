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
from river.tree import HoeffdingTreeClassifier
from river.utils.random import poisson

# ============================================================================
# == FULL BASE CLASS DEFINITIONS (Moved to global scope) =====================
# ============================================================================

def random_subspace(all_features: list, k: int, rng: random.Random):
    n_features = len(all_features); k = min(k, n_features)
    if n_features == 0 or k <= 0: return []
    return rng.sample(all_features, k=k)

class BaseSRPEstimator:
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
        self._background_learner: Optional['BaseSRPClassifier'] | Optional['BaseSRPRegressor'] = None
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
            "model": self.model.clone(), "metric": self.metric.clone(),
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
        k = len(self.features) if self.features is not None else 0
        subspace = None
        if self.features is not None and k > 0:
            subspace = random_subspace(all_features=all_features, k=k, rng=self.rng)
        elif self.features is None: subspace = None
        else: subspace = []

        self.model = self.model.clone() # type: ignore
        self.metric = self.metric.clone() # type: ignore
        self.created_on = n_samples_seen
        if self.drift_detector: self.drift_detector = self.drift_detector.clone() # type: ignore
        if self.warning_detector: self.warning_detector = self.warning_detector.clone() # type: ignore

        self.features = subspace
        self.n_drifts_detected = 0 ; self.n_warnings_detected = 0
        self._background_learner = None

    def predict_one(self, x:dict, **kwargs):
        return self.model.predict_one(x, **kwargs) if hasattr(self.model, 'predict_one') else None # type: ignore

    def predict_proba_one(self, x:dict, **kwargs):
        return self.model.predict_proba_one(x, **kwargs) if hasattr(self.model, 'predict_proba_one') else {} # type: ignore


class BaseSRPClassifier(BaseSRPEstimator):
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
        self.model: base.Classifier
        self.metric: ClassificationMetric
        self._background_learner : Optional[BaseSRPClassifier] = None


    def learn_one(
        self, x: dict, y: base.typing.ClfTarget, *, w: float,
        n_samples_seen: int, all_features: list, **kwargs,
    ):
        x_subset = x
        if self.features is not None and self.features:
            x_subset = {k_feat: x[k_feat] for k_feat in self.features if k_feat in x}
        if not x_subset and x:
             pass

        _predict_x = x_subset if self.features is not None and self.features and x_subset else x
        y_pred_for_drift = self.model.predict_one(_predict_x)

        correctly_classifies = (y_pred_for_drift == y) if y_pred_for_drift is not None else False

        learn_kwargs = kwargs.copy()
        model_params = self.model._get_params()
        supports_weight = 'weight' in model_params or 'sample_weight' in model_params or hasattr(self.model, '_supports_weights')
        if supports_weight: learn_kwargs['w'] = w

        if not x_subset and x:
            pass

        try:
            self.model.learn_one(x=x_subset, y=y, **learn_kwargs)
        except TypeError as e:
            if supports_weight and ('unexpected keyword argument' in str(e) or 'takes no keyword arguments' in str(e)):
                 learn_kwargs.pop('w', None); learn_kwargs.pop('sample_weight', None)
                 for _ in range(int(round(w))): self.model.learn_one(x=x_subset, y=y, **learn_kwargs)
            else:
                 if not supports_weight:
                      for _ in range(int(round(w))): self.model.learn_one(x=x_subset, y=y, **learn_kwargs)
                 else: raise e


        if self._background_learner:
            self._background_learner.learn_one(
                x=x, y=y, w=w, n_samples_seen=n_samples_seen,
                all_features=all_features, **kwargs,
            )

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

    def predict_proba_one(self, x, **kwargs):
        x_subset = x
        if self.features is not None and self.features:
            x_subset = {k_feat: x[k_feat] for k_feat in self.features if k_feat in x}
        if not x_subset and x:
            return {}

        if hasattr(self.model, 'predict_proba_one'):
            try: return self.model.predict_proba_one(x_subset, **kwargs)
            except Exception: return {}
        else:
            try:
                pred = self.model.predict_one(x_subset, **kwargs)
                return {pred: 1.0} if pred is not None else {}
            except Exception: return {}

    def predict_one(self, x: dict, **kwargs) -> base.typing.ClfTarget | None:
         y_proba = self.predict_proba_one(x, **kwargs)
         if not y_proba: return None
         try: return max(y_proba, key=y_proba.get)
         except ValueError: return None


class BaseSRPEnsemble(base.Wrapper, base.Ensemble):
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
        super().__init__([])
        self.model = model ; self.n_models = n_models
        self.subspace_size = subspace_size ; self.training_method = training_method
        self.lam = lam ; self.drift_detector = drift_detector ; self.warning_detector = warning_detector
        self.disable_weighted_vote = disable_weighted_vote ; self.disable_detector = disable_detector
        self.metric = metric ; self.seed = seed ; self._rng = random.Random(self.seed)
        self._n_samples_seen = 0 ; self._subspaces: list = [] ; self._features: list = []
        self._base_learner_class: type[BaseSRPClassifier] | type[BaseSRPRegressor] | None = None # type: ignore

    @property
    def _min_number_of_models(self): return 0 # THIS IS THE PROPERTY
    @property
    def _wrapped_model(self): return self.model
    @classmethod
    def _unit_test_params(cls): yield {"n_models": 3, "seed": 42}
    def _unit_test_skips(self): return {
        "check_shuffle_features_no_impact", "check_emerging_features", "check_disappearing_features",
    }

    def _generate_subspaces(self, features: list):
        n_features = len(features)
        if not features:
            self._subspaces = [[]] * self.n_models
            return

        self._features = features
        self._subspaces = [None] * self.n_models # type: ignore
        k=0

        if self.training_method != self._TRAIN_RESAMPLING:
            if isinstance(self.subspace_size, float) and 0.0 < self.subspace_size <= 1:
                k = round(n_features * self.subspace_size)
                if k == 0 and n_features > 0: k = 1
                if k < 2 and n_features >= 2: k = 2
                elif n_features < 2: k = n_features
            elif isinstance(self.subspace_size, int) and self.subspace_size >= 1:
                k = min(self.subspace_size, n_features)
            elif self.subspace_size == self._FEATURES_SQRT:
                k = round(math.sqrt(n_features))
                if n_features > 0 and k == 0 : k=1
            elif self.subspace_size == self._FEATURES_SQRT_INV:
                k = n_features - round(math.sqrt(n_features))
                if n_features > 0 and k == 0 : k=1
            else: raise ValueError(f"Invalid subspace_size: {self.subspace_size}")
            k = max(1 if n_features > 0 else 0, min(k, n_features))

            if k > 0 and k < n_features:
                try: num_combinations = math.comb(n_features, k)
                except ValueError: num_combinations = float('inf')

                if n_features <= 25 and num_combinations <= max(self.n_models * 10, 100):
                    all_combinations = list(itertools.combinations(features, k))
                    if len(all_combinations) < self.n_models:
                         indices = [i % len(all_combinations) for i in range(self.n_models)]
                         self._rng.shuffle(indices)
                         self._subspaces = [list(all_combinations[i]) for i in indices]
                    else:
                         selected_indices = self._rng.sample(range(len(all_combinations)), k=self.n_models)
                         self._subspaces = [list(all_combinations[i]) for i in selected_indices]
                else:
                    self._subspaces = [random_subspace(all_features=features, k=k, rng=self._rng) for _ in range(self.n_models)]
            elif k == n_features:
                self._subspaces = [features] * self.n_models
            elif k == 0 and n_features == 0:
                self._subspaces = [[]] * self.n_models
            else:
                self.training_method = self._TRAIN_RESAMPLING
                self._subspaces = [None] * self.n_models # type: ignore
        else:
            self._subspaces = [features] * self.n_models

    def _init_ensemble(self, features: list):
        self._generate_subspaces(features=features)
        subspace_indices = list(range(self.n_models))
        if (self.training_method == self._TRAIN_RANDOM_PATCHES or
            self.training_method == self._TRAIN_RANDOM_SUBSPACES):
            self._rng.shuffle(subspace_indices)

        self.data = []
        for i in range(self.n_models):
            current_subspace = self._subspaces[subspace_indices[i]]
            if self.training_method == self._TRAIN_RESAMPLING:
                 current_subspace = self._features if current_subspace is None else current_subspace


            if self.model is None or self.metric is None or self._base_learner_class is None:
                raise ValueError("Model, metric, or base_learner_class not properly initialized.")

            base_model_clone = self.model.clone()
            metric_clone = self.metric.clone()
            drift_detector_clone = self.drift_detector.clone() if self.drift_detector else None
            warning_detector_clone = self.warning_detector.clone() if self.warning_detector else None

            self.append(
                self._base_learner_class(
                    idx_original=i, model=base_model_clone, metric=metric_clone,
                    created_on=self._n_samples_seen, drift_detector=drift_detector_clone,
                    warning_detector=warning_detector_clone, is_background_learner=False,
                    rng=self._rng, features=current_subspace,
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

BaseSRPRegressor = BaseSRPEstimator


# ---------------------------------------------------------------------------
# SmartSRPClassifier with ARF-style Growth AND 0.9/1.1 Dynamic Weighting
# ---------------------------------------------------------------------------
class SmartSRPDW(BaseSRPEnsemble, base.Classifier):
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
        max_models: int = 30,
        accuracy_drop_threshold: float = 0.75,
        monitor_window: int = 200,
        **kwargs,
    ):
        if not isinstance(max_models, int) or max_models < 1: raise ValueError("max_models must be > 0.")
        if not isinstance(monitor_window, int) or monitor_window < 1: raise ValueError("monitor_window must be > 0.")
        if not isinstance(accuracy_drop_threshold, float) or not 0.0 < accuracy_drop_threshold <= 1.0:
             raise ValueError("accuracy_drop_threshold must be in (0, 1].")
        
        eff_model = model if model is not None else HoeffdingTreeClassifier(grace_period=50, delta=0.01, nominal_attributes=None)
        if not isinstance(eff_model, base.Classifier): raise ValueError("model must be a Classifier.")
        
        eff_metric = metric if metric is not None else Accuracy()
        if not isinstance(eff_metric, ClassificationMetric): raise ValueError("metric must be ClassificationMetric.")
        
        effective_drift_detector = drift_detector
        effective_warning_detector = warning_detector
        if disable_detector == "off":
             if effective_drift_detector is None: effective_drift_detector = ADWIN(delta=1e-5)
             if effective_warning_detector is None: effective_warning_detector = ADWIN(delta=1e-4)
        elif disable_detector == "drift": effective_drift_detector = None; effective_warning_detector = None
        elif disable_detector == "warning":
             if effective_drift_detector is None and drift_detector is None: effective_drift_detector = ADWIN(delta=1e-5) # type: ignore
             effective_warning_detector = None
        else: raise AttributeError(f"Invalid disable_detector option: {disable_detector}")

        super().__init__(
            model=eff_model, n_models=n_models, subspace_size=subspace_size, training_method=training_method,
            lam=lam, drift_detector=effective_drift_detector, warning_detector=effective_warning_detector,
            disable_detector=disable_detector, disable_weighted_vote=disable_weighted_vote,
            seed=seed, metric=eff_metric, **kwargs
        )

        self.max_models = max_models
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.monitor_window = monitor_window
        self.model_count_history: List[int] = []
        self._accuracy_window: List[collections.deque] = []
        self._warned_ids: Set[int] = set()
        self._warning_step: Dict[int, int] = {}
        self._warned_acc: Dict[int, float] = {}
        self._dynamic_perf_scores: List[float] = []
        self._base_learner_class = BaseSRPClassifier

    def _init_ensemble(self, features: list):
        super()._init_ensemble(features)

        self._accuracy_window = [collections.deque(maxlen=self.monitor_window) for _ in range(len(self))]
        
        if not self.disable_weighted_vote:
            self._dynamic_perf_scores = [1.0] * len(self)
            
        self._warned_ids.clear()
        self._warning_step.clear()
        self._warned_acc.clear()

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs):
        self._n_samples_seen += 1
        step = self._n_samples_seen

        if not self:
            current_features = list(x.keys())
            if not current_features and not self._features:
                print("Warning: No features available to initialize ensemble. Models may behave unexpectedly.")
                self._features = []
            elif not self._features:
                 self._features = current_features
            self._init_ensemble(self._features)

        if not self._features and x:
            self._features = list(x.keys())

        self.model_count_history.append(len(self))

        models_to_add: list[BaseSRPClassifier] = []
        indices_to_reset: list[int] = []
        indices_to_clear_background: list[int] = []
        indices_to_remove_on_error: list[int] = []
        
        for i, model_instance in list(enumerate(self)):
             if i >= len(self): continue

             y_pred = model_instance.predict_one(x)
             correct = 0
             if y_pred is not None:
                 correct = int(y_pred == y)
                 try: model_instance.metric.update(y_true=y, y_pred=y_pred)
                 except Exception: pass 
             if i < len(self._accuracy_window): self._accuracy_window[i].append(correct)

             if not self.disable_weighted_vote:
                 if i < len(self._dynamic_perf_scores):
                     self._dynamic_perf_scores[i] *= 0.9 if correct else 1.1
                 else:
                     print(f"Warning: Score update skipped for model {i}, _dynamic_perf_scores out of sync.")

             k = 1.0 if self.training_method == self._TRAIN_RANDOM_SUBSPACES else poisson(rate=self.lam, rng=self._rng)
             if k > 0:
                 try:
                     model_instance.learn_one(x=x, y=y, w=k, n_samples_seen=step, all_features=self._features, **kwargs)
                 except Exception as e:
                     print(f"Error during base learn_one for model {i} (orig_idx {model_instance.idx_original}): {e}. Marking for removal.")
                     indices_to_remove_on_error.append(i)
                     continue

             is_currently_warned = (not model_instance.disable_background_learner and
                                    model_instance.warning_detector is not None and
                                    model_instance.warning_detector.drift_detected)
             if is_currently_warned and i not in self._warned_ids and i < len(self._accuracy_window):
                  current_acc = self._get_recent_acc(i)
                  self._warned_ids.add(i)
                  self._warning_step[i] = step
                  self._warned_acc[i] = current_acc

             if not model_instance.disable_drift_detector and model_instance.drift_detector is not None:
                 drift_input_val = float(not correct)
                 drift_detected_before = model_instance.drift_detector.drift_detected
                 model_instance.drift_detector.update(drift_input_val)

                 if model_instance.drift_detector.drift_detected and not drift_detected_before:
                     print(f"🚩 Drift detected for model {i} (orig_idx {model_instance.idx_original}) at step {step}.")
                     if not model_instance.disable_background_learner and model_instance._background_learner is not None:
                         print(f"  -> Promoting background learner of model {i} to new model.")
                         if isinstance(model_instance._background_learner, BaseSRPClassifier):
                             models_to_add.append(model_instance._background_learner)
                         else:
                             print(f"  -> Error: Background learner of model {i} is not BaseSRPClassifier. Cannot promote.")
                         indices_to_clear_background.append(i)
                     else:
                         print(f"  -> Resetting model {i} in place.")
                         indices_to_reset.append(i)

        if indices_to_remove_on_error:
             indices_to_remove_on_error.sort(reverse=True)
             for idx_to_remove in indices_to_remove_on_error:
                  if 0 <= idx_to_remove < len(self):
                       self._remove_model(idx_to_remove)

        indices_to_reset_final = []
        temp_indices_to_reset = indices_to_reset[:] # Work on a copy
        for original_idx_target_val in temp_indices_to_reset:
            found_model_to_reset = False
            for current_idx, model_in_ensemble in enumerate(self):
                # This logic assumes that indices_to_reset contains original indices
                # and we need to find the model corresponding to that index IF IT STILL EXISTS
                # A better approach is to store (original_model_id, action)
                # For now, if the index is still valid and we assume no major reshuffles before this point:
                if original_idx_target_val < len(self): # If the original index is still somewhat valid
                    # And we assume the model at self[original_idx_target_val] is the one we intended to reset
                    indices_to_reset_final.append(original_idx_target_val) # Use the current index if it's plausible
                    found_model_to_reset = True
                    break # Assuming only one model matches this criteria or first one is fine
            # If not found by simple index, it implies model might have been removed or IDs are needed.
            # The current logic relies on index stability which is fragile after removals.
            # Let's simplify: if original_idx_target_val is a valid current index, use it.

        indices_to_reset_final = sorted(list(set(indices_to_reset_final)), reverse=True) # Unique and sorted
        for i_to_reset in indices_to_reset_final:
             if 0 <= i_to_reset < len(self):
                  print(f"Executing reset for model {i_to_reset} (orig_idx {self[i_to_reset].idx_original})")
                  self[i_to_reset].reset(all_features=self._features, n_samples_seen=step)
                  if not self.disable_weighted_vote and i_to_reset < len(self._dynamic_perf_scores):
                      self._dynamic_perf_scores[i_to_reset] = 1.0

        indices_to_clear_bg_final = []
        temp_indices_to_clear_bg = indices_to_clear_background[:]
        for original_idx_target_val in temp_indices_to_clear_bg:
            if original_idx_target_val < len(self): # If index is currently valid
                indices_to_clear_bg_final.append(original_idx_target_val)

        indices_to_clear_bg_final = sorted(list(set(indices_to_clear_bg_final)), reverse=True)
        for i_to_clear in indices_to_clear_bg_final:
            if 0 <= i_to_clear < len(self):
                print(f"Clearing background learner for model {i_to_clear} (orig_idx {self[i_to_clear].idx_original})")
                self[i_to_clear]._background_learner = None
                if self[i_to_clear].drift_detector:
                    self[i_to_clear].drift_detector = self[i_to_clear].drift_detector.clone() # type: ignore

        self._check_prune_accuracy_drop(step)

        for new_model_to_add in models_to_add:
            while len(self) >= self.max_models:
                if len(self) <= self._min_number_of_models : break # CORRECTED
                worst_idx = self._find_worst_model_idx()
                if worst_idx is None or not (0 <= worst_idx < len(self)):
                     print("Warning: Could not prune worst model to make space for new one.")
                     break 
                print(f"Capacity ({self.max_models}) reached. Pruning worst model {worst_idx} to add new one.")
                self._remove_model(worst_idx)

            if len(self) < self.max_models:
                 print(f"Adding new model (orig_idx {new_model_to_add.idx_original}). New size: {len(self) + 1}")
                 new_model_to_add.is_background_learner = False
                 new_model_to_add.created_on = step
                 if new_model_to_add.drift_detector: new_model_to_add.drift_detector = self.drift_detector.clone() if self.drift_detector else None # type: ignore
                 if new_model_to_add.warning_detector: new_model_to_add.warning_detector = self.warning_detector.clone() if self.warning_detector else None # type: ignore

                 self.append(new_model_to_add)
                 self._ensure_acc_window(len(self) - 1)
                 if not self.disable_weighted_vote:
                     if len(self._dynamic_perf_scores) == len(self) - 1:
                         self._dynamic_perf_scores.append(1.0)
                     else:
                         self._dynamic_perf_scores = [1.0] * len(self)
            else:
                 print("Warning: Could not add promoted background learner, max_models reached and pruning failed.")

        self._check_prune_max_models()
        return self

    def _ensure_acc_window(self, idx: int):
        while len(self._accuracy_window) <= idx:
            self._accuracy_window.append(collections.deque(maxlen=self.monitor_window))

    def _get_recent_acc(self, idx: int) -> float:
        if idx >= len(self._accuracy_window): return 0.0
        window = self._accuracy_window[idx]
        if not window: return 1.0
        return sum(window) / len(window)

    def _check_prune_accuracy_drop(self, current_step: int):
        models_to_prune: List[int] = []
        models_to_stop_monitoring: List[int] = []
        for model_idx in list(self._warned_ids):
            if model_idx >= len(self): models_to_stop_monitoring.append(model_idx); continue
            warning_start_step = self._warning_step.get(model_idx, current_step)
            age_since_warning = current_step - warning_start_step
            
            current_acc = self._get_recent_acc(model_idx)
            past_acc = self._warned_acc.get(model_idx, 1.0)

            if past_acc > 1e-9 and current_acc < (self.accuracy_drop_threshold * past_acc - 1e-9):
                print(f"📉 Pruning Model {model_idx} (Acc Drop: {past_acc:.3f} -> {current_acc:.3f}, Age: {age_since_warning})")
                models_to_prune.append(model_idx)
                models_to_stop_monitoring.append(model_idx)
            elif age_since_warning > self.monitor_window :
                models_to_stop_monitoring.append(model_idx)

        for model_idx in models_to_stop_monitoring:
            self._warned_ids.discard(model_idx)
            self._warning_step.pop(model_idx, None); self._warned_acc.pop(model_idx, None)
        
        if models_to_prune:
            unique_indices_to_prune = sorted(list(set(models_to_prune)), reverse=True)
            for model_idx in unique_indices_to_prune:
                 if 0 <= model_idx < len(self) and len(self) > self._min_number_of_models: # CORRECTED
                    self._remove_model(model_idx)

    def _check_prune_max_models(self):
        while len(self) > self.max_models:
            if len(self) <= self._min_number_of_models: break # CORRECTED
            worst_idx = self._find_worst_model_idx()
            if worst_idx is None or not (0 <= worst_idx < len(self)):
                 print("Warning: Cannot prune by max_models, failed to find worst model."); break
            print(f"📉 Pruning Model {worst_idx} (Max models exceeded: {len(self)} > {self.max_models})")
            self._remove_model(worst_idx)

    def _find_worst_model_idx(self) -> Optional[int]:
        if not self: return None
        if len(self) <= self._min_number_of_models: return None # CORRECTED

        scores = []; valid_indices = []; metric_bigger_is_better = True; has_metric_info = False
        for i, model_instance in enumerate(self):
            try:
                score = model_instance.metric.get()
                if not has_metric_info and hasattr(model_instance.metric, 'bigger_is_better'):
                    metric_bigger_is_better = model_instance.metric.bigger_is_better; has_metric_info = True
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
                min_acc_val = min(accuracies); worst_acc_index_in_list = accuracies.index(min_acc_val)
                return acc_valid_indices[worst_acc_index_in_list]
            else:
                  return len(self) - 1 if len(self) > self._min_number_of_models else None # CORRECTED
        
        min_score_val = min(scores); worst_score_index_in_list = scores.index(min_score_val)
        return valid_indices[worst_score_index_in_list]

    def _remove_model(self, index: int):
        if not (0 <= index < len(self)): return
        if len(self) <= self._min_number_of_models: return # CORRECTED

        del self[index] 
        if index < len(self._accuracy_window): del self._accuracy_window[index]
        
        if not self.disable_weighted_vote:
            if index < len(self._dynamic_perf_scores):
                del self._dynamic_perf_scores[index]
            elif len(self._dynamic_perf_scores) != len(self):
                self._dynamic_perf_scores = [1.0] * len(self)

        new_warned_ids = set(); new_warning_step = {}; new_warned_acc = {}
        for warned_idx in self._warned_ids:
            if warned_idx == index: continue
            new_idx = warned_idx - 1 if warned_idx > index else warned_idx
            if new_idx >= 0:
                new_warned_ids.add(new_idx)
                if warned_idx in self._warning_step: new_warning_step[new_idx] = self._warning_step[warned_idx]
                if warned_idx in self._warned_acc: new_warned_acc[new_idx] = self._warned_acc[warned_idx]
        self._warned_ids = new_warned_ids
        self._warning_step = new_warning_step
        self._warned_acc = new_warned_acc

    def reset(self):
        super().reset()
        self.model_count_history = []
        self._accuracy_window = []
        if not self.disable_weighted_vote:
            self._dynamic_perf_scores = []
        self._warned_ids.clear()
        self._warning_step.clear()
        self._warned_acc.clear()

    def predict_proba_one(self, x, **kwargs):
        y_pred_counter = collections.Counter()
        if not self: return {}

        final_weights: List[float] = []

        if self.disable_weighted_vote or len(self) == 0:
            if len(self) > 0: final_weights = [1.0 / len(self)] * len(self)
            else: final_weights = []
        else:
            if len(self._dynamic_perf_scores) != len(self):
                self._dynamic_perf_scores = [1.0] * len(self)

            raw_weights = [1.0 / (1.0 + score) for score in self._dynamic_perf_scores]
            sum_raw_weights = sum(raw_weights)
            if sum_raw_weights > 1e-9:
                final_weights = [w / sum_raw_weights for w in raw_weights]
            elif len(self) > 0:
                final_weights = [1.0 / len(self)] * len(self)
            else: final_weights = []
        
        if not final_weights and len(self) > 0:
            final_weights = [1.0 / len(self)] * len(self)

        total_effective_weight = 0.0
        for i, model_instance in enumerate(self):
            if i >= len(final_weights): continue

            y_proba_model = model_instance.predict_proba_one(x, **kwargs)
            weight = final_weights[i]

            if weight > 1e-9:
                for label, proba in y_proba_model.items():
                    y_pred_counter[label] += proba * weight
                total_effective_weight += weight
        
        if total_effective_weight > 1e-9:
            return {label: proba_sum / total_effective_weight for label, proba_sum in y_pred_counter.items()}
        elif len(y_pred_counter) > 0:
            return {label: 1.0 / len(y_pred_counter) for label in y_pred_counter}
        else: return {}

    def predict_one(self, x: dict, **kwargs) -> base.typing.ClfTarget | None:
         y_proba = self.predict_proba_one(x, **kwargs)
         if not y_proba: return None
         try: return max(y_proba, key=y_proba.get)
         except ValueError: return None