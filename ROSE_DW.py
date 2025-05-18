# Import necessary components
from __future__ import annotations
import abc
import collections
import itertools
import math
import random
import typing
import numpy as np

from river import base, metrics, stats
from river.drift import ADWIN, NoDrift
from river.tree import HoeffdingTreeClassifier
from river.metrics.base import ClassificationMetric, Metric, RegressionMetric
from river.utils.random import poisson
from river.utils.rolling import Rolling
# Using river.metrics.Accuracy directly, aliased as AccMetric
from river.metrics import Accuracy as AccMetric


# ================================================================
# BASE SRP CLASSES (Ensure BaseSRPEstimator.reset is robust)
# ================================================================
class BaseSRPEnsemble(base.Wrapper, base.Ensemble):
    _TRAIN_RANDOM_SUBSPACES = "subspaces"; _TRAIN_RESAMPLING = "resampling"; _TRAIN_RANDOM_PATCHES = "patches"
    _FEATURES_SQRT = "sqrt"; _FEATURES_SQRT_INV = "rmsqrt"
    _VALID_TRAINING_METHODS = {_TRAIN_RANDOM_PATCHES, _TRAIN_RESAMPLING, _TRAIN_RANDOM_SUBSPACES}
    def __init__(self, model: base.Estimator | None = None, n_models: int = 100,
        subspace_size: int | float | str = 0.6, training_method: str = "patches",
        lam: float = 6.0, drift_detector: base.DriftDetector | None = None,
        warning_detector: base.DriftDetector | None = None, disable_detector: str = "off",
        disable_weighted_vote: bool = False, seed: int | None = None, metric: Metric | None = None):
        super().__init__([])
        self.model = model; self.n_models = n_models; self.subspace_size = subspace_size
        self.training_method = training_method; self.lam = lam
        self.drift_detector = drift_detector; self.warning_detector = warning_detector
        self.disable_weighted_vote = disable_weighted_vote; self.disable_detector = disable_detector
        self.metric = metric; self.seed = seed; self._rng = random.Random(self.seed)
        self._n_samples_seen = 0; self._subspaces: list = []; self._base_learner_class: typing.Any = None
    @property
    def _min_number_of_models(self): return 0
    @property
    def _wrapped_model(self): return self.model
    @classmethod
    def _unit_test_params(cls): yield {"n_models": 3, "seed": 42}
    def _unit_test_skips(self): return {"check_shuffle_features_no_impact", "check_emerging_features", "check_disappearing_features"}
    def learn_one(self, x: dict, y: base.typing.Target, **kwargs):
        self._n_samples_seen += 1
        if not self: self._init_ensemble(list(x.keys()))
        for model_wrapper in self:
            y_pred = model_wrapper.predict_one(x)
            if y_pred is not None and hasattr(model_wrapper, 'metric') and model_wrapper.metric is not None:
                model_wrapper.metric.update(y_true=y, y_pred=y_pred)
            if self.training_method == self._TRAIN_RANDOM_SUBSPACES: k = 1
            else:
                k = poisson(rate=self.lam, rng=self._rng)
                if k == 0: continue
            model_wrapper.learn_one(x=x, y=y, w=k, n_samples_seen=self._n_samples_seen, **kwargs)
    def _generate_subspaces(self, features: list):
        n_features = len(features); self._subspaces = [None] * self.n_models
        if self.training_method != self._TRAIN_RESAMPLING:
            k: int | float
            if isinstance(self.subspace_size, float) and 0.0 < self.subspace_size <= 1:
                k = self.subspace_size; percent = (1.0 + k) / 1.0 if k < 0 else k; k = round(n_features * percent)
                if k < 2: k = round(n_features * percent) + 1
            elif isinstance(self.subspace_size, int) and self.subspace_size >= 2: k = self.subspace_size
            elif self.subspace_size == self._FEATURES_SQRT: k = round(math.sqrt(n_features)) + 1
            elif self.subspace_size == self._FEATURES_SQRT_INV: k = n_features - round(math.sqrt(n_features)) + 1
            else: raise ValueError(f"Invalid subspace_size: {self.subspace_size}...")
            if k < 0: k = n_features + k
            if k <= 0: k = 1
            if k > n_features: k = n_features
            if k != 0 and k < n_features:
                if n_features <= 20 or k < 2:
                    if k == 1 and n_features > 2: k = 2
                    self._subspaces = []
                    for i, combination in enumerate(itertools.cycle(itertools.combinations(features, k))):
                        if i == self.n_models: break
                        self._subspaces.append(list(combination))
                else: self._subspaces = [random_subspace(all_features=features, k=k, rng=self._rng) for _ in range(self.n_models)]
            else: self.training_method = self._TRAIN_RESAMPLING
    def _init_ensemble(self, features: list):
        self._generate_subspaces(features=features)
        subspace_indexes = list(range(self.n_models))
        if self.training_method == self._TRAIN_RANDOM_PATCHES or self.training_method == self._TRAIN_RANDOM_SUBSPACES:
            self._rng.shuffle(subspace_indexes)
        self.data = []
        eval_window_size = getattr(self, 'evaluator_window_size', getattr(self, 'rose_window_size', 500))
        for i in range(self.n_models):
            subspace = self._subspaces[subspace_indexes[i]] if self._subspaces and i < len(self._subspaces) else None
            self.append(self._base_learner_class(idx_original=i, model_prototype=self.model, metric_prototype=self.metric,
                created_on=self._n_samples_seen, drift_detector_prototype=self.drift_detector,
                warning_detector_prototype=self.warning_detector, is_background_learner=False,
                rng=self._rng, features=subspace, evaluator_window_size=eval_window_size))
    def reset(self): self.data = []; self._n_samples_seen = 0; self._rng = random.Random(self.seed)

class BaseSRPEstimator:
    def __init__(self, idx_original: int, model_prototype: base.Estimator, metric_prototype: Metric, created_on: int,
        drift_detector_prototype: base.DriftDetector | None, warning_detector_prototype: base.DriftDetector | None,
        is_background_learner, rng: random.Random, features=None, evaluator_window_size: int = 500):
        self.idx_original = idx_original; self.created_on = created_on
        self._model_prototype = model_prototype; self._metric_prototype = metric_prototype
        self._drift_detector_prototype = drift_detector_prototype; self._warning_detector_prototype = warning_detector_prototype
        self.model = self._model_prototype.clone(); self.metric = self._metric_prototype.clone() if self._metric_prototype else None
        self.features = features
        self.disable_drift_detector = self._drift_detector_prototype is None
        self.drift_detector = self._drift_detector_prototype.clone() if self._drift_detector_prototype else None
        self.disable_background_learner = self._warning_detector_prototype is None
        self.warning_detector = self._warning_detector_prototype.clone() if self._warning_detector_prototype else None
        self.evaluator_window_size = evaluator_window_size # Used for rolling metrics
        self.rolling_accuracy = Rolling(AccMetric(), window_size=evaluator_window_size) # For ROSE-like evaluation
        self.is_background_learner = is_background_learner; self.n_drifts_detected = 0; self.n_warnings_detected = 0
        self.rng = rng; self._background_learner: typing.Any = None
    def _trigger_warning(self, all_features, n_samples_seen: int):
        subspace = None
        if self.features is not None:
            k_original_subspace = len(self.features)
            subspace = random_subspace(all_features=all_features, k=k_original_subspace, rng=self.rng)
        self._background_learner = self.__class__(idx_original=self.idx_original, model_prototype=self._model_prototype,
            metric_prototype=self._metric_prototype, created_on=n_samples_seen,
            drift_detector_prototype=self._drift_detector_prototype, warning_detector_prototype=self._warning_detector_prototype,
            is_background_learner=True, rng=self.rng, features=subspace, evaluator_window_size=self.evaluator_window_size)
        if self.warning_detector: self.warning_detector = self._warning_detector_prototype.clone() if self._warning_detector_prototype else None
    def reset(self, all_features: list, n_samples_seen: int):
        if not self.disable_background_learner and self._background_learner is not None:
            self.model = self._background_learner.model; self.metric = self._background_learner.metric
            self.drift_detector = self._background_learner.drift_detector; self.warning_detector = self._background_learner.warning_detector
            self.created_on = self._background_learner.created_on; self.features = self._background_learner.features
            self.n_drifts_detected = self._background_learner.n_drifts_detected; self.n_warnings_detected = self._background_learner.n_warnings_detected
            self.rolling_accuracy = self._background_learner.rolling_accuracy
            self._background_learner = None
        else:
            new_subspace = None
            if self.features is not None:
                k_original_subspace = len(self.features)
                new_subspace = random_subspace(all_features=all_features, k=k_original_subspace, rng=self.rng)
            self.model = self._model_prototype.clone(); self.metric = self._metric_prototype.clone() if self._metric_prototype else None
            self.created_on = n_samples_seen
            self.drift_detector = self._drift_detector_prototype.clone() if self._drift_detector_prototype else None
            self.warning_detector = self._warning_detector_prototype.clone() if self._warning_detector_prototype else None
            self.features = new_subspace; self.n_drifts_detected = 0; self.n_warnings_detected = 0
            self.rolling_accuracy = Rolling(AccMetric(), window_size=self.evaluator_window_size)

def random_subspace(all_features: list, k: int, rng: random.Random):
    corrected_k = min(len(all_features), k); return rng.sample(all_features, k=corrected_k)

class SRPClassifier(BaseSRPEnsemble, base.Classifier):
    def __init__(self, model: base.Estimator | None = None, n_models: int = 10,
        subspace_size: int | float | str = 0.6, training_method: str = "patches", lam: int = 6,
        drift_detector: base.DriftDetector | None = None, warning_detector: base.DriftDetector | None = None,
        disable_detector: str = "off", disable_weighted_vote: bool = False, seed: int | None = None,
        metric: ClassificationMetric | None = None, evaluator_window_size: int = 500):
        drift_detector_proto = drift_detector; warning_detector_proto = warning_detector; metric_proto = metric
        if model is None: model = HoeffdingTreeClassifier(grace_period=50, delta=0.01)
        if drift_detector_proto is None and disable_detector != "drift": drift_detector_proto = ADWIN(delta=1e-5)
        if warning_detector_proto is None and disable_detector == "off": warning_detector_proto = ADWIN(delta=1e-4)
        if disable_detector == "drift": drift_detector_proto = None; warning_detector_proto = None
        elif disable_detector == "warning": warning_detector_proto = None
        elif disable_detector != "off": raise AttributeError(f"{disable_detector} is not a valid...")
        if metric_proto is None: metric_proto = AccMetric()
        super().__init__(model=model, n_models=n_models, subspace_size=subspace_size, training_method=training_method,
            lam=lam, drift_detector=drift_detector_proto, warning_detector=warning_detector_proto,
            disable_detector=disable_detector, disable_weighted_vote=disable_weighted_vote, seed=seed, metric=metric_proto)
        self._base_learner_class = BaseSRPClassifier; self.evaluator_window_size = evaluator_window_size
    def predict_proba_one(self, x, **kwargs):
        y_pred = collections.Counter()
        if not self.models: self._init_ensemble(features=list(x.keys())); return y_pred
        for model_wrapper in self.models:
            y_proba_temp = model_wrapper.predict_proba_one(x, **kwargs)
            metric_val = model_wrapper.metric.get() if hasattr(model_wrapper, 'metric') and model_wrapper.metric is not None else None
            if not self.disable_weighted_vote and metric_val is not None and metric_val > 0.0:
                y_proba_temp = {k_l: val * metric_val for k_l, val in y_proba_temp.items()}
            y_pred.update(y_proba_temp)
        total = sum(y_pred.values())
        if total > 0: return {label: proba / total for label, proba in y_pred.items()}
        return y_pred
    @property
    def _multiclass(self): return True

class BaseSRPClassifier(BaseSRPEstimator):
    def __init__(self, idx_original: int, model_prototype: base.Classifier, metric_prototype: ClassificationMetric,
        created_on: int, drift_detector_prototype: base.DriftDetector | None, warning_detector_prototype: base.DriftDetector | None,
        is_background_learner, rng: random.Random, features=None, evaluator_window_size: int = 500):
        super().__init__(idx_original=idx_original, model_prototype=model_prototype, metric_prototype=metric_prototype,
            created_on=created_on, drift_detector_prototype=drift_detector_prototype,
            warning_detector_prototype=warning_detector_prototype, is_background_learner=is_background_learner,
            rng=rng, features=features, evaluator_window_size=evaluator_window_size)
    def learn_one(self, x: dict, y: base.typing.ClfTarget, *, w: int, n_samples_seen: int, **kwargs):
        all_features_for_reset = kwargs.pop('all_features', list(x.keys()))
        x_subset = {k: x[k] for k in self.features if k in x} if self.features is not None else x
        for _ in range(int(w)): self.model.learn_one(x=x_subset, y=y, **kwargs) # type: ignore
        if self._background_learner:
            self._background_learner.learn_one(x=x, y=y, w=w, n_samples_seen=n_samples_seen, all_features=all_features_for_reset, **kwargs)
        if not self.disable_drift_detector and not self.is_background_learner:
            prediction_for_drift = self.model.predict_one(x_subset) # type: ignore
            if prediction_for_drift is None: return
            correctly_classifies = prediction_for_drift == y
            if self.metric is not None: self.metric.update(y_true=y, y_pred=prediction_for_drift) # type: ignore
            # Update rolling accuracy for ROSE-like mechanisms
            self.rolling_accuracy.update(y_true=y, y_pred=prediction_for_drift)

            if not self.disable_background_learner and self.warning_detector:
                self.warning_detector.update(int(not correctly_classifies))
                if self.warning_detector.drift_detected:
                    self.n_warnings_detected += 1; self._trigger_warning(all_features=all_features_for_reset, n_samples_seen=n_samples_seen)
            if self.drift_detector:
                self.drift_detector.update(int(not correctly_classifies))
                if self.drift_detector.drift_detected:
                    self.n_drifts_detected += 1; self.reset(all_features=all_features_for_reset, n_samples_seen=n_samples_seen)
    def predict_proba_one(self, x, **kwargs):
        x_subset = {k: x[k] for k in self.features if k in x} if self.features is not None else x
        return self.model.predict_proba_one(x_subset, **kwargs) # type: ignore
    def predict_one(self, x: dict, **kwargs) -> base.typing.ClfTarget:
        y_pred_proba = self.predict_proba_one(x, **kwargs)
        if y_pred_proba: return max(y_pred_proba, key=y_pred_proba.get) # type: ignore
        return None

# =======================================================================
# SRPWiseRoseRiver Classifier (Merging StreamWiseRandomPatches and ROSE)
# =======================================================================

class InstanceWithTimestamp:
    def __init__(self, instance_data: dict, target: base.typing.Target, timestamp: int):
        self.x = instance_data; self.y = target; self.timestamp = timestamp
    def __lt__(self, other): return self.timestamp < other.timestamp

class BaseSRPWiseRoseLearner(BaseSRPClassifier): # Base learner for the merged algo
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Rolling accuracy is already in BaseSRPEstimator

class SRPWiseRoseRiver(SRPClassifier):
    def __init__(
        self,
        model: base.Estimator | None = None,
        n_models: int = 10, # ensemble_size from StreamWise, ensembleSizeOption from ROSE
        subspace_size: int | float | str = 0.6, # SRP
        training_method: str = "patches",   # SRP
        lam: float = 6.0,                   # lambda_base for Poisson (from SRP & ROSE lambdaOption)
        drift_detector: base.DriftDetector | None = None,    # For base learners
        warning_detector: base.DriftDetector | None = None,  # For base learners
        disable_detector: str = "off",      # For base learners
        seed: int | None = None,            # random_seed
        metric: ClassificationMetric | None = None, # For base learner ADWINs
        # StreamWiseRandomPatches specific
        prune_threshold: float = 0.5,
        min_samples_for_pruning: int = 50, # From previous river version, good addition
        dynamic_score_update_factors: tuple[float, float] = (0.9, 1.1), # 0.9/1.1 rule
        dynamic_score_bounds: tuple[float, float] = (0.1, 10.0), # From previous river version
        # ROSE-specific parameters
        theta_class_decay: float = 0.99,
        rose_window_size: int = 500, # For instance window & background eval period
    ):
        super().__init__(
            model=model if model is not None else HoeffdingTreeClassifier(grace_period=50, delta=0.01),
            n_models=n_models, subspace_size=subspace_size, training_method=training_method,
            lam=lam, drift_detector=drift_detector, warning_detector=warning_detector,
            disable_detector=disable_detector,
            disable_weighted_vote=True, # This class uses its own combined weighting
            seed=seed, metric=metric if metric is not None else AccMetric(),
            evaluator_window_size=rose_window_size # For BaseSRPEstimator's rolling_accuracy
        )
        self._base_learner_class = BaseSRPWiseRoseLearner

        # StreamWise parts
        self.prune_threshold = prune_threshold
        self.min_samples_for_pruning = min_samples_for_pruning
        self.correct_factor, self.incorrect_factor = dynamic_score_update_factors
        self.dynamic_score_min_val, self.dynamic_score_max_val = dynamic_score_bounds
        self._dynamic_perf_scores: list[float] = [] # tree_accuracies (lower is better)
        self._dynamic_weights: list[float] = []     # ensemble_weights
        self._pruning_correct_predictions: list[int] = []
        self._pruning_total_predictions: list[int] = []
        self.weight_history: list[list[float]] = []

        # ROSE parts
        self.theta_class_decay = theta_class_decay
        self.rose_window_size = rose_window_size
        self.base_lambda = lam # Store original lambda for ROSE adaptive poisson
        self._class_sizes_decayed: typing.Optional[np.ndarray] = None
        self._n_classes: int = 0
        self._ensemble_warning_active: bool = False
        self._first_warning_timestamp: int = 0
        self._background_learners: list[BaseSRPWiseRoseLearner] = []
        self._instance_window_per_class: typing.Optional[list[collections.deque]] = None

        if len(self.models) > 0: self._init_all_custom_metrics()

    def _init_all_custom_metrics(self):
        # StreamWise metrics
        self._dynamic_perf_scores = [1.0] * self.n_models
        equal_weight = 1.0 / self.n_models if self.n_models > 0 else 1.0
        self._dynamic_weights = [equal_weight] * self.n_models
        self._pruning_correct_predictions = [0] * self.n_models
        self._pruning_total_predictions = [0] * self.n_models
        self.weight_history = []
        # ROSE metrics (class sizes, warning state) are init in _init_rose_specific_state

    def _init_rose_specific_state(self, instance_features: dict, y_true: base.typing.Target):
        if hasattr(self.model, '_n_classes_seen') and self.model._n_classes_seen is not None and self.model._n_classes_seen > 0:
             self._n_classes = self.model._n_classes_seen
        elif isinstance(y_true, int): self._n_classes = int(y_true) + 1
        else: raise ValueError("SRPRoseRiver: Cannot determine n_classes.")
        if self._n_classes <= 0 : raise ValueError(f"SRPRoseRiver: n_classes {self._n_classes} invalid.")
        self._class_sizes_decayed = np.array([1.0 / self._n_classes] * self._n_classes, dtype=float)
        per_class_limit = max(1, self.rose_window_size // self._n_classes if self._n_classes > 0 else self.rose_window_size)
        self._instance_window_per_class = [collections.deque(maxlen=per_class_limit) for _ in range(self._n_classes)]
        self._ensemble_warning_active = False

    def _init_ensemble(self, features: list): # Override to init all metrics
        super()._init_ensemble(features) # Creates model_wrappers
        self._init_all_custom_metrics()  # Init StreamWise metrics for these wrappers

    def _update_dynamic_weights_from_scores(self): # StreamWise weighting
        if not self._dynamic_perf_scores: return
        raw_weights = [1.0 / (1.0 + score) for score in self._dynamic_perf_scores]
        total_weight = sum(raw_weights)
        if total_weight > 0: self._dynamic_weights = [w / total_weight for w in raw_weights]
        else: self._dynamic_weights = [(1.0/self.n_models if self.n_models > 0 else 1.0)] * self.n_models
        if hasattr(self, 'weight_history'): self.weight_history.append(list(self._dynamic_weights))

    def _reset_streamwise_metrics_for_model(self, model_idx: int): # For StreamWise
        self._dynamic_perf_scores[model_idx] = 1.0
        self._dynamic_perf_scores[model_idx] = max(self.dynamic_score_min_val, min(self._dynamic_perf_scores[model_idx], self.dynamic_score_max_val))
        self._pruning_correct_predictions[model_idx] = 0
        self._pruning_total_predictions[model_idx] = 0

    def _prune_models_streamwise(self, current_features_list: list): # StreamWise pruning
        for i, model_wrapper in enumerate(self.models):
            if self._pruning_total_predictions[i] >= self.min_samples_for_pruning:
                pruning_accuracy = 0.0
                if self._pruning_total_predictions[i] > 0:
                    pruning_accuracy = self._pruning_correct_predictions[i] / self._pruning_total_predictions[i]
                if pruning_accuracy < self.prune_threshold:
                    # print(f"SW Pruning: Tree {i} (Acc: {pruning_accuracy:.2f})")
                    model_wrapper.reset(all_features=current_features_list, n_samples_seen=self._n_samples_seen)
                    self._reset_streamwise_metrics_for_model(i)

    def _update_rose_class_sizes(self, y_true_idx: int): # ROSE
        if self._class_sizes_decayed is None or y_true_idx >= self._n_classes: return
        for i in range(self._n_classes):
            is_curr = 1.0 if i == y_true_idx else 0.0
            self._class_sizes_decayed[i] = (self.theta_class_decay * self._class_sizes_decayed[i] + \
                                           (1.0 - self.theta_class_decay) * is_curr)
        current_sum = np.sum(self._class_sizes_decayed)
        if current_sum > 1e-9: self._class_sizes_decayed /= current_sum
        else: self._class_sizes_decayed.fill(1.0 / self._n_classes if self._n_classes > 0 else 0)

    def _get_rose_adaptive_lambda(self, y_true_idx: int) -> float: # ROSE
        if self._class_sizes_decayed is None or not self._class_sizes_decayed.any() or y_true_idx >= self._n_classes:
            return self.base_lambda
        max_cs = np.max(self._class_sizes_decayed); curr_cs = self._class_sizes_decayed[y_true_idx]
        if curr_cs <= 1e-9: adaptive_lambda = self.base_lambda + self.base_lambda * math.log(max_cs / 1e-9 if max_cs > 1e-9 else 100.0)
        else: adaptive_lambda = self.base_lambda + self.base_lambda * math.log(max_cs / curr_cs)
        return max(1.0, min(adaptive_lambda, self.base_lambda * 5)) # Cap lambda

    def learn_one(self, x: dict, y: base.typing.Target, **kwargs):
        self._n_samples_seen += 1; y_true_idx = int(y)
        current_feature_names = list(x.keys())

        if self._class_sizes_decayed is None: # First time setup for ROSE parts
            self._init_rose_specific_state(x, y)
        if not self.models: # First time setup for ensemble and StreamWise metrics
            self._init_ensemble(current_feature_names) # This calls _init_all_custom_metrics

        # Dynamically expand class-related structures if a new class is seen
        if y_true_idx >= self._n_classes:
            old_n_classes = self._n_classes; self._n_classes = y_true_idx + 1
            if self._class_sizes_decayed is not None: # Should exist
                new_cs = np.zeros(self._n_classes, dtype=float); new_cs[:old_n_classes] = self._class_sizes_decayed
                new_cs[old_n_classes:] = 1.0 / self._n_classes
                self._class_sizes_decayed = new_cs
                sum_cs = np.sum(self._class_sizes_decayed); self._class_sizes_decayed /= (sum_cs if sum_cs > 1e-9 else 1.0)
            if self._instance_window_per_class is not None: # Should exist
                per_class_limit = max(1, self.rose_window_size // self._n_classes if self._n_classes > 0 else self.rose_window_size)
                for _i in range(old_n_classes, self._n_classes): self._instance_window_per_class.append(collections.deque(maxlen=per_class_limit))
                for class_deque in self._instance_window_per_class: class_deque.maxlen = per_class_limit

        # ROSE: Update class sizes and store instance in window
        self._update_rose_class_sizes(y_true_idx)
        if self._instance_window_per_class is not None and y_true_idx < len(self._instance_window_per_class):
            self._instance_window_per_class[y_true_idx].append(InstanceWithTimestamp(x, y, self._n_samples_seen))

        # ROSE: Get adaptive lambda for current instance
        adaptive_poisson_lambda = self._get_rose_adaptive_lambda(y_true_idx)

        # --- StreamWise: Update metrics & ROSE: Update rolling accuracy (before training this step) ---
        new_overall_warning_this_step = False
        for i, model_wrapper in enumerate(self.models):
            y_pred_tree = model_wrapper.predict_one(x)
            if y_pred_tree is not None:
                y_pred_idx = int(y_pred_tree)
                # StreamWise metrics update
                if y_pred_idx == y_true_idx:
                    self._dynamic_perf_scores[i] *= self.correct_factor
                    self._pruning_correct_predictions[i] += 1
                else:
                    self._dynamic_perf_scores[i] *= self.incorrect_factor
                self._pruning_total_predictions[i] += 1
                self._dynamic_perf_scores[i] = max(self.dynamic_score_min_val, min(self._dynamic_perf_scores[i], self.dynamic_score_max_val))
                # ROSE rolling accuracy (used for BG selection and potentially voting)
                model_wrapper.rolling_accuracy.update(y_true=y_true_idx, y_pred=y_pred_idx)
                # Metric for base learner's ADWIN
                if model_wrapper.metric is not None: model_wrapper.metric.update(y_true=y_true_idx, y_pred=y_pred_idx)
            else: # No prediction
                self._dynamic_perf_scores[i] *= self.incorrect_factor
                self._pruning_total_predictions[i] += 1
                self._dynamic_perf_scores[i] = max(self.dynamic_score_min_val, min(self._dynamic_perf_scores[i], self.dynamic_score_max_val))

            # Check for ROSE overall warning trigger
            if not self._ensemble_warning_active and \
               hasattr(model_wrapper, 'warning_detector') and model_wrapper.warning_detector and \
               model_wrapper.warning_detector.drift_detected:
                new_overall_warning_this_step = True

        # --- StreamWise: Update dynamic weights based on scores ---
        self._update_dynamic_weights_from_scores()

        # --- StreamWise: Explicit Pruning ---
        self._prune_models_streamwise(current_features_list=current_feature_names)
        self._update_dynamic_weights_from_scores() # Re-update weights if pruning occurred

        # --- ROSE: Background Ensemble Initialization & Training (if warning triggered) ---
        if new_overall_warning_this_step and not self._ensemble_warning_active:
            self._ensemble_warning_active = True; self._first_warning_timestamp = self._n_samples_seen
            # print(f"Step {self._n_samples_seen}: ROSE Ensemble warning. Init BG learners.")
            self._background_learners = []
            for i in range(self.n_models):
                subspace = self._subspaces[i % len(self._subspaces)] if self._subspaces else None
                bg_rng = random.Random(self._rng.randint(0, 2**32-1))
                bg_learner = BaseSRPWiseRoseLearner(idx_original=i+self.n_models, model_prototype=self.model,
                    metric_prototype=self.metric, created_on=self._n_samples_seen,
                    drift_detector_prototype=self.drift_detector, warning_detector_prototype=self.warning_detector, # BG usually no warning
                    is_background_learner=True, rng=bg_rng, features=subspace, evaluator_window_size=self.rose_window_size)
                self._background_learners.append(bg_learner)
            
            all_window_instances: list[InstanceWithTimestamp] = []
            if self._instance_window_per_class:
                for class_deque in self._instance_window_per_class: all_window_instances.extend(list(class_deque))
            all_window_instances.sort() # By timestamp
            for inst_ts in all_window_instances:
                k_bg_win = poisson(rate=self.base_lambda, rng=self._rng) # Use base_lambda for window training
                if k_bg_win > 0:
                    for bg_model in self._background_learners:
                        bg_model.learn_one(x=inst_ts.x, y=inst_ts.y, w=k_bg_win, n_samples_seen=inst_ts.timestamp, **kwargs)

        # --- Train Primary Ensemble & Handle Internal Drifts ---
        for i, model_wrapper in enumerate(self.models):
            k_train = poisson(rate=adaptive_poisson_lambda, rng=self._rng) # Use adaptive lambda
            if k_train > 0:
                old_n_drifts = model_wrapper.n_drifts_detected
                model_wrapper.learn_one(x=x, y=y, w=k_train, n_samples_seen=self._n_samples_seen, **kwargs)
                if model_wrapper.n_drifts_detected > old_n_drifts: # Internal ADWIN reset
                    # print(f"Internal Drift: Tree {i}")
                    self._reset_streamwise_metrics_for_model(i) # Reset StreamWise metrics

        # --- ROSE: Train Background Ensemble (if active) & Evaluate/Replace ---
        if self._ensemble_warning_active and self._background_learners:
            for bg_model in self._background_learners:
                y_pred_bg = bg_model.predict_one(x)
                if y_pred_bg is not None:
                    y_pred_bg_idx = int(y_pred_bg)
                    bg_model.rolling_accuracy.update(y_true=y_true_idx, y_pred=y_pred_bg_idx)
                    if bg_model.metric is not None: bg_model.metric.update(y_true=y_true_idx, y_pred=y_pred_bg_idx)
                k_bg_curr = poisson(rate=adaptive_poisson_lambda, rng=self._rng)
                if k_bg_curr > 0:
                    bg_model.learn_one(x=x, y=y, w=k_bg_curr, n_samples_seen=self._n_samples_seen, **kwargs)

            if self._n_samples_seen - self._first_warning_timestamp >= self.rose_window_size:
                # print(f"Step {self._n_samples_seen}: ROSE BG Eval Period OVER.")
                all_candidates = list(self.models) + list(self._background_learners)
                candidate_scores = [(learner.rolling_accuracy.get() if learner.rolling_accuracy.get() is not None else 0.0) for learner in all_candidates]
                
                sorted_indices = sorted(range(len(all_candidates)), key=lambda k_idx: candidate_scores[k_idx], reverse=True)
                new_primary_ensemble: list[BaseSRPWiseRoseLearner] = []
                
                existing_streamwise_metrics = {
                    "scores": list(self._dynamic_perf_scores),
                    "correct": list(self._pruning_correct_predictions),
                    "total": list(self._pruning_total_predictions)
                }
                new_streamwise_metrics = {
                    "scores": [1.0] * self.n_models, # Default for new/potentially replaced
                    "correct": [0] * self.n_models,
                    "total": [0] * self.n_models
                }

                for rank, new_idx in enumerate(range(self.n_models)):
                    if rank < len(sorted_indices):
                        selected_cand_original_idx = sorted_indices[rank]
                        selected_learner = all_candidates[selected_cand_original_idx]
                        selected_learner.is_background_learner = False
                        if selected_learner._warning_detector_prototype: # Reset warning detector for primary
                             selected_learner.warning_detector = selected_learner._warning_detector_prototype.clone()
                        else: selected_learner.warning_detector = None
                        selected_learner.disable_background_learner = selected_learner.warning_detector is None
                        new_primary_ensemble.append(selected_learner)

                        # Carry over StreamWise metrics IF the selected learner was from the original primary ensemble
                        # And it's being placed back. This is complex due to potential reordering.
                        # Simplest: reset StreamWise metrics for all in new primary ensemble.
                        # Or, try to map if possible:
                        if selected_learner in self.models:
                            try:
                                original_pos_in_primary = self.models.index(selected_learner)
                                new_streamwise_metrics["scores"][new_idx] = existing_streamwise_metrics["scores"][original_pos_in_primary]
                                new_streamwise_metrics["correct"][new_idx] = existing_streamwise_metrics["correct"][original_pos_in_primary]
                                new_streamwise_metrics["total"][new_idx] = existing_streamwise_metrics["total"][original_pos_in_primary]
                            except ValueError: # Should not happen if 'in self.models' is true
                                pass 
                        # else, it's a new learner from BG, so StreamWise metrics remain default (1.0, 0, 0)
                
                self.data = new_primary_ensemble
                self._dynamic_perf_scores = new_streamwise_metrics["scores"]
                self._pruning_correct_predictions = new_streamwise_metrics["correct"]
                self._pruning_total_predictions = new_streamwise_metrics["total"]
                
                self._background_learners = []
                self._ensemble_warning_active = False
                # print(f"Step {self._n_samples_seen}: ROSE New primary ensemble selected.")

        # --- Final Update to StreamWise Weights ---
        self._update_dynamic_weights_from_scores()

    def predict_proba_one(self, x: dict, **kwargs) -> dict[base.typing.ClfTarget, float]:
        # Using StreamWise dynamic weighting for prediction
        y_pred_proba: typing.Counter = collections.Counter()
        if not self.models:
            if self._class_sizes_decayed is None and 'y' in kwargs: self._init_rose_specific_state(x, kwargs['y'])
            if not self.models: self._init_ensemble(list(x.keys()))
            if not self.models: return {}
        if not self._dynamic_weights: self._init_all_custom_metrics() # Ensure weights are there

        for i, model_wrapper in enumerate(self.models):
            y_proba_tree = model_wrapper.predict_proba_one(x, **kwargs)
            weight = self._dynamic_weights[i] # From StreamWise 0.9/1.1 rule
            if weight > 0.0:
                for label, proba in y_proba_tree.items():
                    y_pred_proba[label] += proba * weight
        
        total_proba = sum(y_pred_proba.values())
        if total_proba > 0: return {label: proba / total_proba for label, proba in y_pred_proba.items()}
        return {}

    def reset(self):
        super().reset()
        self._class_sizes_decayed = None; self._n_classes = 0
        self._ensemble_warning_active = False; self._first_warning_timestamp = 0
        self._background_learners = []; self._instance_window_per_class = None
        # StreamWise metrics
        self._dynamic_perf_scores = []; self._dynamic_weights = []
        self._pruning_correct_predictions = []; self._pruning_total_predictions = []
        self.weight_history = []
