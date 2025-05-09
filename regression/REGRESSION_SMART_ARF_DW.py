from __future__ import annotations

import abc
import collections
import copy
import math
import random
import typing

import numpy as np
from river import base, metrics, stats
from river.drift import ADWIN, NoDrift
# HoeffdingTreeClassifier and its nodes are not directly used by the new regressor,
# but BaseForest might reference BaseTreeClassifier in type hints.
from river.tree.hoeffding_tree_classifier import HoeffdingTreeClassifier
from river.tree.hoeffding_tree_regressor import HoeffdingTreeRegressor
from river.tree.nodes.arf_htr_nodes import RandomLeafAdaptive, RandomLeafMean, RandomLeafModel
from river.tree.splitter import Splitter
from river.utils.random import poisson

import logging
import matplotlib.pyplot as plt

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ================================================================
# Existing BaseForest, BaseTreeRegressor, ARFRegressor (from user)
# MINOR MODIFICATION:
# - Added _drift_norm[i] = stats.Var() on tree reset in ARFRegressor.learn_one
# - Made BaseForest.__init__ consistent with ARFRegressor's __init__
# ================================================================
class BaseForest(base.Ensemble):
    _FEATURES_SQRT = "sqrt"
    _FEATURES_LOG2 = "log2"

    def __init__( # Changed from init to __init__ for consistency with ARFRegressor
        self,
        n_models: int,
        max_features: bool | str | int,
        lambda_value: int,
        drift_detector: base.DriftDetector,
        warning_detector: base.DriftDetector,
        metric: metrics.base.MultiClassMetric | metrics.base.RegressionMetric,
        disable_weighted_vote,
        seed,
    ):
        super().__init__([])  # type: ignore
        self.n_models = n_models
        self.max_features = max_features
        self.lambda_value = lambda_value
        self.metric = metric
        self.disable_weighted_vote = disable_weighted_vote
        self.drift_detector = drift_detector
        self.warning_detector = warning_detector
        self.seed = seed

        self._rng = random.Random(self.seed)

        self._warning_detectors: list[base.DriftDetector]
        self._warning_detection_disabled = True
        if not isinstance(self.warning_detector, NoDrift):
            # Initialize based on the initial self.n_models
            self._warning_detectors = [self.warning_detector.clone() for _ in range(self.n_models)]
            self._warning_detection_disabled = False

        self._drift_detectors: list[base.DriftDetector]
        self._drift_detection_disabled = True
        if not isinstance(self.drift_detector, NoDrift):
            # Initialize based on the initial self.n_models
            self._drift_detectors = [self.drift_detector.clone() for _ in range(self.n_models)]
            self._drift_detection_disabled = False

        self._background: list[HoeffdingTreeClassifier | BaseTreeRegressor | None] = ( # Adjusted type hint
            [] if self._warning_detection_disabled else [None] * self.n_models # Init with [], filled in _init_ensemble
        )

        self._metrics = [self.metric.clone() for _ in range(self.n_models)] # Init with [], filled in _init_ensemble

        self._warning_tracker: dict = (
            collections.defaultdict(int) if not self._warning_detection_disabled else {} # type: ignore
        )
        self._drift_tracker: dict = (
            collections.defaultdict(int) if not self._drift_detection_disabled else {} # type: ignore
        )

    @property
    def _min_number_of_models(self):
        return 0

    @classmethod
    def _unit_test_params(cls):
        yield {"n_models": 3}

    def _unit_test_skips(self):
        return {"check_shuffle_features_no_impact"}

    @abc.abstractmethod
    def _drift_detector_input(
        self,
        tree_id: int,
        y_true,
        y_pred,
    ) -> int | float:
        raise NotImplementedError

    @abc.abstractmethod
    def _new_base_model(self) -> HoeffdingTreeClassifier | BaseTreeRegressor: # Adjusted type hint
        raise NotImplementedError

    def n_warnings_detected(self, tree_id: int | None = None) -> int:
        if self._warning_detection_disabled: return 0
        count = 0
        num_models = len(self.data)
        if self._warning_tracker is not None:
            if tree_id is None: count = sum(self._warning_tracker.values())
            elif tree_id < num_models: count = self._warning_tracker.get(tree_id, 0)
        return count

    def n_drifts_detected(self, tree_id: int | None = None) -> int:
        if self._drift_detection_disabled: return 0
        count = 0
        num_models = len(self.data)
        if self._drift_tracker is not None:
            if tree_id is None: count = sum(self._drift_tracker.values())
            elif tree_id < num_models: count = self._drift_tracker.get(tree_id, 0)
        return count

    # learn_one will be implemented by subclasses like ARFRegressor or ARFRegressorDynamicWeights
    def learn_one(self, x: dict, y: base.typing.Target, **kwargs):
        raise NotImplementedError

    def _init_ensemble(self, features: list):
        self._set_max_features(len(features))
        self.data = [self._new_base_model() for _ in range(self.n_models)]
        # Properly initialize lists based on current self.n_models
        if not self._warning_detection_disabled:
            self._warning_detectors = [self.warning_detector.clone() for _ in range(self.n_models)]
            self._background = [None] * self.n_models
        if not self._drift_detection_disabled:
            self._drift_detectors = [self.drift_detector.clone() for _ in range(self.n_models)]
        self._metrics = [self.metric.clone() for _ in range(self.n_models)]

        if self._warning_tracker is not None: self._warning_tracker.clear()
        if self._drift_tracker is not None: self._drift_tracker.clear()


    def _set_max_features(self, n_features):
        orig_max_features = self.max_features
        if self.max_features == "sqrt": self.max_features = round(math.sqrt(n_features))
        elif self.max_features == "log2": self.max_features = round(math.log2(n_features))
        elif isinstance(self.max_features, int): pass
        elif isinstance(self.max_features, float): self.max_features = int(self.max_features * n_features)
        elif self.max_features is None: self.max_features = n_features
        else: raise AttributeError(f"Invalid max_features: {orig_max_features}...")
        if self.max_features < 0: self.max_features += n_features
        if self.max_features <= 0: self.max_features = 1
        if self.max_features > n_features: self.max_features = n_features

class BaseTreeRegressor(HoeffdingTreeRegressor):
    def __init__(
        self,
        max_features: int = 2,
        grace_period: int = 200,
        max_depth: int | None = None,
        delta: float = 1e-7,
        tau: float = 0.05,
        leaf_prediction: str = "adaptive",
        leaf_model: base.Regressor | None = None,
        model_selector_decay: float = 0.95,
        nominal_attributes: list | None = None,
        splitter: Splitter | None = None,
        min_samples_split: int = 5,
        binary_split: bool = False,
        max_size: float = 100.0,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
        rng: random.Random | None = None,
    ):
        super().__init__(
            grace_period=grace_period, max_depth=max_depth, delta=delta, tau=tau,
            leaf_prediction=leaf_prediction, leaf_model=leaf_model,
            model_selector_decay=model_selector_decay, nominal_attributes=nominal_attributes,
            splitter=splitter, min_samples_split=min_samples_split, binary_split=binary_split,
            max_size=max_size, memory_estimate_period=memory_estimate_period,
            stop_mem_management=stop_mem_management, remove_poor_attrs=remove_poor_attrs,
            merit_preprune=merit_preprune,
        )
        self.max_features = max_features
        self.rng = rng

    def _new_leaf(self, initial_stats=None, parent=None):  # noqa
        if parent is not None: depth = parent.depth + 1
        else: depth = 0
        leaf_model = None
        if self.leaf_prediction in {self._MODEL, self._ADAPTIVE}:
            if parent is None: leaf_model = copy.deepcopy(self.leaf_model)
            else:
                try: leaf_model = copy.deepcopy(parent._leaf_model)  # noqa
                except AttributeError: leaf_model = copy.deepcopy(self.leaf_model)

        if self.leaf_prediction == self._TARGET_MEAN:
            return RandomLeafMean(initial_stats, depth, self.splitter, self.max_features, self.rng)
        elif self.leaf_prediction == self._MODEL:
            return RandomLeafModel(initial_stats, depth, self.splitter, self.max_features, self.rng, leaf_model=leaf_model)
        else:  # adaptive learning node
            new_adaptive = RandomLeafAdaptive(initial_stats, depth, self.splitter, self.max_features, self.rng, leaf_model=leaf_model)
            if parent is not None and isinstance(parent, RandomLeafAdaptive):
                new_adaptive._fmse_mean = parent._fmse_mean; new_adaptive._fmse_model = parent._fmse_model  # noqa
            return new_adaptive

class ARFRegressor(BaseForest, base.Regressor):
    _MEAN = "mean"
    _MEDIAN = "median"
    _VALID_AGGREGATION_METHOD = [_MEAN, _MEDIAN]

    def __init__(
        self, n_models: int = 10, max_features="sqrt", aggregation_method: str = "mean", # Changed default to mean for consistency
        lambda_value: int = 6, metric: metrics.base.RegressionMetric | None = None,
        disable_weighted_vote=True, # Original ARF regressor often uses unweighted mean or median
        drift_detector: base.DriftDetector | None = None, warning_detector: base.DriftDetector | None = None,
        grace_period: int = 50, max_depth: int | None = None, delta: float = 0.01, tau: float = 0.05,
        leaf_prediction: str = "adaptive", leaf_model: base.Regressor | None = None,
        model_selector_decay: float = 0.95, nominal_attributes: list | None = None,
        splitter: Splitter | None = None, min_samples_split: int = 5, binary_split: bool = False,
        max_size: float = 500.0, memory_estimate_period: int = 2_000_000,
        stop_mem_management: bool = False, remove_poor_attrs: bool = False, merit_preprune: bool = True,
        seed: int | None = None,
    ):
        super().__init__(
            n_models=n_models, max_features=max_features, lambda_value=lambda_value,
            metric=metric or metrics.MSE(), disable_weighted_vote=disable_weighted_vote,
            drift_detector=drift_detector or ADWIN(0.001),
            warning_detector=warning_detector or ADWIN(0.01), seed=seed,
        )
        self.grace_period=grace_period; self.max_depth=max_depth; self.delta=delta; self.tau=tau
        self.leaf_prediction=leaf_prediction; self.leaf_model=leaf_model
        self.model_selector_decay=model_selector_decay; self.nominal_attributes=nominal_attributes
        self.splitter=splitter; self.min_samples_split=min_samples_split; self.binary_split=binary_split
        self.max_size=max_size; self.memory_estimate_period=memory_estimate_period
        self.stop_mem_management=stop_mem_management; self.remove_poor_attrs=remove_poor_attrs
        self.merit_preprune=merit_preprune

        if aggregation_method in self._VALID_AGGREGATION_METHOD: self.aggregation_method = aggregation_method
        else: raise ValueError(f"Invalid aggregation_method: {aggregation_method}.")

        self._drift_norm: list[stats.Var] = [] # Will be initialized in _init_ensemble by ARFRegressorDynamicWeights or explicitly here if needed.
                                              # For plain ARFRegressor, it's initialized on first use or in _init_ensemble.
        # Let's initialize it properly in _init_ensemble of ARFRegressor itself
        # if len(self.data) > 0: # If ensemble is already created (e.g. from loading a model)
        #    self._drift_norm = [stats.Var() for _ in range(len(self.data))]


    def _init_ensemble(self, features: list):
        super()._init_ensemble(features)
        # Initialize _drift_norm here, tied to the actual number of models created
        self._drift_norm = [stats.Var() for _ in range(len(self.data))]


    @property
    def _mutable_attributes(self):
        return { "max_features", "aggregation_method", "lambda_value", "grace_period",
                 "delta", "tau", "model_selector_decay"}

    def learn_one(self, x: dict, y: base.typing.Target, **kwargs):
        if len(self.data) == 0: self._init_ensemble(sorted(x.keys()))

        for i, model in enumerate(self):
            y_pred = model.predict_one(x)
            self._metrics[i].update(y_true=y, y_pred=y_pred)

            k = poisson(rate=self.lambda_value, rng=self._rng)
            if k > 0:
                drift_input = None # Defined early to ensure it's available
                if not self._warning_detection_disabled:
                    # Check bounds for _background and _warning_detectors
                    if self._background and i < len(self._background) and self._background[i] is not None:
                         self._background[i].learn_one(x=x, y=y, w=k) # type: ignore

                    if i < len(self._warning_detectors):
                        drift_input = self._drift_detector_input(i, y, y_pred)
                        self._warning_detectors[i].update(drift_input)
                        if self._warning_detectors[i].drift_detected:
                            if self._background and i < len(self._background):
                                self._background[i] = self._new_base_model()
                            self._warning_detectors[i] = self.warning_detector.clone()
                            if self._warning_tracker: self._warning_tracker[i] += 1

                if not self._drift_detection_disabled and i < len(self._drift_detectors):
                    drift_input = (drift_input if drift_input is not None
                                   else self._drift_detector_input(i, y, y_pred))
                    self._drift_detectors[i].update(drift_input)

                    if self._drift_detectors[i].drift_detected:
                        if (not self._warning_detection_disabled and self._background and
                                i < len(self._background) and self._background[i] is not None):
                            self.data[i] = self._background[i]
                            if i < len(self._background): self._background[i] = None
                            if i < len(self._warning_detectors): self._warning_detectors[i] = self.warning_detector.clone()
                        else:
                            self.data[i] = self._new_base_model()

                        # Reset related states for the replaced tree
                        if i < len(self._drift_detectors): self._drift_detectors[i] = self.drift_detector.clone()
                        if i < len(self._metrics): self._metrics[i] = self.metric.clone()
                        if i < len(self._drift_norm): self._drift_norm[i] = stats.Var() # Crucial reset
                        if self._drift_tracker: self._drift_tracker[i] += 1

                # Train the main model (or the newly replaced one)
                # Ensure model at index i is the one we intend to train
                if i < len(self.data):
                    self.data[i].learn_one(x=x, y=y, w=k)
        return self


    def predict_one(self, x: dict) -> base.typing.RegTarget:
        if len(self.data) == 0: self._init_ensemble(sorted(x.keys())); return 0.0

        y_preds = np.zeros(len(self.data))
        for i, model in enumerate(self): y_preds[i] = model.predict_one(x)

        if self.aggregation_method == self._MEAN:
            if not self.disable_weighted_vote:
                weights = np.zeros(len(self.data))
                sum_weights = 0.0
                for i in range(len(self.data)):
                    metric_val = self._metrics[i].get()
                    # Assuming metric is error (smaller is better)
                    # A simple inverse proportional weighting, or 1 - normalized_error
                    # Original ARF uses (sum_errors - error_i) for weights
                    weights[i] = metric_val # Store raw error
                    sum_weights += metric_val

                if sum_weights > 0:
                    # Higher error = lower weight. (sum_metric_values - metric_value_i)
                    final_weights = sum_weights - weights
                     # Handle case where all errors are same, resulting in zero weights if not careful
                    if np.sum(final_weights) > 0: final_weights /= np.sum(final_weights)
                    else: final_weights = np.ones_like(weights) / len(weights) if len(weights) > 0 else [] # Equal weights

                    if len(final_weights) == len(y_preds): return float(np.sum(y_preds * final_weights))

            return float(np.mean(y_preds)) if len(y_preds) > 0 else 0.0
        elif self.aggregation_method == self._MEDIAN:
            return float(np.median(y_preds)) if len(y_preds) > 0 else 0.0
        return 0.0


    def _new_base_model(self):
        return BaseTreeRegressor(
            max_features=self.max_features, grace_period=self.grace_period, max_depth=self.max_depth,
            delta=self.delta, tau=self.tau, leaf_prediction=self.leaf_prediction, leaf_model=self.leaf_model,
            model_selector_decay=self.model_selector_decay, nominal_attributes=self.nominal_attributes,
            splitter=self.splitter, binary_split=self.binary_split, max_size=self.max_size,
            memory_estimate_period=self.memory_estimate_period, stop_mem_management=self.stop_mem_management,
            remove_poor_attrs=self.remove_poor_attrs, merit_preprune=self.merit_preprune, rng=self._rng,
        )

    def _drift_detector_input(self, tree_id: int, y_true: int | float, y_pred: int | float) -> int | float:
        drift_input_val = y_true - y_pred
        # Ensure tree_id is within bounds for _drift_norm
        if tree_id < len(self._drift_norm):
            self._drift_norm[tree_id].update(drift_input_val)
            if self._drift_norm[tree_id].mean.n == 1: return 0.5
            sd = math.sqrt(self._drift_norm[tree_id].get())
            return (drift_input_val + 3 * sd) / (6 * sd) if sd > 0 else 0.5
        return 0.5 # Default if tree_id is out of bounds (should not happen with proper init)

# ================================================================
# NEW CLASS 1: ARFRegressorDynamicWeights
# ================================================================
class ARFRegressorDynamicWeights(ARFRegressor):
    """
    Adaptive Random Forest Regressor with dynamic model weighting (0.9/1.1 rule).
    The weighting score for each tree is updated based on its prediction error
    relative to its own historical error's standard deviation.
    """
    def __init__(self, *args, dynamic_weighting_error_factor: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic_weighting_error_factor = dynamic_weighting_error_factor
        self._dynamic_perf_scores: list[float] = []
        self._dynamic_weights: list[float] = []
        # If ensemble loaded/already initialized, set up dynamic weights
        if len(self.data) > 0: self._init_dynamic_weights()

    def _init_dynamic_weights(self):
        num_models = len(self.data)
        self._dynamic_perf_scores = [1.0] * num_models
        equal_weight = 1.0 / num_models if num_models > 0 else 1.0
        self._dynamic_weights = [equal_weight] * num_models

    def _init_ensemble(self, features: list):
        super()._init_ensemble(features) # Calls ARFRegressor._init_ensemble
        self._init_dynamic_weights()     # Then init dynamic parts

    def _update_dynamic_weights(self):
        num_models = len(self.data)
        if not self._dynamic_perf_scores or len(self._dynamic_perf_scores) != num_models:
            self._init_dynamic_weights() # Ensure lists are correctly sized
            if num_models == 0: return

        # Scores are error-like: smaller is better for raw score.
        # Weights should be inverse to scores.
        raw_weights = [1.0 / (1.0 + score) if score >= 0 else 1.0 for score in self._dynamic_perf_scores]
        total_weight = sum(raw_weights)

        if total_weight > 0:
            self._dynamic_weights = [w / total_weight for w in raw_weights]
        else: # Fallback to equal weights if total_weight is 0 (e.g., all scores are huge negative?)
            equal_weight = 1.0 / num_models if num_models > 0 else 1.0
            self._dynamic_weights = [equal_weight] * num_models
            # logging.warning("Total dynamic weight is zero, falling back to equal weights.")


    def learn_one(self, x: dict, y: base.typing.RegTarget, **kwargs):
        if len(self.data) == 0: self._init_ensemble(sorted(x.keys()))

        tree_predictions = [0.0] * len(self.data)

        # --- Stage 1: Predictions and Metric/Score Updates ---
        for i, model in enumerate(self.data):
            if i >= len(self._dynamic_perf_scores): # Safety check, should be handled by init
                self._init_dynamic_weights() # Attempt to resize/reinit
                if i >= len(self._dynamic_perf_scores): # If still out of bounds
                    logging.error(f"Dynamic perf score index {i} out of bounds after re-init. Skipping update for this tree.")
                    continue

            y_pred_tree = model.predict_one(x)
            tree_predictions[i] = y_pred_tree # Store for later use if needed by detectors

            # Update standard ARF metric (e.g., MSE for this tree)
            self._metrics[i].update(y_true=y, y_pred=y_pred_tree)

            # Update dynamic performance score (0.9/1.1 rule)
            current_error_abs = abs(y - y_pred_tree)
            error_std_dev = 0.0
            threshold_for_good_pred = float('inf') # Default to "bad" if no std_dev

            if i < len(self._drift_norm) and self._drift_norm[i].mean.n > 1: # Need at least 2 samples for variance
                variance = self._drift_norm[i].get()
                if variance > 0:
                    error_std_dev = math.sqrt(variance)
                    threshold_for_good_pred = self.dynamic_weighting_error_factor * error_std_dev

            if current_error_abs <= threshold_for_good_pred : # Good prediction
                self._dynamic_perf_scores[i] *= 0.9
            else: # Bad prediction or insufficient data for std_dev
                self._dynamic_perf_scores[i] *= 1.1
            # Bound the scores to prevent extreme values, e.g., [0.1, 10] or similar
            self._dynamic_perf_scores[i] = max(0.01, min(self._dynamic_perf_scores[i], 100.0))


        # --- Stage 2: Drift/Warning Detection and Tree Management (Standard ARF logic) ---
        for i, model in enumerate(self.data):
            y_pred_tree_for_detector = tree_predictions[i] # Use stored prediction
            drift_input = None

            # Check for warnings
            if not self._warning_detection_disabled and i < len(self._warning_detectors):
                drift_input = self._drift_detector_input(i, y, y_pred_tree_for_detector)
                self._warning_detectors[i].update(drift_input)
                if self._warning_detectors[i].drift_detected:
                    if self._background and i < len(self._background):
                        self._background[i] = self._new_base_model()
                    self._warning_detectors[i] = self.warning_detector.clone()
                    if self._warning_tracker: self._warning_tracker[i] += 1

            # Check for drifts
            if not self._drift_detection_disabled and i < len(self._drift_detectors):
                drift_input = (drift_input if drift_input is not None
                               else self._drift_detector_input(i, y, y_pred_tree_for_detector))
                self._drift_detectors[i].update(drift_input)

                if self._drift_detectors[i].drift_detected:
                    if (not self._warning_detection_disabled and self._background and
                            i < len(self._background) and self._background[i] is not None):
                        self.data[i] = self._background[i]
                        if i < len(self._background): self._background[i] = None
                        if i < len(self._warning_detectors): self._warning_detectors[i] = self.warning_detector.clone()
                    else:
                        self.data[i] = self._new_base_model()

                    # Reset states for the replaced tree
                    if i < len(self._drift_detectors): self._drift_detectors[i] = self.drift_detector.clone()
                    if i < len(self._metrics): self._metrics[i] = self.metric.clone()
                    if i < len(self._drift_norm): self._drift_norm[i] = stats.Var() # Reset for regressor
                    if i < len(self._dynamic_perf_scores): self._dynamic_perf_scores[i] = 1.0 # Reset dynamic score
                    if self._drift_tracker: self._drift_tracker[i] += 1

        # --- Stage 3: Recalculate Weights and Train Models ---
        self._update_dynamic_weights() # Update based on new scores

        for i, model in enumerate(self.data):
            k = poisson(rate=self.lambda_value, rng=self._rng)
            if k > 0:
                # Train background model if it exists
                if not self._warning_detection_disabled and self._background and i < len(self._background) and self._background[i] is not None:
                    self._background[i].learn_one(x=x, y=y, w=k) # type: ignore
                # Train the main model
                model.learn_one(x=x, y=y, w=k)
        return self

    def predict_one(self, x: dict) -> base.typing.RegTarget:
        if len(self.data) == 0:
            self._init_ensemble(sorted(x.keys())) # Initialize if called before learn_one
            return 0.0

        # Ensure dynamic weights are initialized and correctly sized
        if not self._dynamic_weights or len(self._dynamic_weights) != len(self.data):
            self._init_dynamic_weights()

        y_preds_list = np.zeros(len(self.data))
        for i, model in enumerate(self.data):
            y_preds_list[i] = model.predict_one(x)

        if self.aggregation_method == self._MEAN:
            # Always use dynamic weights for mean aggregation if available
            # The disable_weighted_vote is more for the original ARF's metric-based weighting.
            # Here, "dynamic weights" are the primary weighting mechanism.
            if self._dynamic_weights and len(self._dynamic_weights) == len(y_preds_list):
                weights_arr = np.array(self._dynamic_weights)
                # Dynamic weights should already sum to 1 from _update_dynamic_weights
                # sum_w = np.sum(weights_arr)
                # if sum_w > 0: # Ensure weights are valid
                return float(np.sum(y_preds_list * weights_arr))
                # else: # Fallback if weights are somehow invalid (e.g., all zero)
                #     return float(np.mean(y_preds_list)) if len(y_preds_list) > 0 else 0.0
            else: # Fallback if dynamic weights are not set up
                return float(np.mean(y_preds_list)) if len(y_preds_list) > 0 else 0.0
        elif self.aggregation_method == self._MEDIAN:
            # Median typically does not use weights
            return float(np.median(y_preds_list)) if len(y_preds_list) > 0 else 0.0
        else: # Should not be reached if aggregation_method is validated
            return float(np.mean(y_preds_list)) if len(y_preds_list) > 0 else 0.0


# ================================================================
# NEW CLASS 2: SmartARFDynamicWeightsRegressor
# ================================================================
class SmartARFDynamicWeightsRegressor(ARFRegressorDynamicWeights):
    """
    Adaptive Random Forest Regressor that combines:
    1. Dynamic tree weighting (adapted 0.9/1.1 rule for regression).
    2. Dynamic ensemble size management (adding trees on drift, pruning on
       proxy "accuracy" drop or exceeding max_models).
    """
    def __init__(
        self,
        n_models: int = 10,
        max_models: int = 30,
        regression_pruning_error_threshold: float = 0.1, # Absolute error for "accurate"
        accuracy_drop_threshold: float = 0.5, # Pruning if proxy acc drops by this factor
        monitor_window: int = 100,
        **kwargs # Pass other ARFRegressorDynamicWeights params
    ):
        super().__init__(n_models=n_models, **kwargs) # Pass n_models for initial setup
        self.max_models = max_models
        self.regression_pruning_error_threshold = regression_pruning_error_threshold
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.monitor_window = monitor_window

        self.model_count_history: list[int] = [] # To plot ensemble size
        self._accuracy_window: list[collections.deque] = [] # For pruning based on acc drop
        self._warned_tree_ids: set[int] = set()
        self._warning_step: dict[int, int] = {} # Instance step when warning started
        self._warned_recent_acc: dict[int, float] = {} # Accuracy at time of warning

        if len(self.data) > 0: # If ensemble already exists (e.g. from loading)
            self._init_pruning_state()

    def _init_pruning_state(self):
        num_models = len(self.data)
        # Accuracy window stores 0s and 1s (1 if |error| <= threshold)
        self._accuracy_window = [collections.deque(maxlen=self.monitor_window) for _ in range(num_models)]
        self._warned_tree_ids.clear()
        self._warning_step.clear()
        self._warned_recent_acc.clear()

    def _init_ensemble(self, features: list):
        # Call parent's _init_ensemble (ARFRegressorDynamicWeights -> ARFRegressor)
        super()._init_ensemble(features)
        # Initialize smart pruning specific states
        self._init_pruning_state()


    def learn_one(self, x: dict, y: base.typing.RegTarget, **kwargs):
        if len(self.data) == 0: self._init_ensemble(sorted(x.keys()))

        current_step = sum(model.total_weight_observed for model in self.data) # A proxy for time
        self.model_count_history.append(len(self.data))

        # --- Stage 0: Prepare for iteration ---
        num_models_at_start_of_step = len(self.data) # Cache initial number for this step
        tree_predictions = [0.0] * num_models_at_start_of_step
        drift_detected_indices = []
        warning_detected_indices = []

        # --- Stage 1: Predictions and Local Updates ---
        for i in range(num_models_at_start_of_step):
            # Make sure lists are long enough; they should be due to _init calls
            # and management in _remove_model/_add_model.
            if i >= len(self.data) or \
               i >= len(self._metrics) or \
               i >= len(self._dynamic_perf_scores) or \
               i >= len(self._accuracy_window) or \
               (not self._warning_detection_disabled and i >= len(self._warning_detectors)) or \
               (not self._drift_detection_disabled and i >= len(self._drift_detectors)):
                # This indicates a bug in list management if it occurs
                logging.error(f"Index {i} out of bounds for core lists in learn_one. Skipping tree.")
                continue

            model = self.data[i]
            y_pred_tree = model.predict_one(x)
            tree_predictions[i] = y_pred_tree

            # Update standard metric (e.g., MSE, for finding worst model)
            self._metrics[i].update(y_true=y, y_pred=y_pred_tree)

            # Update dynamic performance score (for weighting)
            current_error_abs = abs(y - y_pred_tree)
            error_std_dev = 0.0
            threshold_for_good_pred = float('inf')
            if i < len(self._drift_norm) and self._drift_norm[i].mean.n > 1:
                variance = self._drift_norm[i].get()
                if variance > 0:
                    error_std_dev = math.sqrt(variance)
                    threshold_for_good_pred = self.dynamic_weighting_error_factor * error_std_dev
            if current_error_abs <= threshold_for_good_pred:
                self._dynamic_perf_scores[i] *= 0.9
            else:
                self._dynamic_perf_scores[i] *= 1.1
            self._dynamic_perf_scores[i] = max(0.01, min(self._dynamic_perf_scores[i], 100.0))


            # Update accuracy window (for pruning)
            # _ensure_accuracy_window_exists(i) # Should be managed by add/remove
            is_accurate_for_pruning = int(abs(y - y_pred_tree) <= self.regression_pruning_error_threshold)
            self._accuracy_window[i].append(is_accurate_for_pruning)

            # --- Check Detectors ---
            drift_input_for_detectors = None
            if not self._warning_detection_disabled:
                drift_input_for_detectors = self._drift_detector_input(i, y, y_pred_tree)
                self._warning_detectors[i].update(drift_input_for_detectors)
                if self._warning_detectors[i].drift_detected: warning_detected_indices.append(i)

            if not self._drift_detection_disabled:
                if drift_input_for_detectors is None: # Calculate if not done for warning
                     drift_input_for_detectors = self._drift_detector_input(i, y, y_pred_tree)
                self._drift_detectors[i].update(drift_input_for_detectors)
                if self._drift_detectors[i].drift_detected: drift_detected_indices.append(i)

        # --- Stage 2: Process Detections and Manage Ensemble Size ---
        indices_of_reset_trees = set()

        # Handle warnings
        for i in warning_detected_indices:
            if i >= len(self.data): continue # Tree might have been removed
            if self._background is not None and i < len(self._background) and self._background[i] is None:
                self._background[i] = self._new_base_model()
                # logging.info(f"🌳 Background learner started for tree {i} due to warning.")
            if i < len(self._warning_detectors): self._warning_detectors[i] = self.warning_detector.clone()
            if self._warning_tracker is not None: self._warning_tracker[i] += 1

            if i not in self._warned_tree_ids: # Start monitoring if not already
                 self._warned_tree_ids.add(i)
                 self._warning_step[i] = current_step
                 self._warned_recent_acc[i] = self._get_recent_accuracy(i) # Proxy accuracy
                 # logging.info(f"📉 Started monitoring proxy accuracy drop for tree {i} (current acc: {self._warned_recent_acc[i]:.3f}).")

        # Handle drifts (potential model addition or reset)
        # Keep track of indices that were modified to avoid issues if a drift leads to pruning
        # that affects subsequent indices in drift_detected_indices.
        # Iterating over a copy or adjusting indices after removal is safer.
        # For now, let's assume indices in drift_detected_indices are relative to the start of the step.
        # We need to re-evaluate indices if removals happen.

        processed_drift_indices_this_step = set()

        for original_idx in drift_detected_indices:
            # The actual index might have shifted due to prior removals in this loop (if any)
            # This simple loop structure assumes no removals YET, or indices are stable.
            # The _find_worst_model and _remove_model logic later handles index shifts for pruning.
            # If a drift directly causes pruning, careful index management is needed.
            # Here, we add first, then prune later if max_models is exceeded.

            current_idx = original_idx # This needs careful thought if removals happen mid-loop.
                                   # For now, assume original_idx is valid against current self.data
            if current_idx in processed_drift_indices_this_step or current_idx >= len(self.data):
                continue

            # logging.info(f"💥 Drift detected by detector for tree {current_idx}.")
            if self._drift_tracker is not None: self._drift_tracker[current_idx] += 1

            new_tree_candidate = None
            promoted_from_background = False
            if self._background is not None and current_idx < len(self._background) and self._background[current_idx] is not None:
                new_tree_candidate = self._background[current_idx]
                self._background[current_idx] = None # Consume background tree
                promoted_from_background = True
                # logging.info(f"➕ Candidate tree from background of tree {current_idx} for potential addition.")

            if new_tree_candidate: # Try to add this tree
                if len(self.data) >= self.max_models:
                    worst_idx_to_prune = self._find_worst_model()
                    if worst_idx_to_prune is not None:
                        # logging.info(f"📦 Ensemble at max capacity ({self.max_models}). Pruning worst tree {worst_idx_to_prune} before adding.")
                        self._remove_model(worst_idx_to_prune)
                        # If worst_idx_to_prune was current_idx, it's gone.
                        # If worst_idx_to_prune < current_idx, current_idx shifts.
                        # This makes direct addition complex if current_idx itself is pruned.
                        # For simplicity, let's assume current_idx is NOT the one pruned,
                        # or that the pruning logic correctly shifts indices.
                        # The _remove_model adjusts trackers for indices > removed_index.
                        if worst_idx_to_prune < current_idx:
                            current_idx -=1 # Adjust current_idx if an earlier model was removed
                    else: # Cannot prune, so cannot add
                        # logging.warning("Ensemble at max capacity, but no worst model found to prune. Cannot add new tree.")
                        new_tree_candidate = None # Do not add
                
                if new_tree_candidate and len(self.data) < self.max_models: # Add if there's space or space was made
                    # logging.info(f"➕ Adding new model to ensemble. Ensemble size: {len(self.data)+1}")
                    self.data.append(new_tree_candidate)
                    self._metrics.append(self.metric.clone())
                    if not self._drift_detection_disabled: self._drift_detectors.append(self.drift_detector.clone())
                    if not self._warning_detection_disabled: self._warning_detectors.append(self.warning_detector.clone())
                    if self._background is not None: self._background.append(None) # Placeholder for new tree
                    self._drift_norm.append(stats.Var()) # For new tree
                    self._accuracy_window.append(collections.deque(maxlen=self.monitor_window))
                    self._dynamic_perf_scores.append(1.0) # Default score for new tree
                    # self.n_models should be len(self.data) effectively
                else:
                    # logging.info(f"🔁 Could not add tree from background (max models or no prune). Resetting tree {current_idx} in place.")
                    # Fall through to reset logic if not added
                    new_tree_candidate = None


            if not new_tree_candidate: # Reset the existing tree at current_idx (if not added from background)
                # logging.info(f"🔁 Resetting tree {current_idx} in place due to drift.")
                self.data[current_idx] = self._new_base_model()
                self._metrics[current_idx] = self.metric.clone()
                if not self._drift_detection_disabled: self._drift_detectors[current_idx] = self.drift_detector.clone()
                if not self._warning_detection_disabled: self._warning_detectors[current_idx] = self.warning_detector.clone()
                self._drift_norm[current_idx] = stats.Var()
                self._accuracy_window[current_idx].clear()
                self._dynamic_perf_scores[current_idx] = 1.0
                indices_of_reset_trees.add(current_idx)
                if current_idx in self._warned_tree_ids: self._clear_warning_state(current_idx)

            # Reset the specific drift detector that triggered for original_idx (now current_idx)
            if not self._drift_detection_disabled and current_idx < len(self._drift_detectors):
                 self._drift_detectors[current_idx] = self.drift_detector.clone()
            
            processed_drift_indices_this_step.add(original_idx) # Mark original index as processed

        # --- Prune again if ensemble grew beyond max_models (e.g. multiple drifts added trees)
        while len(self.data) > self.max_models:
            worst_idx = self._find_worst_model()
            if worst_idx is not None:
                # logging.info(f"✂️ Ensemble ({len(self.data)}) exceeds max_models ({self.max_models}). Pruning worst tree {worst_idx}.")
                self._remove_model(worst_idx)
            else:
                # logging.warning("Exceeds max_models, but no worst model to prune. Strange state.")
                break # Avoid infinite loop

        # --- Stage 3: Check Pruning based on Accuracy Drop ---
        self._check_prune_on_accuracy_drop(current_step) # This also calls _remove_model

        # --- Stage 4: Update Global State (dynamic weights) and Train ---
        self.n_models = len(self.data) # Update official count
        self._update_dynamic_weights() # Recalculate for current ensemble

        # Train all *current* models
        for i in range(len(self.data)): # Iterate up to current length of self.data
            model_to_train = self.data[i] # Get current model at index i
            k = poisson(rate=self.lambda_value, rng=self._rng)
            if k > 0:
                # Train background model if it exists (check bounds)
                if not self._warning_detection_disabled and self._background and i < len(self._background) and self._background[i] is not None:
                    self._background[i].learn_one(x=x, y=y, w=k) # type: ignore
                # Train the main model
                model_to_train.learn_one(x=x, y=y, w=k)
        return self

    def _get_recent_accuracy(self, tree_idx: int) -> float:
        if tree_idx >= len(self._accuracy_window): return 0.0 # Should not happen
        acc_deque = self._accuracy_window[tree_idx]
        return sum(acc_deque) / len(acc_deque) if acc_deque else 1.0 # Default to 1.0 if empty (optimistic)

    def _check_prune_on_accuracy_drop(self, current_step: int):
        indices_to_remove = []
        # Iterate over a copy of warned IDs as the set might change during iteration due to _clear_warning_state
        for i in list(self._warned_tree_ids):
            if i >= len(self.data): # Tree might have been removed by other means
                self._clear_warning_state(i)
                continue

            # If tree was reset due to drift, it should have been cleared from warned_ids.
            # But double check: if it was reset, its accuracy window is new.
            # The age since warning is key.
            age_since_warning = current_step - self._warning_step.get(i, current_step)

            # Stop monitoring if enough time has passed without significant drop
            # or if window is not full yet (no reliable current_acc)
            if age_since_warning > self.monitor_window * 1.5 or len(self._accuracy_window[i]) < self.monitor_window:
                if len(self._accuracy_window[i]) == self.monitor_window : # Only "survive" if monitored for full window
                    # logging.info(f"🌳 Tree {i} survived proxy accuracy drop monitoring.")
                    self._clear_warning_state(i)
                continue # Keep monitoring if window not full yet, unless very old

            current_proxy_acc = self._get_recent_accuracy(i)
            past_proxy_acc = self._warned_recent_acc.get(i, 1.0) # Acc at time of warning

            # Prune if current acc is significantly lower than acc at time of warning
            if past_proxy_acc > 1e-6 and current_proxy_acc < self.accuracy_drop_threshold * past_proxy_acc:
                # logging.info(f"⚠️ Tree {i} proxy accuracy dropped: {past_proxy_acc:.3f} -> {current_proxy_acc:.3f}. Pruning.")
                indices_to_remove.append(i)
                # _clear_warning_state(i) will be called by _remove_model or after loop
            # else:
                # logging.debug(f"Tree {i} acc: {past_proxy_acc:.3f} -> {current_proxy_acc:.3f}. No prune.")

        # Remove marked trees (in reverse order to maintain index validity)
        for i in sorted(indices_to_remove, reverse=True):
            self._remove_model(i) # This will also clear warning state for 'i'

    def _find_worst_model(self) -> int | None:
        if not self.data: return None
        
        metric_values = []
        valid_indices = []
        for idx, m_metric in enumerate(self._metrics):
            if idx >= len(self.data): continue # Should not happen
            val = m_metric.get()
            if isinstance(val, (int, float)) and not (math.isnan(val) or math.isinf(val)):
                metric_values.append(val)
                valid_indices.append(idx)
        
        if not metric_values: return None

        # self.metric.bigger_is_better is False for typical regression errors (MSE, MAE)
        if self.metric.bigger_is_better: # We want to remove the one with the smallest good value
            worst_val_idx_in_list = np.argmin(metric_values)
        else: # We want to remove the one with the largest error value
            worst_val_idx_in_list = np.argmax(metric_values)
        
        return valid_indices[worst_val_idx_in_list]

    def _remove_model(self, index: int):
        if not (0 <= index < len(self.data)):
            # logging.warning(f"Attempted to remove model at invalid index {index}. Ensemble size: {len(self.data)}")
            return

        # removed_metric_score_val = 'N/A'
        # if index < len(self._metrics): removed_metric_score_val = self._metrics[index].get()
        # logging.info(f"🪓 Removing tree at index {index} (metric: {removed_metric_score_val}). Ensemble size: {len(self.data)-1}")

        del self.data[index]
        if index < len(self._metrics): del self._metrics[index]
        if not self._drift_detection_disabled and self._drift_detectors and index < len(self._drift_detectors):
            del self._drift_detectors[index]
        if not self._warning_detection_disabled and self._warning_detectors and index < len(self._warning_detectors):
            del self._warning_detectors[index]
        if self._background is not None and index < len(self._background):
            del self._background[index]
        if self._drift_norm and index < len(self._drift_norm): # Manage _drift_norm
            del self._drift_norm[index]
        if self._accuracy_window and index < len(self._accuracy_window):
            del self._accuracy_window[index]
        if self._dynamic_perf_scores and index < len(self._dynamic_perf_scores):
            del self._dynamic_perf_scores[index]
        # Note: _dynamic_weights gets rebuilt by _update_dynamic_weights, so direct removal isn't critical here

        # Adjust pruning tracker indices for elements that came after the removed one
        self._clear_warning_state(index) # Remove the exact index first
        new_warned_ids = set()
        new_warning_step = {}
        new_warned_recent_acc = {}
        for warned_idx in self._warned_tree_ids:
            if warned_idx > index: # Shift indices greater than the removed one
                new_idx = warned_idx - 1
                new_warned_ids.add(new_idx)
                if warned_idx in self._warning_step: new_warning_step[new_idx] = self._warning_step[warned_idx]
                if warned_idx in self._warned_recent_acc: new_warned_recent_acc[new_idx] = self._warned_recent_acc[warned_idx]
            # elif warned_idx < index: # Indices before the removed one are unaffected, keep them (implicitly handled as we build new set)
            #    new_warned_ids.add(warned_idx) # This line is redundant if we iterate self._warned_tree_ids and only modify for > index
            #    if warned_idx in self._warning_step: new_warning_step[warned_idx] = self._warning_step[warned_idx]
            #    if warned_idx in self._warned_recent_acc: new_warned_recent_acc[warned_idx] = self._warned_recent_acc[warned_idx]
            elif warned_idx < index: # explicit re-add for clarity
                 new_warned_ids.add(warned_idx)
                 if warned_idx in self._warning_step: new_warning_step[warned_idx] = self._warning_step[warned_idx]
                 if warned_idx in self._warned_recent_acc: new_warned_recent_acc[warned_idx] = self._warned_recent_acc[warned_idx]

        self._warned_tree_ids = new_warned_ids
        self._warning_step = new_warning_step
        self._warned_recent_acc = new_warned_recent_acc
        
        # self.n_models = len(self.data) # Updated in learn_one after all modifications

    def _clear_warning_state(self, index: int):
        self._warned_tree_ids.discard(index)
        self._warning_step.pop(index, None)
        self._warned_recent_acc.pop(index, None)

    def plot_model_count(self):
        if not self.model_count_history:
            print("No model count history to plot. Train the model first.")
            return
        plt.figure(figsize=(10, 5))
        plt.plot(self.model_count_history, label='Number of Active Models', drawstyle='steps-post')
        plt.xlabel("Instances Processed (or learn_one calls)")
        plt.ylabel("Number of Models")
        plt.title(f"{self.__class__.__name__} Ensemble Size Over Time")
        if hasattr(self, 'max_models'):
            plt.axhline(y=self.max_models, color='r', linestyle='--', label=f'Max Models ({self.max_models})')
        # Min models is usually implicitly 1 if pruning happens, or n_models if no pruning below initial
        # plt.axhline(y=self.n_models, color='g', linestyle=':', label=f'Initial Models ({self.n_models})') # self.n_models changes
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()