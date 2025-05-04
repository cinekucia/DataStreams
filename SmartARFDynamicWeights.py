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
from river.tree.hoeffding_tree_classifier import HoeffdingTreeClassifier
# Assuming HoeffdingTreeRegressor and its nodes are not needed, omitting for brevity
# from river.tree.hoeffding_tree_regressor import HoeffdingTreeRegressor
from river.tree.nodes.arf_htc_nodes import (
    RandomLeafMajorityClass,
    RandomLeafNaiveBayes,
    RandomLeafNaiveBayesAdaptive,
)
from river.tree.splitter import Splitter
from river.utils.random import poisson
import logging # Added for logging info
import matplotlib.pyplot as plt # Added for plotting

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ================================================================
# BaseForest, BaseTreeClassifier, ARFClassifier (as context, unchanged from previous response)
# ================================================================

class BaseForest(base.Ensemble):
    _FEATURES_SQRT = "sqrt"
    _FEATURES_LOG2 = "log2"
    def __init__( self, n_models: int, max_features: bool | str | int, lambda_value: int,
                  drift_detector: base.DriftDetector, warning_detector: base.DriftDetector,
                  metric: metrics.base.MultiClassMetric | metrics.base.RegressionMetric,
                  disable_weighted_vote, seed):
        super().__init__([]) ; self.n_models = n_models; self.max_features = max_features
        self.lambda_value = lambda_value; self.metric = metric
        self.disable_weighted_vote = disable_weighted_vote; self.drift_detector = drift_detector
        self.warning_detector = warning_detector; self.seed = seed; self._rng = random.Random(self.seed)
        self._warning_detectors: list[base.DriftDetector]; self._warning_detection_disabled = True
        if not isinstance(self.warning_detector, NoDrift):
            self._warning_detectors = [self.warning_detector.clone() for _ in range(self.n_models)]
            self._warning_detection_disabled = False
        self._drift_detectors: list[base.DriftDetector]; self._drift_detection_disabled = True
        if not isinstance(self.drift_detector, NoDrift):
            self._drift_detectors = [self.drift_detector.clone() for _ in range(self.n_models)]
            self._drift_detection_disabled = False
        self._background: list[BaseTreeClassifier | None] = None if self._warning_detection_disabled else [None] * self.n_models
        self._metrics = [self.metric.clone() for _ in range(self.n_models)]
        self._warning_tracker: dict = collections.defaultdict(int) if not self._warning_detection_disabled else None
        self._drift_tracker: dict = collections.defaultdict(int) if not self._drift_detection_disabled else None
    @property
    def _min_number_of_models(self): return 0
    @classmethod
    def _unit_test_params(cls): yield {"n_models": 3}
    def _unit_test_skips(self): return {"check_shuffle_features_no_impact"}
    @abc.abstractmethod
    def _drift_detector_input(self, tree_id: int, y_true, y_pred) -> int | float: raise NotImplementedError
    @abc.abstractmethod
    def _new_base_model(self) -> BaseTreeClassifier: raise NotImplementedError # Simplified for Classifier
    def n_warnings_detected(self, tree_id: int | None = None) -> int:
        if self._warning_detection_disabled: return 0
        count = 0
        num_models = len(self.data) # Use current ensemble size
        if self._warning_tracker is not None:
             if tree_id is None: count = sum(self._warning_tracker.values())
             elif tree_id < num_models: count = self._warning_tracker.get(tree_id, 0)
        return count
    def n_drifts_detected(self, tree_id: int | None = None) -> int:
        if self._drift_detection_disabled: return 0
        count = 0
        num_models = len(self.data) # Use current ensemble size
        if self._drift_tracker is not None:
             if tree_id is None: count = sum(self._drift_tracker.values())
             elif tree_id < num_models: count = self._drift_tracker.get(tree_id, 0)
        return count
    # learn_one MUST be implemented by subclasses like ARFClassifierDynamicWeights or SmartARFDynamicWeights
    def learn_one(self, x: dict, y: base.typing.Target, **kwargs): raise NotImplementedError
    def _init_ensemble(self, features: list):
        self._set_max_features(len(features))
        # Initialize standard metrics, detectors etc based on self.n_models (initial size)
        self.data = [self._new_base_model() for _ in range(self.n_models)]
        if not self._warning_detection_disabled: self._warning_detectors = [self.warning_detector.clone() for _ in range(self.n_models)]
        if not self._drift_detection_disabled: self._drift_detectors = [self.drift_detector.clone() for _ in range(self.n_models)]
        if not self._warning_detection_disabled: self._background = [None] * self.n_models
        self._metrics = [self.metric.clone() for _ in range(self.n_models)]
        if self._warning_tracker is not None: self._warning_tracker.clear()
        if self._drift_tracker is not None: self._drift_tracker.clear()

    def _set_max_features(self, n_features):
        orig_max_features = self.max_features # Store original setting
        if self.max_features == "sqrt": self.max_features = round(math.sqrt(n_features))
        elif self.max_features == "log2": self.max_features = round(math.log2(n_features))
        elif isinstance(self.max_features, int): pass
        elif isinstance(self.max_features, float): self.max_features = int(self.max_features * n_features)
        elif self.max_features is None: self.max_features = n_features
        else: raise AttributeError(f"Invalid max_features: {orig_max_features}...")
        if self.max_features < 0: self.max_features += n_features
        if self.max_features <= 0: self.max_features = 1
        if self.max_features > n_features: self.max_features = n_features


class BaseTreeClassifier(HoeffdingTreeClassifier):
    def __init__( self, max_features: int = 2, grace_period: int = 200, max_depth: int | None = None,
                  split_criterion: str = "info_gain", delta: float = 1e-7, tau: float = 0.05,
                  leaf_prediction: str = "nba", nb_threshold: int = 0, nominal_attributes: list | None = None,
                  splitter: Splitter | None = None, binary_split: bool = False, min_branch_fraction: float = 0.01,
                  max_share_to_split: float = 0.99, max_size: float = 100.0, memory_estimate_period: int = 1000000,
                  stop_mem_management: bool = False, remove_poor_attrs: bool = False, merit_preprune: bool = True,
                  rng: random.Random | None = None):
        super().__init__(grace_period=grace_period, max_depth=max_depth, split_criterion=split_criterion, delta=delta,
                         tau=tau, leaf_prediction=leaf_prediction, nb_threshold=nb_threshold,
                         nominal_attributes=nominal_attributes, splitter=splitter, binary_split=binary_split,
                         min_branch_fraction=min_branch_fraction, max_share_to_split=max_share_to_split, max_size=max_size,
                         memory_estimate_period=memory_estimate_period, stop_mem_management=stop_mem_management,
                         remove_poor_attrs=remove_poor_attrs, merit_preprune=merit_preprune); self.max_features = max_features; self.rng = rng
    def _new_leaf(self, initial_stats=None, parent=None):
        if initial_stats is None: initial_stats = {}
        depth = 0 if parent is None else parent.depth + 1
        if self._leaf_prediction == self._MAJORITY_CLASS: return RandomLeafMajorityClass(initial_stats, depth, self.splitter, self.max_features, self.rng)
        elif self._leaf_prediction == self._NAIVE_BAYES: return RandomLeafNaiveBayes(initial_stats, depth, self.splitter, self.max_features, self.rng)
        else: return RandomLeafNaiveBayesAdaptive(initial_stats, depth, self.splitter, self.max_features, self.rng)


class ARFClassifier(BaseForest, base.Classifier):
    """Original Adaptive Random Forest classifier. (Used for inheritance structure)"""
    def __init__( self, n_models: int = 10, max_features: bool | str | int = "sqrt", lambda_value: int = 6,
                  metric: metrics.base.MultiClassMetric | None = None, disable_weighted_vote=False,
                  drift_detector: base.DriftDetector | None = None, warning_detector: base.DriftDetector | None = None,
                  grace_period: int = 50, max_depth: int | None = None, split_criterion: str = "info_gain",
                  delta: float = 0.01, tau: float = 0.05, leaf_prediction: str = "nba", nb_threshold: int = 0,
                  nominal_attributes: list | None = None, splitter: Splitter | None = None, binary_split: bool = False,
                  min_branch_fraction: float = 0.01, max_share_to_split: float = 0.99, max_size: float = 100.0,
                  memory_estimate_period: int = 2_000_000, stop_mem_management: bool = False, remove_poor_attrs: bool = False,
                  merit_preprune: bool = True, seed: int | None = None):
        super().__init__( n_models=n_models, max_features=max_features, lambda_value=lambda_value,
                          metric=metric or metrics.Accuracy(), disable_weighted_vote=disable_weighted_vote,
                          drift_detector=drift_detector or ADWIN(delta=0.001),
                          warning_detector=warning_detector or ADWIN(delta=0.01), seed=seed)
        self.grace_period=grace_period; self.max_depth=max_depth; self.split_criterion=split_criterion; self.delta=delta
        self.tau=tau; self.leaf_prediction=leaf_prediction; self.nb_threshold=nb_threshold
        self.nominal_attributes=nominal_attributes; self.splitter=splitter; self.binary_split=binary_split
        self.min_branch_fraction=min_branch_fraction; self.max_share_to_split=max_share_to_split
        self.max_size=max_size; self.memory_estimate_period=memory_estimate_period
        self.stop_mem_management=stop_mem_management; self.remove_poor_attrs=remove_poor_attrs; self.merit_preprune=merit_preprune
    @property
    def _mutable_attributes(self): return {"max_features", "lambda_value", "grace_period", "delta", "tau"}
    @property
    def _multiclass(self): return True
    # Default learn_one for standard ARF
    def learn_one(self, x: dict, y: base.typing.Target, **kwargs):
        if len(self) == 0: self._init_ensemble(sorted(x.keys()))
        for i, model in enumerate(self):
            y_pred = model.predict_one(x)
            self._metrics[i].update(y_true=y, y_pred=(model.predict_proba_one(x) if isinstance(self.metric, metrics.base.ClassificationMetric) and not self.metric.requires_labels else y_pred))
            k = poisson(rate=self.lambda_value, rng=self._rng)
            if k > 0:
                drift_input = None
                if not self._warning_detection_disabled:
                    drift_input = self._drift_detector_input(i, y, y_pred)
                    self._warning_detectors[i].update(drift_input)
                    if self._warning_detectors[i].drift_detected:
                        if self._background is not None: self._background[i] = self._new_base_model()
                        self._warning_detectors[i] = self.warning_detector.clone()
                        if self._warning_tracker is not None: self._warning_tracker[i] += 1
                if not self._drift_detection_disabled:
                    drift_input = drift_input if drift_input is not None else self._drift_detector_input(i, y, y_pred)
                    self._drift_detectors[i].update(drift_input)
                    if self._drift_detectors[i].drift_detected:
                        if not self._warning_detection_disabled and self._background is not None and self._background[i] is not None:
                            self.data[i] = self._background[i]; self._background[i] = None
                        else: self.data[i] = self._new_base_model()
                        if not self._warning_detection_disabled: self._warning_detectors[i] = self.warning_detector.clone()
                        self._drift_detectors[i] = self.drift_detector.clone()
                        self._metrics[i] = self.metric.clone()
                        if self._drift_tracker is not None: self._drift_tracker[i] += 1
                # Train models (including background if active)
                if not self._warning_detection_disabled and self._background is not None and self._background[i] is not None:
                    self._background[i].learn_one(x=x, y=y, w=k)
                model.learn_one(x=x, y=y, w=k)
        return self

    def predict_proba_one(self, x: dict) -> dict[base.typing.ClfTarget, float]:
        y_pred: typing.Counter = collections.Counter()
        if len(self) == 0: self._init_ensemble(sorted(x.keys())); return {}
        for i, model in enumerate(self):
            y_proba_temp = model.predict_proba_one(x); metric_value = self._metrics[i].get()
            # Default ARF weighting based on self._metrics
            if not self.disable_weighted_vote and isinstance(metric_value, (int, float)) and metric_value > 0.0:
                 y_proba_temp = {k: val * metric_value for k, val in y_proba_temp.items()}
            elif self.disable_weighted_vote: # Equal weight if disabled
                 pass # Effectively weight = 1
            else: # Handle cases where metric is bad (e.g., 0 or nan) - treat as equal weight
                 pass
            y_pred.update(y_proba_temp)
        total = sum(y_pred.values()); return {label: proba / total for label, proba in y_pred.items()} if total > 0 else {}
    def _new_base_model(self):
        return BaseTreeClassifier(max_features=self.max_features, grace_period=self.grace_period, split_criterion=self.split_criterion,
                                 delta=self.delta, tau=self.tau, leaf_prediction=self.leaf_prediction, nb_threshold=self.nb_threshold,
                                 nominal_attributes=self.nominal_attributes, splitter=self.splitter, max_depth=self.max_depth,
                                 binary_split=self.binary_split, min_branch_fraction=self.min_branch_fraction,
                                 max_share_to_split=self.max_share_to_split, max_size=self.max_size,
                                 memory_estimate_period=self.memory_estimate_period, stop_mem_management=self.stop_mem_management,
                                 remove_poor_attrs=self.remove_poor_attrs, merit_preprune=self.merit_preprune, rng=self._rng)
    def _drift_detector_input(self, tree_id: int, y_true: base.typing.ClfTarget, y_pred: base.typing.ClfTarget) -> int | float: return int(not y_true == y_pred)

# ================================================================
# ARFClassifierDynamicWeights (Base for the new class) - WITH learn_one ADDED
# ================================================================
class ARFClassifierDynamicWeights(ARFClassifier):
    """ Base class providing dynamic weighting (0.9/1.1 rule). """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dynamic_perf_scores: list[float] = []
        self._dynamic_weights: list[float] = []
        if len(self) > 0: self._init_dynamic_weights() # If ensemble already exists

    def _init_dynamic_weights(self):
        num_models = len(self.data) # Use actual number of models
        self._dynamic_perf_scores = [1.0] * num_models
        equal_weight = 1.0 / num_models if num_models > 0 else 1.0
        self._dynamic_weights = [equal_weight] * num_models

    def _init_ensemble(self, features: list):
        super()._init_ensemble(features)
        self._init_dynamic_weights()

    def _update_dynamic_weights(self):
        num_models = len(self.data)
        if not self._dynamic_perf_scores or len(self._dynamic_perf_scores) != num_models:
             self._init_dynamic_weights()
             if num_models == 0: return

        raw_weights = [1.0 / (1.0 + score) for score in self._dynamic_perf_scores]
        total_weight = sum(raw_weights)
        if total_weight > 0: self._dynamic_weights = [w / total_weight for w in raw_weights]
        else:
            equal_weight = 1.0 / num_models if num_models > 0 else 1.0
            self._dynamic_weights = [equal_weight] * num_models

    # --- ADDED learn_one specific to dynamic weights ---
    def learn_one(self, x: dict, y: base.typing.Target, **kwargs):
        """ Learns from an instance, updates dynamic performance scores/weights. """
        if len(self) == 0: self._init_ensemble(sorted(x.keys()))

        tree_predictions = [None] * len(self.data)

        # --- Stage 1: Predictions and Metric/Score Updates ---
        for i, model in enumerate(self):
            y_pred_tree = model.predict_one(x)
            tree_predictions[i] = y_pred_tree
            # Update standard metric (for compatibility/drift maybe)
            self._metrics[i].update(y_true=y, y_pred=(model.predict_proba_one(x) if isinstance(self.metric, metrics.base.ClassificationMetric) and not self.metric.requires_labels else y_pred_tree))
            # Update dynamic performance score
            if y_pred_tree == y: self._dynamic_perf_scores[i] *= 0.9
            else: self._dynamic_perf_scores[i] *= 1.1

        # --- Stage 2: Drift/Warning Detection and Tree Management (Standard ARF logic) ---
        for i, model in enumerate(self):
            y_pred_tree = tree_predictions[i]
            drift_input = None
            # Check for warnings
            if not self._warning_detection_disabled:
                drift_input = self._drift_detector_input(i, y, y_pred_tree)
                self._warning_detectors[i].update(drift_input)
                if self._warning_detectors[i].drift_detected:
                    if self._background is not None: self._background[i] = self._new_base_model()
                    self._warning_detectors[i] = self.warning_detector.clone()
                    if self._warning_tracker is not None: self._warning_tracker[i] += 1
            # Check for drifts
            if not self._drift_detection_disabled:
                drift_input = drift_input if drift_input is not None else self._drift_detector_input(i, y, y_pred_tree)
                self._drift_detectors[i].update(drift_input)
                if self._drift_detectors[i].drift_detected:
                    # Reset or replace tree
                    if not self._warning_detection_disabled and self._background is not None and self._background[i] is not None:
                        self.data[i] = self._background[i]; self._background[i] = None
                    else: self.data[i] = self._new_base_model()
                    # Reset detectors and standard metrics
                    if not self._warning_detection_disabled: self._warning_detectors[i] = self.warning_detector.clone()
                    self._drift_detectors[i] = self.drift_detector.clone()
                    self._metrics[i] = self.metric.clone()
                    # *** Crucially: Reset dynamic weight score for the replaced tree ***
                    self._dynamic_perf_scores[i] = 1.0 # Reset score to neutral
                    if self._drift_tracker is not None: self._drift_tracker[i] += 1

        # --- Stage 3: Recalculate Weights and Train Models ---
        self._update_dynamic_weights() # Update based on scores

        for i, model in enumerate(self):
            k = poisson(rate=self.lambda_value, rng=self._rng)
            if k > 0:
                if not self._warning_detection_disabled and self._background is not None and self._background[i] is not None:
                    self._background[i].learn_one(x=x, y=y, w=k)
                model.learn_one(x=x, y=y, w=k)
        return self
    # --- End of added learn_one ---

    # predict_proba_one uses dynamic weights (defined in previous response, slightly adjusted here)
    def predict_proba_one(self, x: dict) -> dict[base.typing.ClfTarget, float]:
        y_pred_proba: typing.Counter = collections.Counter()
        if len(self) == 0:
            self._init_ensemble(sorted(x.keys()))
            return {}
        if not self._dynamic_weights or len(self._dynamic_weights) != len(self.data):
            self._init_dynamic_weights()

        # Use the current number of models for iteration
        num_models = len(self.data)
        for i in range(num_models):
            model = self.data[i]
            # Check index bounds again, just in case
            if i >= len(self._dynamic_weights):
                # logging.error(f"Weight index {i} out of bounds ({len(self._dynamic_weights)}) in predict. Re-init.")
                self._init_dynamic_weights() # Try to recover
                # Ensure weights list matches current model count after potential recovery
                if i >= len(self._dynamic_weights): continue # Skip if still out of bounds

            y_proba_tree = model.predict_proba_one(x)
            weight = self._dynamic_weights[i]
            if weight > 0.0:
                for label, proba in y_proba_tree.items(): y_pred_proba[label] += proba * weight

        total_proba = sum(y_pred_proba.values())
        return {label: proba / total_proba for label, proba in y_pred_proba.items()} if total_proba > 0 else {}


# ================================================================
# Integrated Class: SmartARFDynamicWeights (Inherits the above learn_one structure)
# ================================================================

class SmartARFDynamicWeights(ARFClassifierDynamicWeights):
    """
    Adaptive Random Forest Classifier that combines:
    1. Dynamic tree weighting (0.9/1.1 multiplicative score update).
    2. Dynamic ensemble size management (adding trees on drift, pruning on
       accuracy drop or exceeding max_models).
    """
    def __init__(
        self,
        n_models: int = 10,               # Initial number of models
        max_models: int = 30,             # Max number of models allowed
        accuracy_drop_threshold: float = 0.5, # Pruning threshold for accuracy drop
        monitor_window: int = 100,        # Window size for accuracy monitoring
        **kwargs                         # Pass other ARF params (seed, grace_period etc)
    ):
        super().__init__(n_models=n_models, **kwargs)
        self.max_models = max_models
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.monitor_window = monitor_window
        self.model_count_history = []
        self._accuracy_window: list[list[int]] = []
        self._warned_tree_ids: set[int] = set()
        self._warning_step: dict[int, int] = {}
        self._warned_recent_acc: dict[int, float] = {}
        if len(self) > 0: self._init_pruning_state()

    def _init_pruning_state(self):
        num_models = len(self.data)
        self._accuracy_window = [collections.deque(maxlen=self.monitor_window) for _ in range(num_models)] # Use deque
        self._warned_tree_ids.clear()
        self._warning_step.clear()
        self._warned_recent_acc.clear()

    def _init_ensemble(self, features: list):
        super()._init_ensemble(features)
        self._init_pruning_state()

    # Override learn_one to add smart pruning/adding logic
    def learn_one(self, x: dict, y: base.typing.Target, **kwargs):
        if len(self) == 0: self._init_ensemble(sorted(x.keys()))

        # Use a more stable step counter, e.g., number of calls to learn_one
        # Or track total weight processed if using weighted learning meaningfully elsewhere
        current_step = len(self.model_count_history)
        self.model_count_history.append(len(self.data))

        # --- Stage 0: Prepare for iteration ---
        num_models = len(self.data)
        tree_predictions = [None] * num_models
        drift_detected_indices = []
        warning_detected_indices = []

        # --- Stage 1: Predictions and Local Updates ---
        for i in range(num_models):
            # Ensure index validity before accessing model data
            if i >= len(self.data): continue # Should not happen if num_models is updated correctly

            model = self.data[i]
            y_pred_tree = model.predict_one(x)
            tree_predictions[i] = y_pred_tree

            # Update standard metric (used for finding worst model)
            self._metrics[i].update(y_true=y, y_pred=y_pred_tree)

            # Update dynamic performance score (for weighting)
            if i < len(self._dynamic_perf_scores): # Check bounds for dynamic score list
                 if y_pred_tree == y: self._dynamic_perf_scores[i] *= 0.9
                 else: self._dynamic_perf_scores[i] *= 1.1
            else:
                 logging.warning(f"Dynamic score index {i} out of bounds during update.")

            # Update accuracy window (for pruning)
            self._ensure_accuracy_window_exists(i) # Ensure deque exists
            self._accuracy_window[i].append(int(y_pred_tree == y)) # Deque handles maxlen

            # --- Check Detectors ---
            if not self._warning_detection_disabled and i < len(self._warning_detectors):
                 warn_input = self._drift_detector_input(i, y, y_pred_tree)
                 self._warning_detectors[i].update(warn_input)
                 if self._warning_detectors[i].drift_detected: warning_detected_indices.append(i)
            if not self._drift_detection_disabled and i < len(self._drift_detectors):
                 drift_input = self._drift_detector_input(i, y, y_pred_tree)
                 self._drift_detectors[i].update(drift_input)
                 if self._drift_detectors[i].drift_detected: drift_detected_indices.append(i)

        # --- Stage 2: Process Detections and Manage Ensemble Size ---
        indices_needing_reset = set() # Track indices where tree was reset in place

        # Handle warnings
        for i in warning_detected_indices:
            if i < len(self.data): # Check if index is still valid
                if self._background is not None and i < len(self._background) and self._background[i] is None:
                    self._background[i] = self._new_base_model()
                    # logging.info(f"🌳 Background learner started for tree {i} due to warning.")
                if i < len(self._warning_detectors): self._warning_detectors[i] = self.warning_detector.clone()
                if self._warning_tracker is not None: self._warning_tracker[i] += 1
                if i not in self._warned_tree_ids:
                     self._warned_tree_ids.add(i)
                     self._warning_step[i] = current_step
                     self._warned_recent_acc[i] = self._get_recent_accuracy(i)
                     # logging.info(f"📉 Started monitoring accuracy drop for tree {i} (current acc: {self._warned_recent_acc[i]:.3f}).")

        # Handle drifts
        added_model_this_step = False
        indices_processed_for_drift = set() # Avoid double processing if multiple detectors trigger for same index conceptually
        for i in drift_detected_indices:
             if i in indices_processed_for_drift or i >= len(self.data): continue # Skip if already handled or index invalid

             # logging.info(f"💥 Drift detected by detector for tree {i}.")
             if self._drift_tracker is not None: self._drift_tracker[i] += 1

             # Attempt to add new model from background
             promoted_from_background = False
             if self._background is not None and i < len(self._background) and self._background[i] is not None:
                 new_tree = self._background[i]
                 self._background[i] = None # Consume background tree
                 promoted_from_background = True

                 if len(self.data) >= self.max_models:
                     worst_idx = self._find_worst_model()
                     if worst_idx is not None:
                         # logging.info(f"📦 Ensemble at max capacity ({self.max_models}). Pruning worst tree {worst_idx} before adding.")
                         self._remove_model(worst_idx)
                         # Adjust 'i' if the removed index was before it
                         if worst_idx < i: i -= 1
                         # Recheck capacity after pruning
                         if len(self.data) >= self.max_models:
                              logging.warning("Still at max capacity after pruning, cannot add new tree.")
                              continue # Skip adding this drift-triggered tree
                     else: continue # Cannot add if cannot prune

                 # Add the new model and its state
                 # logging.info(f"➕ Adding new model to ensemble (from background of tree {i}). Ensemble size: {len(self.data)+1}")
                 self.data.append(new_tree)
                 self._metrics.append(self.metric.clone())
                 if not self._drift_detection_disabled: self._drift_detectors.append(self.drift_detector.clone())
                 if not self._warning_detection_disabled: self._warning_detectors.append(self.warning_detector.clone())
                 if self._background is not None: self._background.append(None)
                 self._accuracy_window.append(collections.deque(maxlen=self.monitor_window)) # Use deque
                 self._dynamic_perf_scores.append(1.0)
                 self.n_models = len(self.data) # Update official count
                 added_model_this_step = True

             else: # No background tree, reset the existing tree
                 # logging.info(f"🔁 Resetting tree {i} in place due to drift (no background learner).")
                 self.data[i] = self._new_base_model()
                 self._metrics[i] = self.metric.clone()
                 if not self._drift_detection_disabled and i < len(self._drift_detectors): self._drift_detectors[i] = self.drift_detector.clone()
                 if not self._warning_detection_disabled and i < len(self._warning_detectors): self._warning_detectors[i] = self.warning_detector.clone()
                 if i < len(self._accuracy_window): self._accuracy_window[i].clear() # Reset deque
                 if i < len(self._dynamic_perf_scores): self._dynamic_perf_scores[i] = 1.0
                 indices_needing_reset.add(i)
                 if i in self._warned_tree_ids: self._clear_warning_state(i) # Stop monitoring if reset

             # Reset the specific drift detector that triggered, regardless of action taken
             if not self._drift_detection_disabled and i < len(self._drift_detectors):
                 self._drift_detectors[i] = self.drift_detector.clone()

             indices_processed_for_drift.add(i)

        # Handle pruning based on accuracy drop
        self._check_prune_on_accuracy_drop(current_step)

        # --- Stage 3: Update Global State and Train ---
        # Recalculate weights AFTER potential additions/removals/resets
        self._update_dynamic_weights()

        # Train all *current* models
        num_models_after_update = len(self.data)
        for i in range(num_models_after_update):
            # Check if model exists at this index before training
            if i >= len(self.data): continue

            model = self.data[i]
            k = poisson(rate=self.lambda_value, rng=self._rng)
            if k > 0:
                 # Train background model if it exists
                 if not self._warning_detection_disabled and self._background is not None and i < len(self._background) and self._background[i] is not None:
                    self._background[i].learn_one(x=x, y=y, w=k)
                 # Train the main model
                 model.learn_one(x=x, y=y, w=k)
        return self

    # --- Helper methods for SmartARF logic ---
    def _check_prune_on_accuracy_drop(self, current_step):
        indices_to_remove = []
        # Iterate over copy as set might change
        for i in list(self._warned_tree_ids):
            if i >= len(self.data): # Check index validity
                self._clear_warning_state(i); continue
            age_since_warning = current_step - self._warning_step.get(i, current_step)
            # Stop monitoring if enough time has passed
            if age_since_warning > self.monitor_window * 1.5:
                # logging.info(f"🌳 Tree {i} survived accuracy drop monitoring.")
                self._clear_warning_state(i); continue
            current_acc = self._get_recent_accuracy(i)
            past_acc = self._warned_recent_acc.get(i, 1.0)
            if past_acc > 1e-6 and current_acc < self.accuracy_drop_threshold * past_acc:
                logging.info(f"⚠️ Tree {i} accuracy dropped: {past_acc:.3f} -> {current_acc:.3f}. Pruning.")
                indices_to_remove.append(i)
                self._clear_warning_state(i)
        # Remove marked trees (reverse order)
        for i in sorted(indices_to_remove, reverse=True): self._remove_model(i)

    def _get_recent_accuracy(self, i):
        if i >= len(self._accuracy_window): return 0.0
        accs = self._accuracy_window[i]
        # Use deque length for accuracy calculation
        return sum(accs) / len(accs) if accs else 1.0

    def _ensure_accuracy_window_exists(self, i):
        # Adjusted for deques
        while len(self._accuracy_window) <= i:
            self._accuracy_window.append(collections.deque(maxlen=self.monitor_window))

    def _find_worst_model(self) -> int | None:
        if not self.data: return None # Check if data is empty
        metric_values = []
        valid_indices = []
        for idx, m in enumerate(self._metrics):
             # Ensure index is valid for the metric list itself
             if idx >= len(self._metrics): continue
             val = m.get()
             if isinstance(val, (int, float)):
                 metric_values.append(val)
                 valid_indices.append(idx)
        if not metric_values: return None
        if self.metric.bigger_is_better: worst_val_idx = np.argmin(metric_values)
        else: worst_val_idx = np.argmax(metric_values)
        return valid_indices[worst_val_idx]

    def _remove_model(self, index):
        if not (0 <= index < len(self.data)): return # Check bounds

        removed_metric_score = 'N/A'
        if index < len(self._metrics): removed_metric_score = self._metrics[index].get()
        # logging.info(f"🪓 Removing tree at index {index} (score: {removed_metric_score}). Ensemble size: {len(self.data)-1}")

        # --- Remove state carefully, checking bounds ---
        del self.data[index]
        if index < len(self._metrics): del self._metrics[index]
        if not self._drift_detection_disabled and self._drift_detectors and index < len(self._drift_detectors): del self._drift_detectors[index]
        if not self._warning_detection_disabled and self._warning_detectors and index < len(self._warning_detectors): del self._warning_detectors[index]
        if self._background is not None and index < len(self._background): del self._background[index]
        if self._accuracy_window and index < len(self._accuracy_window): del self._accuracy_window[index]
        if self._dynamic_perf_scores and index < len(self._dynamic_perf_scores): del self._dynamic_perf_scores[index]

        # --- Adjust pruning tracker indices ---
        self._clear_warning_state(index) # Remove exact index first
        new_warned_ids = set()
        new_warning_step = {}
        new_warned_recent_acc = {}
        for warned_idx in self._warned_tree_ids:
            if warned_idx > index:
                new_idx = warned_idx - 1
                new_warned_ids.add(new_idx)
                if warned_idx in self._warning_step: new_warning_step[new_idx] = self._warning_step[warned_idx]
                if warned_idx in self._warned_recent_acc: new_warned_recent_acc[new_idx] = self._warned_recent_acc[warned_idx]
            elif warned_idx < index: # Keep indices before
                 new_warned_ids.add(warned_idx)
                 if warned_idx in self._warning_step: new_warning_step[warned_idx] = self._warning_step[warned_idx]
                 if warned_idx in self._warned_recent_acc: new_warned_recent_acc[warned_idx] = self._warned_recent_acc[warned_idx]
        self._warned_tree_ids = new_warned_ids
        self._warning_step = new_warning_step
        self._warned_recent_acc = new_warned_recent_acc

        # Update official model count
        self.n_models = len(self.data)
        # Recalculate weights immediately after removal? Good practice.
        # self._update_dynamic_weights() # Moved out, done once per step in learn_one

    def _clear_warning_state(self, index):
        self._warned_tree_ids.discard(index)
        self._warning_step.pop(index, None)
        self._warned_recent_acc.pop(index, None)

    def plot_model_count(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.model_count_history, label='Number of Active Models', drawstyle='steps-post')
        plt.xlabel("Instances Processed")
        plt.ylabel("Number of Models")
        plt.title("SmartARFDynamicWeights Ensemble Size Over Time")
        if hasattr(self, 'max_models'): plt.axhline(y=self.max_models, color='r', linestyle='--', label=f'Max Models ({self.max_models})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
