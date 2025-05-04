# Import necessary components from the original code provided
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
from river.tree.hoeffding_tree_regressor import HoeffdingTreeRegressor
from river.tree.nodes.arf_htc_nodes import (
    RandomLeafMajorityClass,
    RandomLeafNaiveBayes,
    RandomLeafNaiveBayesAdaptive,
)
# Note: Tree Regressor and its nodes are not strictly needed for Classifier,
# but included here for completeness if adapting BaseForest structure was necessary.
# from river.tree.nodes.arf_htr_nodes import RandomLeafAdaptive, RandomLeafMean, RandomLeafModel
from river.tree.splitter import Splitter
from river.utils.random import poisson

# Assuming BaseForest, BaseTreeClassifier, BaseTreeRegressor are defined as in the provided code
# (They are copy-pasted below for completeness within this block)

# ================================================================
# COPIED BaseForest, BaseTreeClassifier, BaseTreeRegressor for context
# (No changes needed in these base classes for this specific modification)
# ================================================================

class BaseForest(base.Ensemble):
    _FEATURES_SQRT = "sqrt"
    _FEATURES_LOG2 = "log2"

    def __init__(
        self,
        n_models: int,
        max_features: bool | str | int,
        lambda_value: int,
        drift_detector: base.DriftDetector,
        warning_detector: base.DriftDetector,
        metric: metrics.base.MultiClassMetric | metrics.base.RegressionMetric,
        disable_weighted_vote, # Keep this parameter for potential future use or clarity
        seed,
    ):
        super().__init__([])  # type: ignore
        self.n_models = n_models
        self.max_features = max_features
        self.lambda_value = lambda_value
        self.metric = metric # Standard metric, kept for drift/warning but not for voting in derived class
        self.disable_weighted_vote = disable_weighted_vote # Kept, but interpretation might change
        self.drift_detector = drift_detector
        self.warning_detector = warning_detector
        self.seed = seed

        self._rng = random.Random(self.seed)

        self._warning_detectors: list[base.DriftDetector]
        self._warning_detection_disabled = True
        if not isinstance(self.warning_detector, NoDrift):
            self._warning_detectors = [self.warning_detector.clone() for _ in range(self.n_models)]
            self._warning_detection_disabled = False

        self._drift_detectors: list[base.DriftDetector]
        self._drift_detection_disabled = True
        if not isinstance(self.drift_detector, NoDrift):
            self._drift_detectors = [self.drift_detector.clone() for _ in range(self.n_models)]
            self._drift_detection_disabled = False

        # The background models
        self._background: list[BaseTreeClassifier | BaseTreeRegressor | None] = (
            None if self._warning_detection_disabled else [None] * self.n_models  # type: ignore
        )

        # Performance metrics used for weighted voting/aggregation (Standard way)
        # In the derived class, we'll use a different mechanism for weighting.
        self._metrics = [self.metric.clone() for _ in range(self.n_models)]

        # Drift and warning logging
        self._warning_tracker: dict = (
            collections.defaultdict(int) if not self._warning_detection_disabled else None  # type: ignore
        )
        self._drift_tracker: dict = (
            collections.defaultdict(int) if not self._drift_detection_disabled else None  # type: ignore
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
    def _new_base_model(self) -> BaseTreeClassifier | BaseTreeRegressor:
        raise NotImplementedError

    def n_warnings_detected(self, tree_id: int | None = None) -> int:
        if self._warning_detection_disabled: return 0
        if tree_id is None: return sum(self._warning_tracker.values())
        return self._warning_tracker[tree_id]

    def n_drifts_detected(self, tree_id: int | None = None) -> int:
        if self._drift_detection_disabled: return 0
        if tree_id is None: return sum(self._drift_tracker.values())
        return self._drift_tracker[tree_id]

    # learn_one will be overridden in the derived class to include weight updates
    def learn_one(self, x: dict, y: base.typing.Target, **kwargs):
         if len(self) == 0:
             self._init_ensemble(sorted(x.keys()))

         for i, model in enumerate(self):
             y_pred = model.predict_one(x) # Get prediction from THIS tree

             # Update standard performance evaluator (still useful for drift/warning)
             self._metrics[i].update(
                 y_true=y,
                 y_pred=(
                     model.predict_proba_one(x)
                     if isinstance(self.metric, metrics.base.ClassificationMetric)
                     and not self.metric.requires_labels
                     else y_pred
                 ),
             )

             # Original drift/warning logic (requires individual tree prediction)
             drift_input = None
             # Check for warnings
             if not self._warning_detection_disabled:
                 drift_input = self._drift_detector_input(i, y, y_pred) # Use individual prediction
                 self._warning_detectors[i].update(drift_input)
                 if self._warning_detectors[i].drift_detected:
                     if self._background is not None: # Check background is initialized
                         self._background[i] = self._new_base_model()  # type: ignore
                     self._warning_detectors[i] = self.warning_detector.clone()
                     self._warning_tracker[i] += 1

             # Check for drifts
             if not self._drift_detection_disabled:
                 drift_input = (
                     drift_input
                     if drift_input is not None
                     else self._drift_detector_input(i, y, y_pred) # Use individual prediction
                 )
                 self._drift_detectors[i].update(drift_input)
                 if self._drift_detectors[i].drift_detected:
                     # Reset or replace tree
                     if not self._warning_detection_disabled and self._background is not None and self._background[i] is not None:
                         self.data[i] = self._background[i]
                         self._background[i] = None # Clear background slot
                     else:
                         self.data[i] = self._new_base_model()

                     # Reset detectors and metrics for the replaced tree
                     if not self._warning_detection_disabled:
                        self._warning_detectors[i] = self.warning_detector.clone()
                     self._drift_detectors[i] = self.drift_detector.clone()
                     self._metrics[i] = self.metric.clone()
                     # *** NEED TO RESET DYNAMIC WEIGHTS HERE (will add in derived class) ***
                     self._drift_tracker[i] += 1


             # Train model (original logic)
             k = poisson(rate=self.lambda_value, rng=self._rng)
             if k > 0:
                 # Train background model if it exists
                 if not self._warning_detection_disabled and self._background is not None and self._background[i] is not None:
                    self._background[i].learn_one(x=x, y=y, w=k)  # type: ignore
                 # Train the main model
                 model.learn_one(x=x, y=y, w=k)

    def _init_ensemble(self, features: list):
        self._set_max_features(len(features))
        self.data = [self._new_base_model() for _ in range(self.n_models)]
        # *** NEED TO INIT DYNAMIC WEIGHTS HERE (will add in derived class) ***

    def _set_max_features(self, n_features):
        # (Copied logic as provided)
        if self.max_features == "sqrt": self.max_features = round(math.sqrt(n_features))
        elif self.max_features == "log2": self.max_features = round(math.log2(n_features))
        elif isinstance(self.max_features, int): pass
        elif isinstance(self.max_features, float): self.max_features = int(self.max_features * n_features)
        elif self.max_features is None: self.max_features = n_features
        else: raise AttributeError(f"Invalid max_features: {self.max_features}...")
        if self.max_features < 0: self.max_features += n_features
        if self.max_features <= 0: self.max_features = 1
        if self.max_features > n_features: self.max_features = n_features

class BaseTreeClassifier(HoeffdingTreeClassifier):
    # (Copied as provided - No changes needed here)
    def __init__(
        self, max_features: int = 2, grace_period: int = 200, max_depth: int | None = None,
        split_criterion: str = "info_gain", delta: float = 1e-7, tau: float = 0.05,
        leaf_prediction: str = "nba", nb_threshold: int = 0, nominal_attributes: list | None = None,
        splitter: Splitter | None = None, binary_split: bool = False, min_branch_fraction: float = 0.01,
        max_share_to_split: float = 0.99, max_size: float = 100.0, memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False, remove_poor_attrs: bool = False, merit_preprune: bool = True,
        rng: random.Random | None = None,
    ):
        super().__init__(
            grace_period=grace_period, max_depth=max_depth, split_criterion=split_criterion, delta=delta,
            tau=tau, leaf_prediction=leaf_prediction, nb_threshold=nb_threshold,
            nominal_attributes=nominal_attributes, splitter=splitter, binary_split=binary_split,
            min_branch_fraction=min_branch_fraction, max_share_to_split=max_share_to_split, max_size=max_size,
            memory_estimate_period=memory_estimate_period, stop_mem_management=stop_mem_management,
            remove_poor_attrs=remove_poor_attrs, merit_preprune=merit_preprune,
        )
        self.max_features = max_features
        self.rng = rng

    def _new_leaf(self, initial_stats=None, parent=None):
        # (Copied logic as provided)
        if initial_stats is None: initial_stats = {}
        depth = 0 if parent is None else parent.depth + 1
        if self._leaf_prediction == self._MAJORITY_CLASS:
            return RandomLeafMajorityClass(initial_stats, depth, self.splitter, self.max_features, self.rng)
        elif self._leaf_prediction == self._NAIVE_BAYES:
            return RandomLeafNaiveBayes(initial_stats, depth, self.splitter, self.max_features, self.rng)
        else: # NAIVE BAYES ADAPTIVE (default)
            return RandomLeafNaiveBayesAdaptive(initial_stats, depth, self.splitter, self.max_features, self.rng)

class BaseTreeRegressor(HoeffdingTreeRegressor):
    # (Copied as provided - Not needed for Classifier but kept for structure)
    def __init__(self, max_features: int = 2, grace_period: int = 200, max_depth: int | None = None,
                 delta: float = 1e-7, tau: float = 0.05, leaf_prediction: str = "adaptive",
                 leaf_model: base.Regressor | None = None, model_selector_decay: float = 0.95,
                 nominal_attributes: list | None = None, splitter: Splitter | None = None,
                 min_samples_split: int = 5, binary_split: bool = False, max_size: float = 100.0,
                 memory_estimate_period: int = 1000000, stop_mem_management: bool = False,
                 remove_poor_attrs: bool = False, merit_preprune: bool = True,
                 rng: random.Random | None = None):
         super().__init__(grace_period=grace_period, max_depth=max_depth, delta=delta, tau=tau,
                          leaf_prediction=leaf_prediction, leaf_model=leaf_model,
                          model_selector_decay=model_selector_decay, nominal_attributes=nominal_attributes,
                          splitter=splitter, min_samples_split=min_samples_split, binary_split=binary_split,
                          max_size=max_size, memory_estimate_period=memory_estimate_period,
                          stop_mem_management=stop_mem_management, remove_poor_attrs=remove_poor_attrs,
                          merit_preprune=merit_preprune)
         self.max_features = max_features
         self.rng = rng
    # _new_leaf method for Regressor also copied but commented out for brevity
    # def _new_leaf(self, initial_stats=None, parent=None): ...


# ================================================================
# Original ARFClassifier - We will inherit from this
# ================================================================
class ARFClassifier(BaseForest, base.Classifier):
    """Adaptive Random Forest classifier. (Original Docstring)"""
    def __init__(
        self,
        n_models: int = 10,
        max_features: bool | str | int = "sqrt",
        lambda_value: int = 6,
        metric: metrics.base.MultiClassMetric | None = None,
        disable_weighted_vote=False, # This flag will be ignored by the dynamic weighting version
        drift_detector: base.DriftDetector | None = None,
        warning_detector: base.DriftDetector | None = None,
        # Tree parameters
        grace_period: int = 50, max_depth: int | None = None, split_criterion: str = "info_gain",
        delta: float = 0.01, tau: float = 0.05, leaf_prediction: str = "nba",
        nb_threshold: int = 0, nominal_attributes: list | None = None, splitter: Splitter | None = None,
        binary_split: bool = False, min_branch_fraction: float = 0.01, max_share_to_split: float = 0.99,
        max_size: float = 100.0, memory_estimate_period: int = 2_000_000,
        stop_mem_management: bool = False, remove_poor_attrs: bool = False,
        merit_preprune: bool = True, seed: int | None = None,
    ):
        super().__init__(
            n_models=n_models, max_features=max_features, lambda_value=lambda_value,
            metric=metric or metrics.Accuracy(), disable_weighted_vote=disable_weighted_vote,
            drift_detector=drift_detector or ADWIN(delta=0.001),
            warning_detector=warning_detector or ADWIN(delta=0.01), seed=seed,
        )
        # Tree parameters stored
        self.grace_period = grace_period; self.max_depth = max_depth; self.split_criterion = split_criterion
        self.delta = delta; self.tau = tau; self.leaf_prediction = leaf_prediction; self.nb_threshold = nb_threshold
        self.nominal_attributes = nominal_attributes; self.splitter = splitter; self.binary_split = binary_split
        self.min_branch_fraction = min_branch_fraction; self.max_share_to_split = max_share_to_split
        self.max_size = max_size; self.memory_estimate_period = memory_estimate_period
        self.stop_mem_management = stop_mem_management; self.remove_poor_attrs = remove_poor_attrs
        self.merit_preprune = merit_preprune

    @property
    def _mutable_attributes(self): return {"max_features", "lambda_value", "grace_period", "delta", "tau"}
    @property
    def _multiclass(self): return True

    # This predict_proba_one uses the standard _metrics for weighting
    def predict_proba_one(self, x: dict) -> dict[base.typing.ClfTarget, float]:
        y_pred: typing.Counter = collections.Counter()
        if len(self) == 0:
            self._init_ensemble(sorted(x.keys()))
            return y_pred  # type: ignore

        for i, model in enumerate(self):
            y_proba_temp = model.predict_proba_one(x)
            metric_value = self._metrics[i].get() # Uses standard metric
            if not self.disable_weighted_vote and metric_value > 0.0:
                y_proba_temp = {k: val * metric_value for k, val in y_proba_temp.items()}
            y_pred.update(y_proba_temp)

        total = sum(y_pred.values())
        if total > 0: return {label: proba / total for label, proba in y_pred.items()}
        return y_pred # type: ignore

    def _new_base_model(self):
        return BaseTreeClassifier(
            max_features=self.max_features, grace_period=self.grace_period,
            split_criterion=self.split_criterion, delta=self.delta, tau=self.tau,
            leaf_prediction=self.leaf_prediction, nb_threshold=self.nb_threshold,
            nominal_attributes=self.nominal_attributes, splitter=self.splitter, max_depth=self.max_depth,
            binary_split=self.binary_split, min_branch_fraction=self.min_branch_fraction,
            max_share_to_split=self.max_share_to_split, max_size=self.max_size,
            memory_estimate_period=self.memory_estimate_period, stop_mem_management=self.stop_mem_management,
            remove_poor_attrs=self.remove_poor_attrs, merit_preprune=self.merit_preprune, rng=self._rng,
        )

    def _drift_detector_input(self, tree_id: int, y_true: base.typing.ClfTarget, y_pred: base.typing.ClfTarget) -> int | float:
        # Input for drift detector based on error (0 for correct, 1 for incorrect)
        return int(not y_true == y_pred)


# ================================================================
# NEW Class with Dynamic Weighting based on 0.9 / 1.1 rule
# ================================================================

class ARFClassifierDynamicWeights(ARFClassifier):
    """
    Adaptive Random Forest Classifier with dynamic tree weighting based on
    recent performance (0.9/1.1 multiplicative update rule).

    This class modifies the weighting scheme of the standard ARFClassifier.
    Instead of using the global accuracy (`metrics.Accuracy`) of each tree for
    weighted voting, it maintains a separate performance score for each tree.
    This score is updated multiplicatively based on whether the tree's prediction
    for the current instance was correct (multiplied by 0.9) or incorrect
    (multiplied by 1.1). The voting weights are then derived inversely from
    these scores (lower score means higher weight) and normalized.

    Note: The `disable_weighted_vote` parameter from the parent class is
    effectively ignored here, as this class always uses its dynamic weighting.
    The `metric` parameter is still used for the drift/warning detectors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize dynamic weighting specific attributes
        # Performance score (lower is better, starts neutral)
        self._dynamic_perf_scores: list[float] = []
        # Actual normalized weights derived from scores
        self._dynamic_weights: list[float] = []

        # If the ensemble is already initialized (e.g., loaded from state),
        # initialize weights too. Otherwise, _init_ensemble will handle it.
        if len(self) > 0:
            self._init_dynamic_weights()


    def _init_dynamic_weights(self):
        """Initializes or resets the dynamic weight tracking lists."""
        # Start with a neutral performance score (like capymoa example)
        self._dynamic_perf_scores = [1.0] * self.n_models
        # Start with equal weights
        equal_weight = 1.0 / self.n_models if self.n_models > 0 else 1.0
        self._dynamic_weights = [equal_weight] * self.n_models

    def _init_ensemble(self, features: list):
        """Overrides base initialization to include dynamic weights."""
        super()._init_ensemble(features) # Call parent to create models, etc.
        self._init_dynamic_weights()     # Initialize our specific lists

    def _update_dynamic_weights(self):
        """Calculates normalized weights from performance scores."""
        if not self._dynamic_perf_scores: # Should not happen if initialized
             return

        # Calculate weights inversely proportional to score (1 / (1 + score))
        raw_weights = [1.0 / (1.0 + score) for score in self._dynamic_perf_scores]

        total_weight = sum(raw_weights)

        if total_weight > 0:
            # Normalize weights
            self._dynamic_weights = [w / total_weight for w in raw_weights]
        else:
            # Handle edge case (e.g., all scores are infinite? Assign equal weight)
             equal_weight = 1.0 / self.n_models if self.n_models > 0 else 1.0
             self._dynamic_weights = [equal_weight] * self.n_models


    def learn_one(self, x: dict, y: base.typing.Target, **kwargs):
        """
        Learns from an instance, updates dynamic performance scores,
        and recalculates weights.
        """
        if len(self) == 0:
            self._init_ensemble(sorted(x.keys()))

        tree_predictions = [None] * self.n_models # Store individual predictions

        # --- Stage 1: Predictions and Metric Updates ---
        for i, model in enumerate(self):
            # Get prediction from THIS specific tree
            y_pred_tree = model.predict_one(x)
            tree_predictions[i] = y_pred_tree # Store for later use

            # Update standard performance evaluator (for drift/warning detection)
            self._metrics[i].update(
                y_true=y,
                y_pred=(
                    model.predict_proba_one(x) # Use proba if metric requires it
                    if isinstance(self.metric, metrics.base.ClassificationMetric)
                    and not self.metric.requires_labels
                    else y_pred_tree # Use the direct prediction otherwise
                ),
            )

            # Update dynamic performance score based on this tree's prediction
            if y_pred_tree == y:
                 self._dynamic_perf_scores[i] *= 0.9  # Correct prediction -> lower score
            else:
                 self._dynamic_perf_scores[i] *= 1.1  # Incorrect prediction -> higher score

            # Optional: Add a floor/ceiling to prevent scores from going to 0 or infinity
            # self._dynamic_perf_scores[i] = max(0.01, min(self._dynamic_perf_scores[i], 100.0))


        # --- Stage 2: Drift/Warning Detection and Tree Management ---
        for i, model in enumerate(self):
            y_pred_tree = tree_predictions[i] # Use the stored prediction
            drift_input = None

            # Check for warnings using the individual prediction
            if not self._warning_detection_disabled:
                drift_input = self._drift_detector_input(i, y, y_pred_tree)
                self._warning_detectors[i].update(drift_input)
                if self._warning_detectors[i].drift_detected:
                    if self._background is not None:
                        self._background[i] = self._new_base_model()
                    self._warning_detectors[i] = self.warning_detector.clone()
                    self._warning_tracker[i] += 1 # Log warning

            # Check for drifts using the individual prediction
            if not self._drift_detection_disabled:
                drift_input = (
                    drift_input
                    if drift_input is not None
                    else self._drift_detector_input(i, y, y_pred_tree)
                )
                self._drift_detectors[i].update(drift_input)
                if self._drift_detectors[i].drift_detected:
                    # Reset or replace tree
                    if not self._warning_detection_disabled and self._background is not None and self._background[i] is not None:
                        self.data[i] = self._background[i]
                        self._background[i] = None
                    else:
                        self.data[i] = self._new_base_model()

                    # Reset detectors and standard metrics
                    if not self._warning_detection_disabled:
                       self._warning_detectors[i] = self.warning_detector.clone()
                    self._drift_detectors[i] = self.drift_detector.clone()
                    self._metrics[i] = self.metric.clone()

                    # *** Crucially: Reset dynamic weight scores for the replaced tree ***
                    self._dynamic_perf_scores[i] = 1.0 # Reset score to neutral

                    self._drift_tracker[i] += 1 # Log drift


        # --- Stage 3: Recalculate Weights and Train Models ---
        # Update the normalized weights based on the scores updated in Stage 1 & reset in Stage 2
        self._update_dynamic_weights()

        # Train models (original logic using Poisson sampling)
        for i, model in enumerate(self):
             k = poisson(rate=self.lambda_value, rng=self._rng)
             if k > 0:
                 # Train background model if it exists
                 if not self._warning_detection_disabled and self._background is not None and self._background[i] is not None:
                    self._background[i].learn_one(x=x, y=y, w=k) # type: ignore
                 # Train the main model
                 model.learn_one(x=x, y=y, w=k)


    # Override predict_proba_one to use the new dynamic weights
    def predict_proba_one(self, x: dict) -> dict[base.typing.ClfTarget, float]:
        """
        Predicts probabilities using the dynamically calculated weights.
        """
        y_pred_proba: typing.Counter = collections.Counter()

        if len(self) == 0:
            # Handle case where the model hasn't learned anything yet
            # Need to initialize here IF predict is called before learn_one
            # (Though typically learn_one is called first)
            self._init_ensemble(sorted(x.keys()))
            # Return empty or default prediction
            # Depending on expected behavior for unseen classes
            return {} # Return empty probabilities

        if not self._dynamic_weights: # Ensure weights are initialized
             # This might happen if predict is called before learn_one on a loaded model
             self._init_dynamic_weights()


        for i, model in enumerate(self):
            y_proba_tree = model.predict_proba_one(x)
            # Use the dynamic weight for this tree
            weight = self._dynamic_weights[i]

            if weight > 0.0: # Only consider trees with positive weight
                # Apply weight to the probabilities from this tree
                for label, proba in y_proba_tree.items():
                    y_pred_proba[label] += proba * weight

        # The weights are already normalized, so the sum of y_pred_proba values
        # should ideally be close to 1.0. Re-normalizing ensures it sums to exactly 1.
        total_proba = sum(y_pred_proba.values())
        if total_proba > 0:
            return {label: proba / total_proba for label, proba in y_pred_proba.items()}
        else:
             # Handle case where all weights were zero or no probabilities were produced
             # Return uniform distribution over known classes? Or empty?
             # Returning empty seems safer if no tree contributed.
             return {} # type: ignore


    # predict_one uses predict_proba_one, so it automatically benefits
    # from the dynamic weighting without needing an override.
