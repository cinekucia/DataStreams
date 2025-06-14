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
from river.tree.nodes.arf_htr_nodes import RandomLeafAdaptive, RandomLeafMean, RandomLeafModel
from river.tree.splitter import Splitter
from river.utils.random import poisson


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
        disable_weighted_vote,
        seed,
    ):
        super().__init__([])
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
            self._warning_detectors = [self.warning_detector.clone() for _ in range(self.n_models)]
            self._warning_detection_disabled = False

        self._drift_detectors: list[base.DriftDetector]
        self._drift_detection_disabled = True
        if not isinstance(self.drift_detector, NoDrift):
            self._drift_detectors = [self.drift_detector.clone() for _ in range(self.n_models)]
            self._drift_detection_disabled = False

        self._background: list[BaseTreeClassifier | BaseTreeRegressor | None] = (
            None if self._warning_detection_disabled else [None] * self.n_models
        )

        self._metrics = [self.metric.clone() for _ in range(self.n_models)]

        self._warning_tracker: dict = (
            collections.defaultdict(int) if not self._warning_detection_disabled else None
        )
        self._drift_tracker: dict = (
            collections.defaultdict(int) if not self._drift_detection_disabled else None
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
        if self._warning_detection_disabled:
            return 0

        if tree_id is None:
            return sum(self._warning_tracker.values())

        return self._warning_tracker[tree_id]

    def n_drifts_detected(self, tree_id: int | None = None) -> int:
        if self._drift_detection_disabled:
            return 0

        if tree_id is None:
            return sum(self._drift_tracker.values())

        return self._drift_tracker[tree_id]

    def learn_one(self, x: dict, y: base.typing.Target, **kwargs):
        if len(self) == 0:
            self._init_ensemble(sorted(x.keys()))

        for i, model in enumerate(self):
            y_pred = model.predict_one(x)

            self._metrics[i].update(
                y_true=y,
                y_pred=(
                    model.predict_proba_one(x)
                    if isinstance(self.metric, metrics.base.ClassificationMetric)
                    and not self.metric.requires_labels
                    else y_pred
                ),
            )

            k = poisson(rate=self.lambda_value, rng=self._rng)
            if k > 0:
                if not self._warning_detection_disabled and self._background[i] is not None:
                    self._background[i].learn_one(x=x, y=y, w=k)

                model.learn_one(x=x, y=y, w=k)

                drift_input = None
                if not self._warning_detection_disabled:
                    drift_input = self._drift_detector_input(i, y, y_pred)
                    self._warning_detectors[i].update(drift_input)

                    if self._warning_detectors[i].drift_detected:
                        self._background[i] = self._new_base_model()
                        self._warning_detectors[i] = self.warning_detector.clone()
                        self._warning_tracker[i] += 1

                if not self._drift_detection_disabled:
                    drift_input = (
                        drift_input
                        if drift_input is not None
                        else self._drift_detector_input(i, y, y_pred)
                    )
                    self._drift_detectors[i].update(drift_input)

                    if self._drift_detectors[i].drift_detected:
                        if not self._warning_detection_disabled and self._background[i] is not None:
                            self.data[i] = self._background[i]
                            self._background[i] = None
                            self._warning_detectors[i] = self.warning_detector.clone()
                            self._drift_detectors[i] = self.drift_detector.clone()
                            self._metrics[i] = self.metric.clone()
                        else:
                            self.data[i] = self._new_base_model()
                            self._drift_detectors[i] = self.drift_detector.clone()
                            self._metrics[i] = self.metric.clone()

                        self._drift_tracker[i] += 1

    def _init_ensemble(self, features: list):
        self._set_max_features(len(features))
        self.data = [self._new_base_model() for _ in range(self.n_models)]

    def _set_max_features(self, n_features):
        if self.max_features == "sqrt":
            self.max_features = round(math.sqrt(n_features))
        elif self.max_features == "log2":
            self.max_features = round(math.log2(n_features))
        elif isinstance(self.max_features, int):
            pass
        elif isinstance(self.max_features, float):
            self.max_features = int(self.max_features * n_features)
        elif self.max_features is None:
            self.max_features = n_features
        else:
            raise AttributeError(
                f"Invalid max_features: {self.max_features}.\n"
                f"Valid options are: int [2, M], float (0., 1.],"
                f" {self._FEATURES_SQRT}, {self._FEATURES_LOG2}"
            )
        if self.max_features < 0:
            self.max_features += n_features
        if self.max_features <= 0:
            self.max_features = 1
        if self.max_features > n_features:
            self.max_features = n_features


class BaseTreeClassifier(HoeffdingTreeClassifier):
    def __init__(
        self,
        max_features: int = 2,
        grace_period: int = 200,
        max_depth: int | None = None,
        split_criterion: str = "info_gain",
        delta: float = 1e-7,
        tau: float = 0.05,
        leaf_prediction: str = "nba",
        nb_threshold: int = 0,
        nominal_attributes: list | None = None,
        splitter: Splitter | None = None,
        binary_split: bool = False,
        min_branch_fraction: float = 0.01,
        max_share_to_split: float = 0.99,
        max_size: float = 100.0,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
        rng: random.Random | None = None,
    ):
        super().__init__(
            grace_period=grace_period,
            max_depth=max_depth,
            split_criterion=split_criterion,
            delta=delta,
            tau=tau,
            leaf_prediction=leaf_prediction,
            nb_threshold=nb_threshold,
            nominal_attributes=nominal_attributes,
            splitter=splitter,
            binary_split=binary_split,
            min_branch_fraction=min_branch_fraction,
            max_share_to_split=max_share_to_split,
            max_size=max_size,
            memory_estimate_period=memory_estimate_period,
            stop_mem_management=stop_mem_management,
            remove_poor_attrs=remove_poor_attrs,
            merit_preprune=merit_preprune,
        )

        self.max_features = max_features
        self.rng = rng

    def _new_leaf(self, initial_stats=None, parent=None):
        if initial_stats is None:
            initial_stats = {}

        if parent is None:
            depth = 0
        else:
            depth = parent.depth + 1

        if self._leaf_prediction == self._MAJORITY_CLASS:
            return RandomLeafMajorityClass(
                initial_stats,
                depth,
                self.splitter,
                self.max_features,
                self.rng,
            )
        elif self._leaf_prediction == self._NAIVE_BAYES:
            return RandomLeafNaiveBayes(
                initial_stats,
                depth,
                self.splitter,
                self.max_features,
                self.rng,
            )
        else:
            return RandomLeafNaiveBayesAdaptive(
                initial_stats,
                depth,
                self.splitter,
                self.max_features,
                self.rng,
            )


class SelectiveARFClassifier(BaseForest, base.Classifier):
    def __init__(
        self,
        n_models: int = 10,
        max_features: bool | str | int = "sqrt",
        lambda_value: int = 6,
        metric: metrics.base.MultiClassMetric | None = None,
        disable_weighted_vote=False,
        drift_detector: base.DriftDetector | None = None,
        warning_detector: base.DriftDetector | None = None,
        prediction_confidence_threshold: float = 0.0,
        grace_period: int = 50,
        max_depth: int | None = None,
        split_criterion: str = "info_gain",
        delta: float = 0.01,
        tau: float = 0.05,
        leaf_prediction: str = "nba",
        nb_threshold: int = 0,
        nominal_attributes: list | None = None,
        splitter: Splitter | None = None,
        binary_split: bool = False,
        min_branch_fraction: float = 0.01,
        max_share_to_split: float = 0.99,
        max_size: float = 100.0,
        memory_estimate_period: int = 2_000_000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
        seed: int | None = None,
    ):
        super().__init__(
            n_models=n_models,
            max_features=max_features,
            lambda_value=lambda_value,
            metric=metric or metrics.Accuracy(),
            disable_weighted_vote=disable_weighted_vote,
            drift_detector=drift_detector or ADWIN(delta=0.001),
            warning_detector=warning_detector or ADWIN(delta=0.01),
            seed=seed,
        )

        self.last_prediction_voter_count = 0
        
        self.prediction_confidence_threshold = prediction_confidence_threshold
        self.grace_period = grace_period
        self.max_depth = max_depth
        self.split_criterion = split_criterion
        self.delta = delta
        self.tau = tau
        self.leaf_prediction = leaf_prediction
        self.nb_threshold = nb_threshold
        self.nominal_attributes = nominal_attributes
        self.splitter = splitter
        self.binary_split = binary_split
        self.min_branch_fraction = min_branch_fraction
        self.max_share_to_split = max_share_to_split
        self.max_size = max_size
        self.memory_estimate_period = memory_estimate_period
        self.stop_mem_management = stop_mem_management
        self.remove_poor_attrs = remove_poor_attrs
        self.merit_preprune = merit_preprune

    @property
    def _mutable_attributes(self):
        return {
            "max_features",
            "lambda_value",
            "grace_period",
            "delta",
            "tau",
            "prediction_confidence_threshold"
        }

    @property
    def _multiclass(self):
        return True

    def predict_proba_one(
        self, x: dict, **kwargs: typing.Any
    ) -> dict[base.typing.ClfTarget, float]:
        y_pred: typing.Counter = collections.Counter()
        
        if len(self) == 0:
            self._init_ensemble(sorted(x.keys()))
            self.last_prediction_voter_count = 0
            return y_pred

        # <<< MODIFIED LOGIC START
        # Variables to track the best fallback option
        best_fallback_proba = None
        best_fallback_metric_value = 0.0
        max_confidence_seen = -1.0
        
        # A temporary list to store votes from confident models
        confident_votes = []
        
        for i, model in enumerate(self):
            y_proba_temp = model.predict_proba_one(x)
            if not y_proba_temp:
                continue

            confidence_margin = 1.0  # Default for single-class predictions
            if len(y_proba_temp) > 1:
                sorted_probs = sorted(y_proba_temp.values(), reverse=True)
                confidence_margin = sorted_probs[0] - sorted_probs[1]

            # Track the most confident model as a fallback
            if confidence_margin > max_confidence_seen:
                max_confidence_seen = confidence_margin
                best_fallback_proba = y_proba_temp
                best_fallback_metric_value = self._metrics[i].get()

            # If confident enough, add its vote to a temporary list
            if confidence_margin >= self.prediction_confidence_threshold:
                metric_value = self._metrics[i].get()
                confident_votes.append((y_proba_temp, metric_value))
        
        # If no model was confident enough, use the single best fallback
        if not confident_votes and best_fallback_proba is not None:
            # The fallback's vote is also weighted by its own performance
            metric_value = best_fallback_metric_value
            if not self.disable_weighted_vote and metric_value > 0.0:
                weighted_proba = {k: val * metric_value for k, val in best_fallback_proba.items()}
                y_pred.update(weighted_proba)
            else:
                y_pred.update(best_fallback_proba)
            self.last_prediction_voter_count = 1
        else:
            # Otherwise, aggregate all confident votes
            for proba, metric_value in confident_votes:
                if not self.disable_weighted_vote and metric_value > 0.0:
                    weighted_proba = {k: val * metric_value for k, val in proba.items()}
                    y_pred.update(weighted_proba)
                else:
                    y_pred.update(proba)
            self.last_prediction_voter_count = len(confident_votes)

        total = sum(y_pred.values())
        if total > 0:
            return {label: proba / total for label, proba in y_pred.items()}
        
        # This part should now be unreachable if the ensemble is not empty, but as a safeguard:
        self.last_prediction_voter_count = 0
        return {}


    def _new_base_model(self):
        return BaseTreeClassifier(
            max_features=self.max_features,
            grace_period=self.grace_period,
            split_criterion=self.split_criterion,
            delta=self.delta,
            tau=self.tau,
            leaf_prediction=self.leaf_prediction,
            nb_threshold=self.nb_threshold,
            nominal_attributes=self.nominal_attributes,
            splitter=self.splitter,
            max_depth=self.max_depth,
            binary_split=self.binary_split,
            min_branch_fraction=self.min_branch_fraction,
            max_share_to_split=self.max_share_to_split,
            max_size=self.max_size,
            memory_estimate_period=self.memory_estimate_period,
            stop_mem_management=self.stop_mem_management,
            remove_poor_attrs=self.remove_poor_attrs,
            merit_preprune=self.merit_preprune,
            rng=self._rng,
        )

    def _drift_detector_input(
        self, tree_id: int, y_true: base.typing.ClfTarget, y_pred: base.typing.ClfTarget
    ) -> int | float:
        return int(not y_true == y_pred)