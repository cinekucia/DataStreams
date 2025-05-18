# ================================================================
#  SRPRoseRiver with built-in label encoder
# ================================================================
#  Complete, self-contained implementation:
#    • All original classes are present.
#    • The only functional change is that SRPRoseRiver now accepts
#      arbitrary (string, bool, …) class labels by mapping them
#      internally to integer indices and decoding predictions back.
# ================================================================

from __future__ import annotations

# ---------- standard / third-party imports ----------------------
import abc
import collections
import itertools
import math
import random
import typing
from typing import Hashable
import numpy as np

from river import base, metrics, stats
from river.drift import ADWIN, NoDrift
from river.tree import HoeffdingTreeClassifier
from river.metrics.base import ClassificationMetric, Metric, RegressionMetric
from river.metrics import CohenKappa
from river.metrics import Accuracy as AccMetric
from river.utils.random import poisson
from river.utils.rolling import Rolling


# ================================================================
# 1.  BASE SRP CLASSES (unchanged)
# ================================================================

class BaseSRPEnsemble(base.Wrapper, base.Ensemble):
    _TRAIN_RANDOM_SUBSPACES = "subspaces"
    _TRAIN_RESAMPLING       = "resampling"
    _TRAIN_RANDOM_PATCHES   = "patches"
    _FEATURES_SQRT          = "sqrt"
    _FEATURES_SQRT_INV      = "rmsqrt"
    _VALID_TRAINING_METHODS = {
        _TRAIN_RANDOM_PATCHES, _TRAIN_RESAMPLING, _TRAIN_RANDOM_SUBSPACES
    }

    def __init__(
        self,
        model: base.Estimator | None = None,
        n_models: int = 100,
        subspace_size: int | float | str = 0.6,
        training_method: str = "patches",
        lam: float = 6.0,
        drift_detector: base.DriftDetector | None = None,
        warning_detector: base.DriftDetector | None = None,
        disable_detector: str = "off",
        disable_weighted_vote: bool = False,
        seed: int | None = None,
        metric: Metric | None = None,
    ):
        super().__init__([])
        self.model               = model
        self.n_models            = n_models
        self.subspace_size       = subspace_size
        self.training_method     = training_method
        self.lam                 = lam
        self.drift_detector      = drift_detector
        self.warning_detector    = warning_detector
        self.disable_weighted_vote = disable_weighted_vote
        self.disable_detector    = disable_detector
        self.metric              = metric
        self.seed                = seed

        self._rng              = random.Random(self.seed)
        self._n_samples_seen   = 0
        self._subspaces: list  = []
        self._base_learner_class: typing.Any = None

    # ----- required hooks for river.base -------------------------
    @property
    def _min_number_of_models(self): return 0
    @property
    def _wrapped_model(self):       return self.model
    @classmethod
    def _unit_test_params(cls):     yield {"n_models": 3, "seed": 42}
    def _unit_test_skips(self):
        return {
            "check_shuffle_features_no_impact",
            "check_emerging_features",
            "check_disappearing_features",
        }

    # -------------------------------------------------------------
    def learn_one(self, x: dict, y: base.typing.Target, **kwargs):
        self._n_samples_seen += 1
        if not self:
            self._init_ensemble(list(x.keys()))

        for model_wrapper in self:
            # update metric before training (pre-quential)
            y_pred = model_wrapper.predict_one(x)
            if y_pred is not None and model_wrapper.metric is not None:
                model_wrapper.metric.update(y_true=y, y_pred=y_pred)

            # subsampling or Poisson replication
            if self.training_method == self._TRAIN_RANDOM_SUBSPACES:
                k = 1
            else:
                k = poisson(self.lam, self._rng)
                if k == 0:
                    continue

            model_wrapper.learn_one(
                x=x,
                y=y,
                w=k,
                n_samples_seen=self._n_samples_seen,
                **kwargs,
            )

    # -------------------------------------------------------------
    def _generate_subspaces(self, features: list[str]):
        """Pick feature subspaces for each base learner."""
        n_features = len(features)
        self._subspaces = [None] * self.n_models

        if self.training_method == self._TRAIN_RESAMPLING:
            return

        # calculate k (subspace size)
        if isinstance(self.subspace_size, float) and 0.0 < self.subspace_size <= 1:
            percent = self.subspace_size if self.subspace_size > 0 else 1.0 + self.subspace_size
            k = round(n_features * percent)
            if k < 2 and n_features > 1: # Ensure k is at least 2 if possible
                k = 2
            elif k < 1:
                k = 1
        elif isinstance(self.subspace_size, int) and self.subspace_size >= 1: # Allow k=1
            k = self.subspace_size
        elif self.subspace_size == self._FEATURES_SQRT:
            k = round(math.sqrt(n_features))
            if k < 1: k = 1 # Ensure k is at least 1
            if n_features > 1 and k < 2 : k=2 # Prefer 2 if possible
        elif self.subspace_size == self._FEATURES_SQRT_INV:
            k = n_features - round(math.sqrt(n_features))
            if k < 1: k = 1 # Ensure k is at least 1
        else:
            raise ValueError(f"Invalid subspace_size: {self.subspace_size}")

        if k <= 0: # Should be caught above, but defensive
            k = 1
        if k > n_features:
            k = n_features


        if k < n_features: # Only create subspaces if k is less than total features
            if n_features <= 20 or k < 2: # Use combinations for small N or k
                # if k == 1 and n_features > 1: # Prefer k=2 if only 1 feature is selected but more are available
                #     k = 2
                self._subspaces = []
                for i, comb in enumerate(itertools.cycle(itertools.combinations(features, k))):
                    if i == self.n_models:
                        break
                    self._subspaces.append(list(comb))
            else: # Use random sampling for larger N and k
                self._subspaces = [
                    random_subspace(features, k, self._rng)
                    for _ in range(self.n_models)
                ]
        else: # k equals n_features, so all models use all features (effectively resampling)
            self.training_method = self._TRAIN_RESAMPLING
            self._subspaces = [features] * self.n_models # All models use all features

    # -------------------------------------------------------------
    def _init_ensemble(self, features: list[str]):
        """Create base learners; overridden in SRPRoseRiver."""
        self._generate_subspaces(features)
        idxs = list(range(self.n_models))
        if self.training_method in {self._TRAIN_RANDOM_PATCHES,
                                    self._TRAIN_RANDOM_SUBSPACES}:
            self._rng.shuffle(idxs)

        self.data = []
        eval_window = getattr(self, "evaluator_window_size", 500)
        for i in range(self.n_models):
            # Ensure subsp is correctly assigned even if self._subspaces is shorter than n_models (e.g. due to combinations)
            subsp_idx = idxs[i] % len(self._subspaces) if self._subspaces else None
            subsp = self._subspaces[subsp_idx] if subsp_idx is not None and self._subspaces else features

            self.append(
                self._base_learner_class(
                    idx_original=i,
                    model_prototype=self.model,
                    metric_prototype=self.metric,
                    created_on=self._n_samples_seen,
                    drift_detector_prototype=self.drift_detector,
                    warning_detector_prototype=self.warning_detector,
                    is_background_learner=False,
                    rng=self._rng,
                    features=subsp,
                    evaluator_window_size=eval_window,
                )
            )

    # -------------------------------------------------------------
    def reset(self):
        self.data = []
        self._n_samples_seen = 0
        self._rng = random.Random(self.seed)


# ----------------------------------------------------------------
class BaseSRPEstimator:
    """
    Wrapper around the actual base model that holds detectors, metrics,
    background learner, etc.
    """

    def __init__(
        self,
        idx_original: int,
        model_prototype: base.Estimator,
        metric_prototype: Metric,
        created_on: int,
        drift_detector_prototype: base.DriftDetector | None,
        warning_detector_prototype: base.DriftDetector | None,
        is_background_learner: bool,
        rng: random.Random,
        features: list[str] | None = None,
        evaluator_window_size: int = 500,
    ):
        # identifiers
        self.idx_original     = idx_original
        self.created_on       = created_on
        self.is_background_learner = is_background_learner

        # prototypes
        self._model_prototype   = model_prototype
        self._metric_prototype  = metric_prototype
        self._drift_detector_prototype = drift_detector_prototype
        self._warning_detector_prototype = warning_detector_prototype

        # concrete instances
        self.model  = self._model_prototype.clone()
        self.metric = self._metric_prototype.clone()

        self.disable_drift_detector     = self._drift_detector_prototype is None
        self.drift_detector             = (
            self._drift_detector_prototype.clone() if self._drift_detector_prototype else None
        )
        self.disable_background_learner = self._warning_detector_prototype is None
        self.warning_detector           = (
            self._warning_detector_prototype.clone() if self._warning_detector_prototype else None
        )

        self.features = features
        self.rng = rng
        self.evaluator_window_size = evaluator_window_size

        # rolling evaluators for ROSE
        self.rose_evaluator_kappa    = Rolling(CohenKappa(), window_size=evaluator_window_size)
        self.rose_evaluator_accuracy = Rolling(AccMetric(),   window_size=evaluator_window_size)

        # stats
        self.n_drifts_detected   = 0
        self.n_warnings_detected = 0
        self._background_learner: typing.Any = None

    # -------------------------------------------------------------
    def _trigger_warning(self, all_features: list[str], n_samples_seen: int):
        """Instantiate background learner on warning."""
        subspace = None
        if self.features is not None:
            k = len(self.features)
            subspace = random_subspace(all_features, k, self.rng)

        self._background_learner = self.__class__(
            idx_original=self.idx_original,
            model_prototype=self._model_prototype,
            metric_prototype=self._metric_prototype,
            created_on=n_samples_seen,
            drift_detector_prototype=self._drift_detector_prototype,
            warning_detector_prototype=self._warning_detector_prototype,
            is_background_learner=True,
            rng=self.rng, # Should be a new RNG for background learner for independence?
                          # Or pass self.rng.randint(0, 2**32-1) as seed for new Random(seed)
            features=subspace,
            evaluator_window_size=self.evaluator_window_size,
        )
        if self.warning_detector:
            self.warning_detector = self._warning_detector_prototype.clone()

    # -------------------------------------------------------------
    def reset(self, all_features: list[str], n_samples_seen: int):
        """
        Replace current model by background learner (if any) or
        re-initialise from prototype.
        """
        if not self.disable_background_learner and self._background_learner:
            bl = self._background_learner
            (self.model, self.metric, self.drift_detector,
             self.warning_detector, self.created_on, self.features) = (
                bl.model,
                bl.metric,
                bl.drift_detector,
                bl.warning_detector,
                bl.created_on,
                bl.features,
            )
            self.n_drifts_detected   = bl.n_drifts_detected
            self.n_warnings_detected = bl.n_warnings_detected
            self.rose_evaluator_kappa    = bl.rose_evaluator_kappa
            self.rose_evaluator_accuracy = bl.rose_evaluator_accuracy
            self._background_learner = None
        else:
            # full reset
            new_subspace = None
            if self.features is not None:
                k = len(self.features)
                new_subspace = random_subspace(all_features, k, self.rng)

            self.model  = self._model_prototype.clone()
            self.metric = self._metric_prototype.clone()
            self.created_on = n_samples_seen
            if self._drift_detector_prototype:
                self.drift_detector = self._drift_detector_prototype.clone()
            if self._warning_detector_prototype:
                self.warning_detector = self._warning_detector_prototype.clone()
            self.features = new_subspace
            self.n_drifts_detected   = 0
            self.n_warnings_detected = 0
            self.rose_evaluator_kappa    = Rolling(CohenKappa(), window_size=self.evaluator_window_size)
            self.rose_evaluator_accuracy = Rolling(AccMetric(),   window_size=self.evaluator_window_size)


# -------------------------------------------------------------
def random_subspace(all_features: list[str], k: int, rng: random.Random):
    """Utility: return *k* random distinct features."""
    k = min(len(all_features), k)
    if k <= 0 : return [] # Handle k=0 or negative k gracefully
    return rng.sample(all_features, k)


# ----------------------------------------------------------------
class SRPClassifier(BaseSRPEnsemble, base.Classifier):
    """
    Classic Stream-Random Patches ensemble (from river-extras).
    """

    def __init__(
        self,
        model: base.Estimator | None = None,
        n_models: int = 10,
        subspace_size: int | float | str = 0.6,
        training_method: str = "patches",
        lam: int = 6,
        drift_detector: base.DriftDetector | None = None,
        warning_detector: base.DriftDetector | None = None,
        disable_detector: str = "off",
        disable_weighted_vote: bool = False,
        seed: int | None = None,
        metric: ClassificationMetric | None = None,
        evaluator_window_size: int = 500, # Added this parameter
    ):
        # default prototypes
        if model is None:
            model = HoeffdingTreeClassifier(grace_period=50, delta=0.01)
        if drift_detector is None and disable_detector != "drift":
            drift_detector = ADWIN(delta=1e-5)
        if warning_detector is None and disable_detector == "off":
            warning_detector = ADWIN(delta=1e-4)
        if disable_detector == "drift":
            drift_detector = warning_detector = None
        elif disable_detector == "warning":
            warning_detector = None
        elif disable_detector not in {"off", "drift", "warning"}:
            raise AttributeError(f"{disable_detector} is not a valid option.")

        metric = metric or AccMetric()

        super().__init__(
            model=model,
            n_models=n_models,
            subspace_size=subspace_size,
            training_method=training_method,
            lam=lam,
            drift_detector=drift_detector,
            warning_detector=warning_detector,
            disable_detector=disable_detector,
            disable_weighted_vote=disable_weighted_vote,
            seed=seed,
            metric=metric,
        )
        self._base_learner_class = BaseSRPClassifier
        self.evaluator_window_size = evaluator_window_size # Store it

    # -------------------------------------------------------------
    def predict_proba_one(self, x, **kwargs):
        if not self.models:
            self._init_ensemble(list(x.keys()))
            return {}

        vote = collections.Counter()
        for mw in self.models:
            proba = mw.predict_proba_one(x, **kwargs)
            if not proba:
                continue

            weight = 1.0
            if not self.disable_weighted_vote:
                m_val = mw.metric.get()
                if m_val and m_val > 0:
                    weight = m_val

            for lbl, p in proba.items():
                vote[lbl] += p * weight

        tot = sum(vote.values())
        if tot > 0:
            return {lbl: p / tot for lbl, p in vote.items()}
        return {}

    # -------------------------------------------------------------
    @property
    def _multiclass(self):  # river API flag
        return True


# ----------------------------------------------------------------
class BaseSRPClassifier(BaseSRPEstimator):
    """
    Concrete base learner wrapper for classification.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # -------------------------------------------------------------
    def learn_one(
        self,
        x: dict,
        y: base.typing.ClfTarget,
        *,
        w: int,
        n_samples_seen: int,
        **kwargs,
    ):
        all_feats_for_reset = kwargs.pop("all_features", list(x.keys()))
        # Ensure self.features is not None and not empty before creating x_sub
        # If self.features is None or empty, use all features from x
        x_sub = {k_feat: x[k_feat] for k_feat in self.features if k_feat in x} if (self.features and isinstance(self.features, list) and len(self.features) > 0) else x

        for _ in range(int(w)):
            self.model.learn_one(x_sub, y)

        # background learner update
        if self._background_learner:
            self._background_learner.learn_one(
                x=x, # Background learner gets all features by default design
                y=y,
                w=w,
                n_samples_seen=n_samples_seen,
                all_features=all_feats_for_reset, # For its own potential resets
            )

        # drift / warning detection
        if not self.disable_drift_detector and not self.is_background_learner:
            y_pred = self.model.predict_one(x_sub)
            if y_pred is None:
                return

            correct = y_pred == y
            if self.metric:
                self.metric.update(y_true=y, y_pred=y_pred)

            if not self.disable_background_learner and self.warning_detector:
                self.warning_detector.update(int(not correct))
                if self.warning_detector.drift_detected:
                    self.n_warnings_detected += 1
                    self._trigger_warning(all_feats_for_reset, n_samples_seen)

            if self.drift_detector:
                self.drift_detector.update(int(not correct))
                if self.drift_detector.drift_detected:
                    self.n_drifts_detected += 1
                    self.reset(all_feats_for_reset, n_samples_seen)

    # -------------------------------------------------------------
    def predict_proba_one(self, x, **kwargs):
        x_sub = {k_feat: x[k_feat] for k_feat in self.features if k_feat in x} if (self.features and isinstance(self.features, list) and len(self.features) > 0) else x
        return self.model.predict_proba_one(x_sub)

    # -------------------------------------------------------------
    def predict_one(self, x, **kwargs):
        proba = self.predict_proba_one(x, **kwargs)
        return max(proba, key=proba.get) if proba else None


# ================================================================
# 2.  SRP-ROSE River Classifier (MODIFIED)
# ================================================================

class InstanceWithTimestamp:
    """Helper to keep instances with their arrival order."""
    def __init__(self, x, y_idx: int, ts: int):
        self.x = x
        self.y = y_idx # Store the integer encoded y
        self.timestamp = ts
    def __lt__(self, other):  # needed for sorting
        return self.timestamp < other.timestamp


class BaseSRPRoseLearner(BaseSRPClassifier):
    """No change – inherits everything from BaseSRPClassifier."""
    pass


class SRPRoseRiver(SRPClassifier):
    """Stream-Random-Patches with ROSE warning handling."""

    # -------------------------------------------------------------
    def __init__(
        self,
        model: base.Estimator | None = None,
        n_models: int = 10,
        subspace_size: int | float | str = 0.6,
        training_method: str = "patches",
        lam: float = 6.0,
        drift_detector: base.DriftDetector | None = None,
        warning_detector: base.DriftDetector | None = None,
        disable_detector: str = "off",
        seed: int | None = None,
        metric: ClassificationMetric | None = None,
        theta_class_decay: float = 0.99,
        rose_window_size: int = 500,
    ):
        super().__init__(
            model=model or HoeffdingTreeClassifier(grace_period=50, delta=0.01),
            n_models=n_models,
            subspace_size=subspace_size,
            training_method=training_method,
            lam=lam, # Pass float lam here
            drift_detector=drift_detector,
            warning_detector=warning_detector,
            disable_detector=disable_detector,
            disable_weighted_vote=True, # ROSE uses its own weighting via kappa*accuracy
            seed=seed,
            metric=metric or AccMetric(),
            evaluator_window_size=rose_window_size, # Pass to SRPClassifier's init
        )

        # SRP-ROSE specific
        self._base_learner_class = BaseSRPRoseLearner
        self.theta_class_decay   = theta_class_decay
        self.rose_window_size    = rose_window_size
        self.base_lambda         = lam # Store the original lambda

        # dynamic state
        self._class_sizes_decayed: typing.Optional[np.ndarray] = None
        self._n_classes          = 0 # Number of unique *encoded* integer classes seen
        self._ensemble_warning_active = False
        self._first_warning_timestamp = 0
        self._background_learners: list[BaseSRPRoseLearner] = []
        self._instance_window_per_class: typing.Optional[list[collections.deque]] = None

        # NEW: label maps
        self._lbl2idx: dict[Hashable, int] = {} # Maps original label to int index
        self._idx2lbl: dict[int, Hashable] = {} # Maps int index back to original label

    # ------------------------------------------------------------------
    # label helpers (NEW)
    # ------------------------------------------------------------------
    def _encode_label(self, y: Hashable) -> int:
        """Encodes an original label to an integer index."""
        if y not in self._lbl2idx:
            # New label encountered, assign a new integer index
            idx = len(self._lbl2idx) # Next available index
            self._lbl2idx[y] = idx
            self._idx2lbl[idx] = y
            return idx
        return self._lbl2idx[y]

    def _decode_label(self, idx: int) -> Hashable:
        """Decodes an integer index back to the original label."""
        return self._idx2lbl.get(idx, idx) # Fallback to returning idx if not found (should not happen)

    # ------------------------------------------------------------------
    def _init_rose_specific_state(self, x: dict, first_y_idx: int):
        """Initialise ROSE per-class structures based on the first encoded label."""
        # Determine initial _n_classes
        # If base model is pre-trained and has class info, try to use it.
        # Otherwise, derive from first_y_idx.
        n_classes_from_model = 0
        if hasattr(self.model, "_n_classes_seen") and self.model._n_classes_seen is not None and self.model._n_classes_seen > 0:
            n_classes_from_model = self.model._n_classes_seen
        elif hasattr(self.model, "n_classes_") and self.model.n_classes_ is not None and self.model.n_classes_ > 0:
            n_classes_from_model = self.model.n_classes_
        
        # _n_classes should be at least first_y_idx + 1
        # and at least the number of labels already in _lbl2idx
        self._n_classes = max(first_y_idx + 1, n_classes_from_model, len(self._lbl2idx))

        if self._n_classes <= 0: # Should not happen if first_y_idx is valid
            raise ValueError(f"SRPRoseRiver: _n_classes determined as {self._n_classes}, must be positive.")

        self._class_sizes_decayed = np.full(self._n_classes, 1.0 / self._n_classes, dtype=float)
        
        per_cls_limit = max(1, self.rose_window_size // self._n_classes if self._n_classes > 0 else self.rose_window_size)
        self._instance_window_per_class = [
            collections.deque(maxlen=per_cls_limit) for _ in range(self._n_classes)
        ]
        self._ensemble_warning_active = False

    # ------------------------------------------------------------------
    # _init_ensemble is inherited from SRPClassifier, which uses self.evaluator_window_size
    # This is already set to self.rose_window_size in SRPRoseRiver's __init__.

    # ------------------------------------------------------------------
    def _update_class_sizes(self, y_idx: int):
        """Updates decayed class sizes. Assumes y_idx is a valid encoded integer index."""
        if self._class_sizes_decayed is None or y_idx >= self._n_classes or self._n_classes == 0:
            # This state should not be reached if _init_rose_specific_state and class expansion are correct
            return

        for i in range(self._n_classes):
            is_current_class = 1.0 if i == y_idx else 0.0
            self._class_sizes_decayed[i] = (
                self.theta_class_decay * self._class_sizes_decayed[i]
                + (1.0 - self.theta_class_decay) * is_current_class
            )
        
        current_sum = np.sum(self._class_sizes_decayed)
        if current_sum > 1e-9:
            self._class_sizes_decayed /= current_sum
        elif self._n_classes > 0 : # Avoid division by zero if all decayed to zero
             self._class_sizes_decayed.fill(1.0 / self._n_classes)


    # ------------------------------------------------------------------
    def _get_adaptive_lambda(self, y_idx: int) -> float:
        """Calculates adaptive lambda. Assumes y_idx is a valid encoded integer index."""
        if (
            self._class_sizes_decayed is None
            or y_idx >= self._n_classes  # y_idx should always be < _n_classes here
            or self._n_classes == 0
            or not self._class_sizes_decayed.any()
        ):
            return self.base_lambda

        max_sz = np.max(self._class_sizes_decayed)
        cur_sz = self._class_sizes_decayed[y_idx]

        if cur_sz <= 1e-9: # Current class size is effectively zero
            # Give a boost if its max_sz is also very small (e.g. early stages)
            # or significantly penalize if other classes are much larger
            log_term = math.log(max(max_sz, 1e-9) / 1e-9) # Ensure max_sz is not zero for log
            adaptive_lambda = self.base_lambda + self.base_lambda * log_term
        else:
            adaptive_lambda = self.base_lambda + self.base_lambda * math.log(max_sz / cur_sz)
        
        return max(1.0, min(adaptive_lambda, self.base_lambda * 5)) # Cap lambda

    # ------------------------------------------------------------------
    def learn_one(self, x: dict, y: base.typing.Target, **kwargs):
        self._n_samples_seen += 1

        # ---- encode original label to integer index -----------------
        y_idx = self._encode_label(y) # y_idx is now guaranteed to be an int

        # ---- first-time initialisation of ROSE specific state -------
        if self._class_sizes_decayed is None:
            self._init_rose_specific_state(x, y_idx) # Pass the first y_idx
            if not self.models: # If ensemble not initialized by super().learn_one
                self._init_ensemble(list(x.keys()))

        # ---- handle new unseen class (if y_idx expands _n_classes) --
        if y_idx >= self._n_classes:
            old_n_classes = self._n_classes
            self._n_classes = y_idx + 1 # New number of unique integer classes

            # Expand _class_sizes_decayed
            if self._class_sizes_decayed is not None:
                new_sizes = np.zeros(self._n_classes, dtype=float)
                new_sizes[:old_n_classes] = self._class_sizes_decayed
                # Initialize new class(es) - simple average or 1/_n_classes
                num_newly_added = self._n_classes - old_n_classes
                if num_newly_added > 0:
                    # A simple approach for new classes:
                    initial_size_for_new = (1.0 - np.sum(new_sizes[:old_n_classes])) / num_newly_added if np.sum(new_sizes[:old_n_classes]) < 1.0 and num_newly_added > 0 else (1.0 / self._n_classes)
                    initial_size_for_new = max(1e-9, initial_size_for_new) # Ensure non-zero
                    new_sizes[old_n_classes:] = initial_size_for_new

                self._class_sizes_decayed = new_sizes
                current_sum = np.sum(self._class_sizes_decayed)
                if current_sum > 1e-9:
                    self._class_sizes_decayed /= current_sum
                elif self._n_classes > 0:
                    self._class_sizes_decayed.fill(1.0 / self._n_classes)
            else: # Should not happen if _init_rose_specific_state was called
                self._class_sizes_decayed = np.full(self._n_classes, 1.0/self._n_classes, dtype=float)


            # Expand _instance_window_per_class
            if self._instance_window_per_class is not None:
                new_per_cls_limit = max(1, self.rose_window_size // self._n_classes if self._n_classes > 0 else self.rose_window_size)
                
                # Safely update maxlen for existing deques
                for i in range(min(old_n_classes, len(self._instance_window_per_class))): # Iterate only over existing valid indices
                    if self._instance_window_per_class[i].maxlen != new_per_cls_limit:
                        existing_items = list(self._instance_window_per_class[i])
                        self._instance_window_per_class[i] = collections.deque(existing_items, maxlen=new_per_cls_limit)
                
                # Add new deques for new classes
                for _ in range(old_n_classes, self._n_classes):
                    self._instance_window_per_class.append(
                        collections.deque(maxlen=new_per_cls_limit)
                    )
            else: # Should not happen
                per_cls_limit = max(1, self.rose_window_size // self._n_classes if self._n_classes > 0 else self.rose_window_size)
                self._instance_window_per_class = [collections.deque(maxlen=per_cls_limit) for _ in range(self._n_classes)]


        # ---- update per-class decay counters using y_idx ------------
        self._update_class_sizes(y_idx)

        # ---- store instance (with encoded y_idx) for BG training ---
        if self._instance_window_per_class and y_idx < len(self._instance_window_per_class):
            self._instance_window_per_class[y_idx].append(
                InstanceWithTimestamp(x, y_idx, self._n_samples_seen)
            )

        adaptive_lambda = self._get_adaptive_lambda(y_idx)
        new_warning_flag = False

        # ============================================================
        # Train / monitor primary ensemble (using encoded y_idx)
        # ============================================================
        for mw in self.models: # These are BaseSRPRoseLearner instances
            # Predict for metrics (expects original labels, but BaseSRPClassifier handles it if y is int)
            # The base learner's predict_one will return an encoded int label
            y_pred_tree_encoded = mw.predict_one(x) # This will be an int (encoded label)
            
            if y_pred_tree_encoded is not None:
                # Metrics within BaseSRPRoseLearner operate on encoded integer labels
                mw.rose_evaluator_kappa.update(y_true=y_idx, y_pred=y_pred_tree_encoded)
                mw.rose_evaluator_accuracy.update(y_true=y_idx, y_pred=y_pred_tree_encoded)
                if mw.metric: # The general metric also uses encoded labels
                    mw.metric.update(y_true=y_idx, y_pred=y_pred_tree_encoded)

            k = poisson(adaptive_lambda, self._rng)
            if k > 0: # Check if k is positive
                # Pass encoded y_idx to the base learner's learn_one
                mw.learn_one(
                    x=x,
                    y=y_idx, # Pass encoded integer label
                    w=k,
                    n_samples_seen=self._n_samples_seen,
                    all_features=list(x.keys()), # For potential resets within learn_one
                )

            if (
                not self._ensemble_warning_active
                and hasattr(mw, 'warning_detector') and mw.warning_detector # Ensure detector exists
                and mw.warning_detector.drift_detected
            ):
                new_warning_flag = True

        # ============================================================
        # On first warning: create background ensemble
        # ============================================================
        if new_warning_flag and not self._ensemble_warning_active:
            self._ensemble_warning_active = True
            self._first_warning_timestamp = self._n_samples_seen
            self._background_learners = []

            for i in range(self.n_models):
                # Subspace selection for background learner
                subsp_idx = i % len(self._subspaces) if self._subspaces else None
                subsp = self._subspaces[subsp_idx] if subsp_idx is not None and self._subspaces else list(x.keys())

                bg_rng_seed = self._rng.randint(0, 2**32 - 1) # Create a new seed for BG learner's RNG
                bg_rng = random.Random(bg_rng_seed)

                self._background_learners.append(
                    BaseSRPRoseLearner( # This is BaseSRPClassifier
                        idx_original=i + self.n_models, # Unique ID
                        model_prototype=self.model,
                        metric_prototype=self.metric,
                        created_on=self._n_samples_seen,
                        drift_detector_prototype=self.drift_detector, # BG learners also detect drift
                        warning_detector_prototype=None, # BG learners typically don't trigger further warnings
                        is_background_learner=True,
                        rng=bg_rng, # Pass the new RNG
                        features=subsp,
                        evaluator_window_size=self.rose_window_size,
                    )
                )

            # Pre-train BG learners on instances from the ROSE window (using encoded y_idx)
            window_instances: list[InstanceWithTimestamp] = []
            if self._instance_window_per_class:
                for dq_class_window in self._instance_window_per_class:
                    window_instances.extend(list(dq_class_window)) # list() to copy items
            window_instances.sort() # Sort by timestamp

            for inst_in_window in window_instances:
                k_bg = poisson(self.base_lambda, self._rng) # Use base_lambda for window training
                if not k_bg: continue
                for bg_model in self._background_learners:
                    # inst_in_window.y is already the encoded integer y_idx
                    bg_model.learn_one(
                        x=inst_in_window.x,
                        y=inst_in_window.y, # Pass encoded integer label
                        w=k_bg,
                        n_samples_seen=inst_in_window.timestamp,
                        all_features=list(inst_in_window.x.keys()),
                    )

        # ============================================================
        # If warning active: update BG learners and maybe replace
        # ============================================================
        if self._ensemble_warning_active and self._background_learners:
            for bg_model in self._background_learners:
                # Evaluate and update BG model with current instance (using encoded y_idx)
                y_pred_bg_encoded = bg_model.predict_one(x) # Returns encoded int label
                if y_pred_bg_encoded is not None:
                    bg_model.rose_evaluator_kappa.update(y_true=y_idx, y_pred=y_pred_bg_encoded)
                    bg_model.rose_evaluator_accuracy.update(y_true=y_idx, y_pred=y_pred_bg_encoded)
                    if bg_model.metric:
                        bg_model.metric.update(y_true=y_idx, y_pred=y_pred_bg_encoded)

                k_bg_now = poisson(adaptive_lambda, self._rng) # Use adaptive lambda for current instance
                if k_bg_now > 0:
                    bg_model.learn_one(
                        x=x,
                        y=y_idx, # Pass encoded integer label
                        w=k_bg_now,
                        n_samples_seen=self._n_samples_seen,
                        all_features=list(x.keys()),
                    )

            # Time to evaluate replacement?
            if self._n_samples_seen - self._first_warning_timestamp >= self.rose_window_size:
                all_candidates = list(self.models) + self._background_learners
                candidate_scores = []
                for learner_candidate in all_candidates:
                    kappa = learner_candidate.rose_evaluator_kappa.get() or 0.0
                    accuracy = learner_candidate.rose_evaluator_accuracy.get() or 0.0
                    score = kappa * accuracy # Could be other scoring, e.g., just kappa or accuracy
                    candidate_scores.append(score)
                
                # Get indices of candidates sorted by score in descending order
                sorted_indices_of_candidates = sorted(
                    range(len(all_candidates)),
                    key=lambda k_idx: candidate_scores[k_idx],
                    reverse=True
                )
                
                new_primary_ensemble: list[BaseSRPRoseLearner] = []
                for i in range(self.n_models): # Select top N_MODELS best candidates
                    if i < len(sorted_indices_of_candidates):
                        selected_learner_original_idx = sorted_indices_of_candidates[i]
                        selected_learner = all_candidates[selected_learner_original_idx]
                        
                        selected_learner.is_background_learner = False # Mark as primary
                        # Reset warning detector if it's now a primary learner and has a prototype
                        if selected_learner._warning_detector_prototype:
                            selected_learner.warning_detector = selected_learner._warning_detector_prototype.clone()
                        else: # Ensure it's None if no prototype
                            selected_learner.warning_detector = None
                        selected_learner.disable_background_learner = selected_learner.warning_detector is None
                        
                        new_primary_ensemble.append(selected_learner)
                    # else:
                        # If fewer than n_models good candidates, the ensemble might shrink.
                        # Or, could re-initialize new ones, but current SRP logic doesn't do that here.
                
                self.data = new_primary_ensemble # Replace the main ensemble
                self._background_learners = [] # Clear background learners
                self._ensemble_warning_active = False # Reset warning flag

    # ------------------------------------------------------------------
    def predict_proba_one(self, x: dict, **kwargs) -> dict[base.typing.ClfTarget, float]:
        if not self.models: # Ensemble not initialized
            # Attempt initialization if possible (e.g., if 'y' is passed for first time setup)
            if self._class_sizes_decayed is None and "y" in kwargs and kwargs["y"] is not None:
                # Need to encode label first to initialize rose state
                first_y_idx_for_init = self._encode_label(kwargs["y"])
                self._init_rose_specific_state(x, first_y_idx_for_init)
            if not self.models: # If still not initialized (e.g. _init_rose_specific_state didn't trigger _init_ensemble)
                self._init_ensemble(list(x.keys())) # Default ensemble initialization
            if not self.models: # If still truly no models, return empty
                return {}

        combined_weighted_vote = collections.Counter()
        combined_unweighted_vote = collections.Counter()
        num_weighted_voters = 0
        num_unweighted_voters = 0

        for model_wrapper in self.models: # These are BaseSRPRoseLearner instances
            # predict_proba_one from BaseSRPClassifier returns dict with ENCODED integer labels as keys
            proba_dict_encoded_labels = model_wrapper.predict_proba_one(x, **kwargs)
            
            if not proba_dict_encoded_labels: continue # Skip if model returns no prediction
            
            num_unweighted_voters +=1
            for encoded_label, proba_val in proba_dict_encoded_labels.items():
                combined_unweighted_vote[encoded_label] += proba_val

            # Weighting based on ROSE evaluators (kappa * accuracy)
            kappa = model_wrapper.rose_evaluator_kappa.get()
            accuracy = model_wrapper.rose_evaluator_accuracy.get()
            
            # Only use kappa*accuracy for weighting if kappa is positive (better than random)
            # and accuracy is also available.
            if kappa is not None and accuracy is not None and kappa > 0.0:
                weight = kappa * accuracy
                num_weighted_voters +=1
                for encoded_label, proba_val in proba_dict_encoded_labels.items():
                    combined_weighted_vote[encoded_label] += proba_val * weight
        
        final_votes_source_encoded = None
        if num_weighted_voters > 0 and sum(combined_weighted_vote.values()) > 1e-9 :
            final_votes_source_encoded = combined_weighted_vote
        elif num_unweighted_voters > 0 and sum(combined_unweighted_vote.values()) > 1e-9:
            final_votes_source_encoded = combined_unweighted_vote
        
        if final_votes_source_encoded:
            total_sum = sum(final_votes_source_encoded.values())
            if total_sum > 1e-9:
                # DECODE labels before returning
                decoded_probas = {
                    self._decode_label(enc_lbl): val / total_sum
                    for enc_lbl, val in final_votes_source_encoded.items()
                }
                return decoded_probas
        return {} # Return empty if no valid votes


    # ------------------------------------------------------------------
    def reset(self):
        super().reset() # Resets self.data (ensemble models), _n_samples_seen, _rng
        
        # Reset SRPRoseRiver specific state
        self._class_sizes_decayed = None
        self._n_classes = 0 # Will be re-determined on next learn_one
        self._ensemble_warning_active = False
        self._first_warning_timestamp = 0
        self._background_learners = []
        self._instance_window_per_class = None

        # Clear label maps
        self._lbl2idx.clear()
        self._idx2lbl.clear()