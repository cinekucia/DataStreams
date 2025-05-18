# Import necessary components
from __future__ import annotations
import abc
import collections
import itertools
import math
import random
import typing
from river import base, metrics, stats
from river.drift import ADWIN, NoDrift
from river.tree import HoeffdingTreeClassifier # Assuming RandomSubspaceHT is similar in spirit
from river.metrics.base import ClassificationMetric, Metric, RegressionMetric
from river.utils.random import poisson
from river.utils.rolling import Rolling
from river.metrics import Accuracy as AccMetric # For evaluators

# ================================================================
# BASE SRP CLASSES (Assumed to be available and correct)
# ================================================================
# ... (Paste the BaseSRPEnsemble, BaseSRPEstimator, random_subspace,
#      SRPClassifier, BaseSRPClassifier definitions from previous correct versions)
# For brevity, I am omitting the re-paste of these base classes.
# Please assume they are the same as in the previous correct response.

# --- PASTE PREVIOUS BaseSRPEnsemble, BaseSRPEstimator, random_subspace ---
# --- PASTE PREVIOUS SRPClassifier, BaseSRPClassifier ---
# Make sure these are pasted here from the previous response
class BaseSRPEnsemble(base.Wrapper, base.Ensemble):
    _TRAIN_RANDOM_SUBSPACES = "subspaces"
    _TRAIN_RESAMPLING = "resampling"
    _TRAIN_RANDOM_PATCHES = "patches"
    _FEATURES_SQRT = "sqrt"
    _FEATURES_SQRT_INV = "rmsqrt"
    _VALID_TRAINING_METHODS = {_TRAIN_RANDOM_PATCHES, _TRAIN_RESAMPLING, _TRAIN_RANDOM_SUBSPACES}

    def __init__(
        self, model: base.Estimator | None = None, n_models: int = 100,
        subspace_size: int | float | str = 0.6, training_method: str = "patches",
        lam: float = 6.0, drift_detector: base.DriftDetector | None = None,
        warning_detector: base.DriftDetector | None = None, disable_detector: str = "off",
        disable_weighted_vote: bool = False, seed: int | None = None, metric: Metric | None = None,
    ):
        super().__init__([]) # type: ignore
        self.model = model # This is the base model prototype
        self.n_models = n_models; self.subspace_size = subspace_size
        self.training_method = training_method; self.lam = lam
        self.drift_detector = drift_detector # Prototype for base learner's drift detector
        self.warning_detector = warning_detector # Prototype for base learner's warning detector
        self.disable_weighted_vote = disable_weighted_vote
        self.disable_detector = disable_detector
        self.metric = metric # Prototype for base learner's metric
        self.seed = seed
        self._rng = random.Random(self.seed)
        self._n_samples_seen = 0
        self._subspaces: list = []
        self._base_learner_class: typing.Any = None

    @property
    def _min_number_of_models(self): return 0
    @property
    def _wrapped_model(self): return self.model
    @classmethod
    def _unit_test_params(cls): yield {"n_models": 3, "seed": 42}
    def _unit_test_skips(self): return {"check_shuffle_features_no_impact", "check_emerging_features", "check_disappearing_features"}

    def learn_one(self, x: dict, y: base.typing.Target, **kwargs): # This is the learn_one of the *standard* SRPClassifier
        self._n_samples_seen += 1
        if not self: self._init_ensemble(list(x.keys()))

        for model_wrapper in self:
            y_pred = model_wrapper.predict_one(x)
            if y_pred is not None:
                model_wrapper.metric.update(y_true=y, y_pred=y_pred)

            if self.training_method == self._TRAIN_RANDOM_SUBSPACES: k = 1
            else:
                k = poisson(rate=self.lam, rng=self._rng)
                if k == 0: continue
            # kwargs here are passed down to the base model's learn_one (e.g. HoeffdingTree)
            model_wrapper.learn_one(x=x, y=y, w=k, n_samples_seen=self._n_samples_seen, **kwargs)


    def _generate_subspaces(self, features: list):
        n_features = len(features)
        self._subspaces = [None] * self.n_models
        if self.training_method != self._TRAIN_RESAMPLING:
            k: int | float
            if isinstance(self.subspace_size, float) and 0.0 < self.subspace_size <= 1:
                k = self.subspace_size; percent = (1.0 + k) / 1.0 if k < 0 else k; k = round(n_features * percent)
                if k < 2: k = round(n_features * percent) + 1
            elif isinstance(self.subspace_size, int) and self.subspace_size >= 2:
                k = self.subspace_size
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
                else:
                    self._subspaces = [random_subspace(all_features=features, k=k, rng=self._rng) for _ in range(self.n_models)]
            else: self.training_method = self._TRAIN_RESAMPLING

    def _init_ensemble(self, features: list):
        self._generate_subspaces(features=features)
        subspace_indexes = list(range(self.n_models))
        if self.training_method == self._TRAIN_RANDOM_PATCHES or self.training_method == self._TRAIN_RANDOM_SUBSPACES:
            self._rng.shuffle(subspace_indexes)
        self.data = []
        for i in range(self.n_models):
            subspace = self._subspaces[subspace_indexes[i]] if self._subspaces and subspace_indexes[i] < len(self._subspaces) else None
            # self.model, self.metric, self.drift_detector, self.warning_detector are prototypes
            self.append(
                self._base_learner_class( # type: ignore
                    idx_original=i, model_prototype=self.model, metric_prototype=self.metric,
                    created_on=self._n_samples_seen,
                    drift_detector_prototype=self.drift_detector,
                    warning_detector_prototype=self.warning_detector,
                    is_background_learner=False, rng=self._rng, features=subspace,
                )
            )
    def reset(self):
        self.data = []; self._n_samples_seen = 0; self._rng = random.Random(self.seed)


class BaseSRPEstimator: # This class will be used by SRP Rose River
    def __init__(
        self, idx_original: int, model_prototype: base.Estimator,
        metric_prototype: Metric, # This metric_prototype is for ADWIN input usually
        created_on: int,
        drift_detector_prototype: base.DriftDetector | None,
        warning_detector_prototype: base.DriftDetector | None,
        is_background_learner, rng: random.Random, features=None,
        # ROSE specific: each learner has its own windowed evaluator for Kappa/Accuracy
        evaluator_window_size: int = 500 # Default, matches ROSE's windowSizeOption
    ):
        self.idx_original = idx_original; self.created_on = created_on
        self._model_prototype = model_prototype
        self._metric_prototype = metric_prototype # For ADWIN
        self._drift_detector_prototype = drift_detector_prototype
        self._warning_detector_prototype = warning_detector_prototype

        self.model = self._model_prototype.clone()
        self.metric = self._metric_prototype.clone() # For ADWIN
        self.features = features

        self.disable_drift_detector = self._drift_detector_prototype is None
        self.drift_detector = self._drift_detector_prototype.clone() if self._drift_detector_prototype else None
        self.disable_background_learner = self._warning_detector_prototype is None
        self.warning_detector = self._warning_detector_prototype.clone() if self._warning_detector_prototype else None
        
        # ROSE: Each learner has its own performance evaluator (Kappa, Accuracy)
        # Using river's Rolling metrics
        self.rose_evaluator_accuracy = Rolling(AccMetric(), window_size=evaluator_window_size)


        self.is_background_learner = is_background_learner
        self.n_drifts_detected = 0; self.n_warnings_detected = 0; self.rng = rng
        self._background_learner: typing.Any = None
        self.evaluator_window_size = evaluator_window_size # Store for cloning

    def _trigger_warning(self, all_features, n_samples_seen: int):
        subspace = None
        if self.features is not None:
            k_original_subspace = len(self.features)
            subspace = random_subspace(all_features=all_features, k=k_original_subspace, rng=self.rng)
        self._background_learner = self.__class__( # type: ignore
            idx_original=self.idx_original, model_prototype=self._model_prototype,
            metric_prototype=self._metric_prototype, created_on=n_samples_seen,
            drift_detector_prototype=self._drift_detector_prototype,
            warning_detector_prototype=self._warning_detector_prototype,
            is_background_learner=True, rng=self.rng, features=subspace,
            evaluator_window_size=self.evaluator_window_size # Pass down window size
        )
        if self.warning_detector: self.warning_detector = self._warning_detector_prototype.clone()

    def reset(self, all_features: list, n_samples_seen: int):
        if not self.disable_background_learner and self._background_learner is not None:
            self.model = self._background_learner.model
            self.metric = self._background_learner.metric # For ADWIN
            self.drift_detector = self._background_learner.drift_detector
            self.warning_detector = self._background_learner.warning_detector
            self.created_on = self._background_learner.created_on; self.features = self._background_learner.features
            self.n_drifts_detected = self._background_learner.n_drifts_detected
            self.n_warnings_detected = self._background_learner.n_warnings_detected
            # Carry over ROSE evaluators
            self.rose_evaluator_accuracy = self._background_learner.rose_evaluator_accuracy
            self._background_learner = None
        else:
            new_subspace = None
            if self.features is not None:
                k_original_subspace = len(self.features)
                new_subspace = random_subspace(all_features=all_features, k=k_original_subspace, rng=self.rng)
            self.model = self._model_prototype.clone(); self.metric = self._metric_prototype.clone()
            self.created_on = n_samples_seen
            self.drift_detector = self._drift_detector_prototype.clone() if self._drift_detector_prototype else None
            self.warning_detector = self._warning_detector_prototype.clone() if self._warning_detector_prototype else None
            self.features = new_subspace
            self.n_drifts_detected = 0; self.n_warnings_detected = 0
            # Reset ROSE evaluators
            self.rose_evaluator_accuracy = Rolling(AccMetric(), window_size=self.evaluator_window_size)


def random_subspace(all_features: list, k: int, rng: random.Random):
    corrected_k = min(len(all_features), k)
    return rng.sample(all_features, k=corrected_k)

class SRPClassifier(BaseSRPEnsemble, base.Classifier): # Standard SRP
    def __init__(
        self, model: base.Estimator | None = None, n_models: int = 10,
        subspace_size: int | float | str = 0.6, training_method: str = "patches",
        lam: int = 6, drift_detector: base.DriftDetector | None = None,
        warning_detector: base.DriftDetector | None = None, disable_detector: str = "off",
        disable_weighted_vote: bool = False, seed: int | None = None,
        metric: ClassificationMetric | None = None,
        # Add evaluator_window_size for base learners if they need it (SRP Rose River will)
        evaluator_window_size: int = 500
    ):
        drift_detector_proto = drift_detector; warning_detector_proto = warning_detector
        metric_proto = metric
        if model is None: model = HoeffdingTreeClassifier(grace_period=50, delta=0.01)
        if drift_detector_proto is None and disable_detector != "drift":
            drift_detector_proto = ADWIN(delta=1e-5)
        if warning_detector_proto is None and disable_detector == "off":
            warning_detector_proto = ADWIN(delta=1e-4)
        if disable_detector == "drift":
            drift_detector_proto = None; warning_detector_proto = None
        elif disable_detector == "warning": warning_detector_proto = None
        elif disable_detector != "off": raise AttributeError(f"{disable_detector} is not a valid...")
        if metric_proto is None: metric_proto = AccMetric() # Using river's Accuracy
        super().__init__(
            model=model, n_models=n_models, subspace_size=subspace_size, training_method=training_method,
            lam=lam, drift_detector=drift_detector_proto, warning_detector=warning_detector_proto,
            disable_detector=disable_detector, disable_weighted_vote=disable_weighted_vote,
            seed=seed, metric=metric_proto,
        )
        self._base_learner_class = BaseSRPClassifier # Standard SRP uses BaseSRPClassifier
        self.evaluator_window_size = evaluator_window_size # Store for base learner init

    # _init_ensemble in BaseSRPEnsemble needs to be aware of evaluator_window_size if passed
    # For SRP Rose River, we'll override _init_ensemble to pass it correctly.
    # For standard SRP, BaseSRPEstimator won't use it unless modified.

    def predict_proba_one(self, x, **kwargs):
        y_pred = collections.Counter()
        if not self.models: self._init_ensemble(features=list(x.keys())); return y_pred
        for model_wrapper in self.models:
            y_proba_temp = model_wrapper.predict_proba_one(x, **kwargs)
            metric_value = model_wrapper.metric.get() # This is the ADWIN metric usually
            if not self.disable_weighted_vote and metric_value > 0.0:
                y_proba_temp = {k: val * metric_value for k, val in y_proba_temp.items()}
            y_pred.update(y_proba_temp)
        total = sum(y_pred.values())
        if total > 0: return {label: proba / total for label, proba in y_pred.items()}
        return y_pred

    @property
    def _multiclass(self): return True

class BaseSRPClassifier(BaseSRPEstimator): # Standard Base Learner for SRP
    def __init__(
        self, idx_original: int, model_prototype: base.Classifier,
        metric_prototype: ClassificationMetric, created_on: int,
        drift_detector_prototype: base.DriftDetector | None,
        warning_detector_prototype: base.DriftDetector | None,
        is_background_learner, rng: random.Random, features=None,
        evaluator_window_size: int = 500 # Added to signature, though not used by standard BaseSRP
    ):
        super().__init__(
            idx_original=idx_original, model_prototype=model_prototype,
            metric_prototype=metric_prototype, created_on=created_on,
            drift_detector_prototype=drift_detector_prototype,
            warning_detector_prototype=warning_detector_prototype,
            is_background_learner=is_background_learner, rng=rng, features=features,
            evaluator_window_size=evaluator_window_size # Pass to super
        )

    def learn_one(self, x: dict, y: base.typing.ClfTarget, *, w: int, n_samples_seen: int, **kwargs):
        all_features_for_reset = kwargs.pop('all_features', list(x.keys()))
        x_subset = {k: x[k] for k in self.features if k in x} if self.features is not None else x
        for _ in range(int(w)): self.model.learn_one(x=x_subset, y=y, **kwargs)
        if self._background_learner:
            self._background_learner.learn_one(x=x, y=y, w=w, n_samples_seen=n_samples_seen,
                                                all_features=all_features_for_reset, **kwargs)
        if not self.disable_drift_detector and not self.is_background_learner:
            prediction_for_drift = self.model.predict_one(x_subset)
            if prediction_for_drift is None: return
            correctly_classifies = prediction_for_drift == y
            # Update self.metric (for ADWIN)
            self.metric.update(y_true=y, y_pred=prediction_for_drift)

            if not self.disable_background_learner and self.warning_detector:
                self.warning_detector.update(int(not correctly_classifies))
                if self.warning_detector.drift_detected:
                    self.n_warnings_detected += 1
                    self._trigger_warning(all_features=all_features_for_reset, n_samples_seen=n_samples_seen)
            if self.drift_detector:
                self.drift_detector.update(int(not correctly_classifies))
                if self.drift_detector.drift_detected:
                    self.n_drifts_detected += 1
                    self.reset(all_features=all_features_for_reset, n_samples_seen=n_samples_seen)

    def predict_proba_one(self, x, **kwargs):
        x_subset = {k: x[k] for k in self.features if k in x} if self.features is not None else x
        return self.model.predict_proba_one(x_subset, **kwargs)

    def predict_one(self, x: dict, **kwargs) -> base.typing.ClfTarget:
        y_pred_proba = self.predict_proba_one(x, **kwargs)
        if y_pred_proba: return max(y_pred_proba, key=y_pred_proba.get)
        return None

# =======================================================================
# SRP ROSE River Classifier
# =======================================================================

class InstanceWithTimestamp: # Helper for ROSE instance window
    def __init__(self, instance_data: dict, target: base.typing.Target, timestamp: int):
        self.x = instance_data
        self.y = target
        self.timestamp = timestamp

    def __lt__(self, other): # For sorting by timestamp
        return self.timestamp < other.timestamp

class SRPRoseRiver(SRPClassifier): # Inherit from SRPClassifier for structure
    """
    SRP-based ensemble with ROSE-inspired mechanisms for imbalanced streams.
    Features:
    - Adaptive Poisson sampling rate based on class distribution.
    - Background ensemble triggered by warnings, trained on a window of recent instances.
    - Ensemble member replacement based on Kappa * Accuracy from windowed evaluators.
    - Voting based on Kappa * Accuracy.
    """
    def __init__(
        self,
        model: base.Estimator | None = None, # Base model prototype
        n_models: int = 10,                 # ensembleSize
        subspace_size: int | float | str = 0.6, # For SRP's feature subsampling
        training_method: str = "patches",   # SRP training: patches, resampling, subspaces
        lam: float = 6.0,                   # lambdaOption (base lambda for Poisson)
        drift_detector: base.DriftDetector | None = None,    # For base learners
        warning_detector: base.DriftDetector | None = None,  # For base learners
        disable_detector: str = "off",      # For base learners
        seed: int | None = None,
        metric: ClassificationMetric | None = None, # For base learner's ADWIN
        # ROSE-specific parameters
        theta_class_decay: float = 0.99,    # theta (for class size decay)
        rose_window_size: int = 500,        # windowSizeOption (for instance window & evaluators)
        # featureSpaceOption from ROSE is somewhat covered by SRP's subspace_size.
        # We could add more modes to subspace_size if exact ROSE behavior is needed.
        # percentageFeaturesMean is related to featureSpaceOption=2, not implemented here.
    ):
        super().__init__(
            model=model if model is not None else HoeffdingTreeClassifier(grace_period=50, delta=0.01),
            n_models=n_models,
            subspace_size=subspace_size,
            training_method=training_method,
            lam=lam, # lam is the base lambda for ROSE
            drift_detector=drift_detector, # Passed to base learners
            warning_detector=warning_detector, # Passed to base learners
            disable_detector=disable_detector,
            disable_weighted_vote=True, # ROSE uses its own Kappa*Acc weighting
            seed=seed,
            metric=metric if metric is not None else AccMetric(), # For base learner ADWINs
            evaluator_window_size=rose_window_size # Ensure base learners get this
        )
        # Override base learner class for SRP Rose
        self._base_learner_class = BaseSRPRoseLearner

        self.theta_class_decay = theta_class_decay
        self.rose_window_size = rose_window_size # Also evaluation period for background
        self.base_lambda = lam # Store the original lambda for ROSE logic

        # ROSE internal state
        self._class_sizes_decayed: typing.Optional[np.ndarray] = None
        self._n_classes: int = 0
        self._ensemble_warning_active: bool = False
        self._first_warning_timestamp: int = 0
        self._background_learners: list[BaseSRPRoseLearner] = []

        # Instance window per class (deque stores InstanceWithTimestamp)
        self._instance_window_per_class: typing.Optional[list[collections.deque]] = None

    def _init_rose_specific_state(self, instance: dict, y: base.typing.Target):
        """Initializes ROSE-specific attributes based on the first instance."""
        if hasattr(self.model, 'n_classes') and self.model.n_classes is not None:
            self._n_classes = self.model.n_classes
        elif isinstance(y, int): # Assuming class labels are integers starting from 0
             # This is a common heuristic in River if classes are not predefined
             # It might need adjustment if class labels are not contiguous or 0-indexed
            self._n_classes = y + 1 # A guess, needs to be updated if higher class appears
        else:
            # Attempt to infer from a seen set of classes if available, e.g. model.classes
            # For now, raise error or make a very rough guess
            raise ValueError("Cannot determine number of classes. Pre-train model or ensure y provides class info.")

        if self._n_classes <= 0:
            raise ValueError(f"Number of classes ({self._n_classes}) must be positive.")

        self._class_sizes_decayed = np.array([1.0 / self._n_classes] * self._n_classes)
        
        # Max size for each class's deque in the window
        # ROSE: windowSizeOption.getValue() / instance.numClasses()
        # This means per-class window size is smaller if many classes
        per_class_window_limit = max(1, self.rose_window_size // self._n_classes)
        self._instance_window_per_class = [
            collections.deque(maxlen=per_class_window_limit) for _ in range(self._n_classes)
        ]
        self._ensemble_warning_active = False # ROSE starts with warningDetected = true, then false after init. Here, false initially.

    def _init_ensemble(self, features: list): # Override to pass evaluator_window_size
        self._generate_subspaces(features=features)
        subspace_indexes = list(range(self.n_models))
        if self.training_method == self._TRAIN_RANDOM_PATCHES or self.training_method == self._TRAIN_RANDOM_SUBSPACES:
            self._rng.shuffle(subspace_indexes)
        self.data = [] # Primary ensemble
        for i in range(self.n_models):
            subspace = self._subspaces[subspace_indexes[i]] if self._subspaces and subspace_indexes[i] < len(self._subspaces) else None
            self.append(
                self._base_learner_class( # type: ignore
                    idx_original=i, model_prototype=self.model, metric_prototype=self.metric,
                    created_on=self._n_samples_seen,
                    drift_detector_prototype=self.drift_detector,
                    warning_detector_prototype=self.warning_detector,
                    is_background_learner=False, rng=self._rng, features=subspace,
                    evaluator_window_size=self.rose_window_size # Pass this
                )
            )

    def _update_class_sizes(self, y_true_idx: int):
        if self._class_sizes_decayed is None: return # Not initialized
        for i in range(self._n_classes):
            is_current_class = 1.0 if i == y_true_idx else 0.0
            self._class_sizes_decayed[i] = (
                self.theta_class_decay * self._class_sizes_decayed[i]
                + (1.0 - self.theta_class_decay) * is_current_class
            )
        # Normalize to prevent sum from drifting far from 1 (optional, but good practice)
        current_sum = np.sum(self._class_sizes_decayed)
        if current_sum > 0:
            self._class_sizes_decayed /= current_sum


    def _get_adaptive_lambda(self, y_true_idx: int) -> float:
        if self._class_sizes_decayed is None or not self._class_sizes_decayed.any():
            return self.base_lambda # Fallback if not initialized or all zeros

        max_class_size = np.max(self._class_sizes_decayed)
        current_class_size = self._class_sizes_decayed[y_true_idx]

        if current_class_size <= 1e-9: # Avoid division by zero or log of zero
            # If class size is effectively zero, treat as very rare, give high lambda
            return self.base_lambda + self.base_lambda * math.log(max_class_size / 1e-9 if max_class_size > 1e-9 else 1.0)

        # lambda = lambdaOption.getValue() + lambdaOption.getValue() * Math.log(classSize[Utils.maxIndex(classSize)] / classSize[(int) instance.classValue()]);
        adaptive_lambda = self.base_lambda + self.base_lambda * math.log(max_class_size / current_class_size)
        return max(1.0, adaptive_lambda) # Ensure lambda is at least 1


    def learn_one(self, x: dict, y: base.typing.Target, **kwargs):
        self._n_samples_seen += 1
        y_true_idx = int(y) # Assuming y is class index

        if self._class_sizes_decayed is None: # First instance
            self._init_rose_specific_state(x, y)
            # initEnsemble from ROSE is called here
            if not self.models: # Check if primary ensemble needs init
                 self._init_ensemble(list(x.keys()))


        # Update class sizes and store instance in window
        self._update_class_sizes(y_true_idx)
        if self._instance_window_per_class is not None:
            self._instance_window_per_class[y_true_idx].append(
                InstanceWithTimestamp(x, y, self._n_samples_seen)
            )

        adaptive_lambda = self._get_adaptive_lambda(y_true_idx)

        # 1. Train primary ensemble and check for warnings
        new_warning_this_step = False
        for model_wrapper in self.models: # These are BaseSRPRoseLearner
            y_pred_tree = model_wrapper.predict_one(x) # For evaluation
            if y_pred_tree is not None:
                # Update ROSE evaluators (Kappa, Accuracy)
                model_wrapper.rose_evaluator_accuracy.update(y_true=y_true_idx, y_pred=int(y_pred_tree))
                # Update metric for ADWIN
                model_wrapper.metric.update(y_true=y_true_idx, y_pred=int(y_pred_tree))


            k = poisson(rate=adaptive_lambda, rng=self._rng)
            if k > 0:
                model_wrapper.learn_one(x=x, y=y, w=k, n_samples_seen=self._n_samples_seen, **kwargs)
            
            # Check if this model_wrapper triggered a warning (its internal warning_detector.drift_detected)
            # This is implicitly handled inside model_wrapper.learn_one which sets its own warning flag
            if not self._ensemble_warning_active and model_wrapper.warning_detector and model_wrapper.warning_detector.drift_detected:
                new_warning_this_step = True

        if new_warning_this_step and not self._ensemble_warning_active :
            self._ensemble_warning_active = True
            self._first_warning_timestamp = self._n_samples_seen
            # print(f"Step {self._n_samples_seen}: Ensemble warning DETECTED. Initializing background learners.")

            # Initialize background ensemble
            self._background_learners = []
            for i in range(self.n_models):
                # Create new base learners for background
                # Subspace generation is tricky here, ROSE re-calculates it.
                # For simplicity, let's reuse subspaces or generate new ones if needed.
                subspace = self._subspaces[i % len(self._subspaces)] if self._subspaces else None

                bg_learner = BaseSRPRoseLearner( # type: ignore
                    idx_original=i + self.n_models, # Give distinct IDs
                    model_prototype=self.model, metric_prototype=self.metric,
                    created_on=self._n_samples_seen,
                    drift_detector_prototype=self.drift_detector, # Use same detector types
                    warning_detector_prototype=self.warning_detector, # Background learners don't typically have warnings
                    is_background_learner=True, # Mark as background
                    rng=random.Random(self._rng.randint(0, 2**32 -1)), # New RNG seed
                    features=subspace,
                    evaluator_window_size=self.rose_window_size
                )
                self._background_learners.append(bg_learner)

            # Train background learners on the instance window
            # ROSE sorts all instances from all class windows by timestamp
            all_window_instances_sorted: list[InstanceWithTimestamp] = []
            if self._instance_window_per_class:
                for class_deque in self._instance_window_per_class:
                    all_window_instances_sorted.extend(list(class_deque))
            all_window_instances_sorted.sort() # Sort by timestamp

            for inst_ts in all_window_instances_sorted:
                # Use base_lambda for training from window, not adaptive_lambda
                # as class distribution in window might be different
                k_bg = poisson(rate=self.base_lambda, rng=self._rng)
                if k_bg > 0:
                    for bg_model in self._background_learners:
                        bg_model.learn_one(x=inst_ts.x, y=inst_ts.y, w=k_bg,
                                           n_samples_seen=self._n_samples_seen, # or inst_ts.timestamp?
                                           **kwargs) # Pass original kwargs


        # 2. If warning is active, train background learners on current instance
        if self._ensemble_warning_active and self._background_learners:
            for bg_model in self._background_learners:
                y_pred_bg_tree = bg_model.predict_one(x)
                if y_pred_bg_tree is not None:
                    bg_model.rose_evaluator_accuracy.update(y_true=y_true_idx, y_pred=int(y_pred_bg_tree))
                    # ADWIN metric update for bg_model internal drift
                    bg_model.metric.update(y_true=y_true_idx, y_pred=int(y_pred_bg_tree))


                k_bg_current = poisson(rate=adaptive_lambda, rng=self._rng) # Use adaptive for current instance
                if k_bg_current > 0:
                    bg_model.learn_one(x=x, y=y, w=k_bg_current, n_samples_seen=self._n_samples_seen, **kwargs)

            # Check if evaluation period for background learners is over
            if self._n_samples_seen - self._first_warning_timestamp >= self.rose_window_size:
                # print(f"Step {self._n_samples_seen}: Warning period OVER. Evaluating and selecting models.")
                all_candidates = list(self.models) + list(self._background_learners)
                
                candidate_scores = []
                for learner in all_candidates:
                    accuracy = learner.rose_evaluator_accuracy.get()
                    # ROSE: kappas.get(j) * accuracies.get(j)
                    score = (kappa if kappa is not None else 0.0) * \
                            (accuracy if accuracy is not None else 0.0)
                    candidate_scores.append(score)

                # Select top N models
                sorted_indices = sorted(range(len(all_candidates)),
                                        key=lambda k: candidate_scores[k], reverse=True)
                
                new_primary_ensemble = []
                for i in range(self.n_models):
                    if i < len(sorted_indices):
                        selected_learner = all_candidates[sorted_indices[i]]
                        # Mark as not background if it was from background
                        selected_learner.is_background_learner = False
                        # Reset its warning detector as it's now a primary learner
                        if selected_learner._warning_detector_prototype:
                             selected_learner.warning_detector = selected_learner._warning_detector_prototype.clone()
                        else:
                             selected_learner.warning_detector = None # Or default ADWIN if one was used
                        selected_learner.disable_background_learner = selected_learner.warning_detector is None

                        new_primary_ensemble.append(selected_learner)
                    else: # Should not happen if enough candidates
                        break 
                
                self.data = new_primary_ensemble # self.models is a property for self.data
                self._background_learners = []
                self._ensemble_warning_active = False
                # print(f"Step {self._n_samples_seen}: New primary ensemble selected.")


    def predict_proba_one(self, x: dict, **kwargs) -> dict[base.typing.ClfTarget, float]:
        if not self.models: # Ensemble not initialized
            if self._class_sizes_decayed is None and 'y' in kwargs: # Attempt init if y is passed for predict
                 self._init_rose_specific_state(x, kwargs['y']) # This is a bit of a hack for predict
            if not self.models: self._init_ensemble(list(x.keys())) # Initialize with features
            if not self.models: return {} # Still no models

        combined_vote = collections.Counter()
        combined_vote_unweighted = collections.Counter()
        active_voters = 0

        for model_wrapper in self.models: # These are BaseSRPRoseLearner
            y_proba_tree = model_wrapper.predict_proba_one(x, **kwargs)
            if not y_proba_tree: continue # No prediction from this tree

            active_voters += 1
            # Unweighted sum for fallback
            for label, proba in y_proba_tree.items():
                combined_vote_unweighted[label] += proba

            # Weighted sum using Kappa * Accuracy
            accuracy = model_wrapper.rose_evaluator_accuracy.get()

            if accuracy is not None: # ROSE condition
                weight =  accuracy
                for label, proba in y_proba_tree.items():
                    combined_vote[label] += proba * weight
        
        if sum(combined_vote.values()) > 1e-9 : # If weighted voting produced something meaningful
            total_weighted_vote = sum(combined_vote.values())
            return {label: val / total_weighted_vote for label, val in combined_vote.items()}
        elif sum(combined_vote_unweighted.values()) > 1e-9: # Fallback to unweighted
            total_unweighted_vote = sum(combined_vote_unweighted.values())
            return {label: val / total_unweighted_vote for label, val in combined_vote_unweighted.items()}
        else: # No tree could vote or all weights were zero
            return {}


    def reset(self):
        super().reset()
        self._class_sizes_decayed = None
        self._n_classes = 0
        self._ensemble_warning_active = False
        self._first_warning_timestamp = 0
        self._background_learners = []
        self._instance_window_per_class = None

class BaseSRPRoseLearner(BaseSRPClassifier): # Inherit from BaseSRPClassifier for its learn_one
    """Base learner for SRPRoseRiver, includes windowed evaluators."""
    def __init__(self, *args, **kwargs):
        # Pop evaluator_window_size before passing to super if it's already handled
        # by BaseSRPEstimator's __init__
        eval_window_size = kwargs.pop('evaluator_window_size', 500)
        super().__init__(*args, **kwargs, evaluator_window_size=eval_window_size) # Pass it up

        # These are initialized by BaseSRPEstimator's __init__ now
        # self.rose_evaluator_kappa = Rolling(KappaM(), window_size=eval_window_size)
        # self.rose_evaluator_accuracy = Rolling(AccMetric(), window_size=eval_window_size)

    # learn_one and predict_proba_one are inherited from BaseSRPClassifier
    # The reset method in BaseSRPEstimator handles resetting these evaluators
