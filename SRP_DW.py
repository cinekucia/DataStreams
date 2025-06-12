from __future__ import annotations

import collections
import itertools
import math
import random
import typing # For type hints

import numpy as np # Not directly used in snippet, but common in river ecosystem

from river import base, metrics # metrics.Accuracy, MAE etc.
from river.drift import ADWIN # NoDrift not in SRP, but ADWIN is
from river.metrics.base import ClassificationMetric, Metric, RegressionMetric # Used in SRP
from river.tree import HoeffdingTreeClassifier, HoeffdingTreeRegressor # Used as default models in SRP
from river.utils.random import poisson

# ================================================================
# Original SRP Components (Copied for context - minor fixes for standalone execution if any)
# ================================================================
class BaseSRPEnsemble(base.Wrapper, base.Ensemble):
    _TRAIN_RANDOM_SUBSPACES = "subspaces"
    _TRAIN_RESAMPLING = "resampling"
    _TRAIN_RANDOM_PATCHES = "patches"
    _FEATURES_SQRT = "sqrt"
    _FEATURES_SQRT_INV = "rmsqrt"
    # Corrected _VALID_TRAINING_METHODS to include all options
    _VALID_TRAINING_METHODS = {
        _TRAIN_RANDOM_PATCHES,
        _TRAIN_RESAMPLING,
        _TRAIN_RANDOM_SUBSPACES, # Added
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
        super().__init__([])  # type: ignore
        self.model = model
        self.n_models = n_models
        self.subspace_size = subspace_size
        self.training_method = training_method
        if self.training_method not in self._VALID_TRAINING_METHODS:
             # Allow common string "subspaces" if user forgets the constant
             if self.training_method == "subspaces":
                 self.training_method = self._TRAIN_RANDOM_SUBSPACES
             elif self.training_method not in self._VALID_TRAINING_METHODS : # Check again after potential correction
                raise ValueError(
                    f"Invalid training_method: {self.training_method}. "
                    f"Valid options are: {self._VALID_TRAINING_METHODS}"
                )

        self.lam = lam
        self.drift_detector = drift_detector
        self.warning_detector = warning_detector
        self.disable_weighted_vote = disable_weighted_vote
        self.disable_detector = disable_detector
        self.metric = metric
        self.seed = seed
        self._rng = random.Random(self.seed)
        self._n_samples_seen = 0
        self._subspaces: list = []
        self._base_learner_class: typing.Type[BaseSRPEstimator] | None = None # Type hint for clarity


    @property
    def _min_number_of_models(self): return 0
    @property
    def _wrapped_model(self): return self.model
    @classmethod
    def _unit_test_params(cls): yield {"n_models": 3, "seed": 42}
    def _unit_test_skips(self): return {"check_shuffle_features_no_impact", "check_emerging_features", "check_disappearing_features"}

    def learn_one(self, x: dict, y: base.typing.Target, **kwargs):
        self._n_samples_seen += 1
        if not self.data: # Changed from `if not self:` for clarity and consistency with Ensemble
            self._init_ensemble(list(x.keys()))

        for model_member in self: # model_member is BaseSRPClassifier/Regressor instance
            y_pred = model_member.predict_one(x)
            if y_pred is not None and model_member.metric is not None:
                model_member.metric.update(y_true=y, y_pred=y_pred)

            k_train_weight = 0
            if self.training_method == self._TRAIN_RANDOM_SUBSPACES: k_train_weight = 1
            else: k_train_weight = poisson(rate=self.lam, rng=self._rng)
            if k_train_weight == 0: continue
            model_member.learn_one(x=x, y=y, w=k_train_weight, n_samples_seen=self._n_samples_seen, **kwargs)


    def _generate_subspaces(self, features: list):
        n_features = len(features)
        self._subspaces = [None] * self.n_models # type: ignore

        if self.training_method != self._TRAIN_RESAMPLING:
            k_num_features = 0 # Stores the target number of features for a subspace
            if isinstance(self.subspace_size, float) and 0.0 < self.subspace_size <= 1:
                percent = self.subspace_size # Simpler: if float, it's a percentage
                k_num_features = round(n_features * percent)
            elif isinstance(self.subspace_size, int) and self.subspace_size >= 1:
                k_num_features = self.subspace_size
            elif self.subspace_size == self._FEATURES_SQRT: k_num_features = round(math.sqrt(n_features)) + 1
            elif self.subspace_size == self._FEATURES_SQRT_INV: k_num_features = n_features - (round(math.sqrt(n_features)) + 1) # Corrected logic
            else: raise ValueError(f"Invalid subspace_size: {self.subspace_size}")

            # Ensure k is at least 1 and at most n_features
            k_num_features = max(1, min(k_num_features, n_features))

            if k_num_features < n_features : # Subspace actually smaller than full set
                # Simplified subspace generation logic from original for robustness
                if n_features <= 20 and n_features > 0: # Small number of features, try to cycle combinations
                    # Ensure k_num_features is not larger than n_features for combinations
                    actual_k = min(k_num_features, n_features)
                    all_combs = list(itertools.combinations(features, actual_k))
                    if all_combs: # If combinations are possible
                         self._subspaces = [list(all_combs[i % len(all_combs)]) for i in range(self.n_models)]
                    else: # Fallback if no combinations (e.g. k > n_features implicitly)
                         self._subspaces = [random_subspace(all_features=features, k=actual_k, rng=self._rng) for _ in range(self.n_models)]
                else: # High dimensionality or k_num_features makes combinations too many
                    self._subspaces = [random_subspace(all_features=features, k=k_num_features, rng=self._rng) for _ in range(self.n_models)]
            else: # k_num_features == n_features, so use all features
                self._subspaces = [list(features) for _ in range(self.n_models)]
        else: # Resampling uses all features, subspace is None to signify this to BaseSRPEstimator
             self._subspaces = [None] * self.n_models


    def _init_ensemble(self, features: list):
        self._generate_subspaces(features=features)
        subspace_indexes = list(range(self.n_models))
        if (self.training_method == self._TRAIN_RANDOM_PATCHES or self.training_method == self._TRAIN_RANDOM_SUBSPACES):
            self._rng.shuffle(subspace_indexes)

        # Clear existing models if any, before re-initializing
        self.data.clear()

        for i in range(self.n_models):
            subspace_for_model = self._subspaces[subspace_indexes[i]]
            if self._base_learner_class is None:
                raise RuntimeError("_base_learner_class not set in ensemble.")
            self.append( # Adds to self.data
                self._base_learner_class(
                    idx_original=i, model=self.model, metric=self.metric, #type: ignore
                    created_on=self._n_samples_seen, drift_detector=self.drift_detector,
                    warning_detector=self.warning_detector, is_background_learner=False,
                    rng=random.Random(self._rng.randint(0, 2**32 - 1)), # Give each model its own RNG derived from ensemble's
                    features=subspace_for_model,
                )
            )

    def reset(self): # Resets the ensemble
        super().reset() # Calls Ensemble.reset() which clears self.data
        self._n_samples_seen = 0
        self._rng = random.Random(self.seed) # Re-init RNG
        self._subspaces = []
        # Derived classes might need to reset their specific states here too


class BaseSRPEstimator:
    def __init__(
        self, idx_original: int, model: base.Estimator, metric: Metric | None, # Metric can be None
        created_on: int, drift_detector: base.DriftDetector | None, warning_detector: base.DriftDetector | None,
        is_background_learner, rng: random.Random, features: list | None = None, # features can be None for resampling
    ):
        self.idx_original = idx_original
        self.created_on = created_on
        self.model = model.clone()
        self.metric = metric.clone() if metric else None # Clone only if metric is provided
        self.features = features

        self.disable_drift_detector = drift_detector is None
        self.drift_detector = drift_detector.clone() if drift_detector else None
        self.disable_background_learner = warning_detector is None
        self.warning_detector = warning_detector.clone() if warning_detector else None

        self.is_background_learner = is_background_learner
        self.n_drifts_detected = 0
        self.n_warnings_detected = 0
        self.rng = rng
        self._background_learner: BaseSRPEstimator | None = None


    def _trigger_warning(self, all_features: list, n_samples_seen: int):
        if self.warning_detector is None: return # Should not happen if disable_background_learner is False

        subspace_for_background = None
        if self.features is not None: # If current model uses a feature subset
            subspace_for_background = random_subspace(all_features=all_features, k=len(self.features), rng=self.rng)
        # Else (self.features is None, e.g. resampling), background also uses all features (subspace_for_background remains None)

        self._background_learner = self.__class__(
            idx_original=self.idx_original, model=self.model, metric=self.metric, # type: ignore
            created_on=n_samples_seen,
            drift_detector=self.drift_detector.clone() if self.drift_detector else None, # Background gets its own detectors
            warning_detector=self.warning_detector.clone() if self.warning_detector else None, # (though it won't use warning_detector itself)
            is_background_learner=True,
            rng=random.Random(self.rng.randint(0, 2**32 - 1)), # New RNG for background
            features=subspace_for_background,
        )
        self.warning_detector = self.warning_detector.clone() # Hard-reset current model's warning detector

    def reset(self, all_features: list, n_samples_seen: int):
        if not self.disable_background_learner and self._background_learner is not None:
            # Promote background learner
            self.model = self._background_learner.model
            self.drift_detector = self._background_learner.drift_detector
            self.warning_detector = self._background_learner.warning_detector # Takes over warning detector too
            self.metric = self._background_learner.metric
            self.created_on = self._background_learner.created_on
            self.features = self._background_learner.features
            self.rng = self._background_learner.rng # Take over RNG too
            self._background_learner = None
        else:
            # No background learner to promote, so reset current model
            self.model = self.model.clone() # Re-clone original base model
            self.metric = self.metric.clone() if self.metric else None
            self.created_on = n_samples_seen
            if self.drift_detector: self.drift_detector = self.drift_detector.clone()
            if self.warning_detector: self.warning_detector = self.warning_detector.clone() # Reset warning detector too

            # Regenerate feature subspace if applicable
            if self.features is not None: # If it was using specific features
                 self.features = random_subspace(all_features=all_features, k=len(self.features), rng=self.rng)
            # If self.features was None (resampling), it remains None.
        # Reset drift/warning counts for the "new" model
        self.n_drifts_detected = 0 # Resetting this here for the model itself
        self.n_warnings_detected = 0

def random_subspace(all_features: list, k: int, rng: random.Random):
    if not all_features: return []
    corrected_k = min(len(all_features), k)
    if corrected_k <= 0 : return []
    return rng.sample(all_features, k=corrected_k)


class SRPClassifier(BaseSRPEnsemble, base.Classifier):
    def __init__(
        self, model: base.Estimator | None = None, n_models: int = 10,
        subspace_size: int | float | str = 0.6, training_method: str = "patches",
        lam: float = 6.0,
        drift_detector: base.DriftDetector | None = None,
        warning_detector: base.DriftDetector | None = None,
        disable_detector: str = "off", disable_weighted_vote: bool = False,
        seed: int | None = None, metric: ClassificationMetric | None = None,
    ):
        if model is None: model = HoeffdingTreeClassifier(grace_period=50, delta=0.01)

        # Configure detectors based on disable_detector flag
        _drift_detector_actual = drift_detector
        _warning_detector_actual = warning_detector

        if disable_detector == "off":
            if _drift_detector_actual is None: _drift_detector_actual = ADWIN(delta=1e-5)
            if _warning_detector_actual is None: _warning_detector_actual = ADWIN(delta=1e-4)
        elif disable_detector == "drift":
            _drift_detector_actual, _warning_detector_actual = None, None
        elif disable_detector == "warning":
            _warning_detector_actual = None # Keep drift_detector if user provided one, or if it was default for 'off'
            if _drift_detector_actual is None and drift_detector is None: # If user didn't specify drift and warning disabled, drift also off
                 _drift_detector_actual = ADWIN(delta=1e-5) # Still give a default drift if not specified by user
        else: raise AttributeError(f"Invalid disable_detector: {disable_detector}. Valid options: 'off', 'drift', 'warning'")

        super().__init__(
            model=model, n_models=n_models, subspace_size=subspace_size, training_method=training_method,
            lam=lam, drift_detector=_drift_detector_actual, warning_detector=_warning_detector_actual,
            disable_detector=disable_detector, disable_weighted_vote=disable_weighted_vote,
            seed=seed, metric=metric or metrics.Accuracy(), # Default to Accuracy for classification
        )
        self._base_learner_class = BaseSRPClassifier # type: ignore

    @property
    def _multiclass(self): return True

    def predict_proba_one(self, x, **kwargs):
        y_pred_counter = collections.Counter()
        if not self.data: # If ensemble not initialized
            self._init_ensemble(features=list(x.keys())) # Initialize if first call
            return {} # Return empty dict as per ARF behavior for unlearned ensemble

        for model_member in self: # Iterate over BaseSRPClassifier instances
            y_proba_tree = model_member.predict_proba_one(x, **kwargs)
            if not self.disable_weighted_vote and model_member.metric is not None:
                metric_value = model_member.metric.get()
                # Ensure metric_value is float, some metrics might return dicts
                if isinstance(metric_value, dict): # e.g. F1 or Precision might return dict
                    # Heuristic: try to get a primary value if dict, or default to 1.0
                    # This part is tricky if metric isn't a single float.
                    # For Accuracy, .get() is a float.
                    # Let's assume metric.get() returns a float or can be treated as such for weighting.
                    # A more robust solution would require specific handling for different metric types.
                    # For now, we'll assume it's a float or can be cast.
                    try:
                        weight_val = float(metric_value)
                    except (TypeError, ValueError):
                        weight_val = 1.0 # Default weight if conversion fails
                else:
                    weight_val = float(metric_value) if metric_value is not None else 0.0


                if weight_val > 0.0: # Standard weighting
                    y_proba_tree = {k_label: val * weight_val for k_label, val in y_proba_tree.items()}
            y_pred_counter.update(y_proba_tree)

        total = sum(y_pred_counter.values())
        if total > 0: return {label: proba / total for label, proba in y_pred_counter.items()}
        return {}

class BaseSRPClassifier(BaseSRPEstimator):
    def __init__(
        self, idx_original: int, model: base.Classifier, metric: ClassificationMetric | None,
        created_on: int, drift_detector: base.DriftDetector | None, warning_detector: base.DriftDetector | None,
        is_background_learner, rng: random.Random, features: list | None =None,
    ):
        super().__init__(
            idx_original, model, metric, created_on, drift_detector, warning_detector,
            is_background_learner, rng, features=features
        )

    def learn_one(self, x: dict, y: base.typing.ClfTarget, *, w: float, n_samples_seen: int, **kwargs,):
        x_subset = {k: x[k] for k in self.features if k in x} if self.features is not None else x
        
        # If x_subset becomes empty due to feature selection, and original x was not empty,
        # it might be an issue. For now, proceed; base model handles empty input.
        if not x_subset and x and self.features is not None: # Only if features were specified and resulted in empty
            # Potentially skip learning or use full x as fallback, this depends on desired strategy.
            # print(f"Warning: Model {self.idx_original} has empty feature subset for non-empty input.")
            pass


        for _ in range(int(w)): # w is float from Poisson, cast to int
            if hasattr(self.model, 'learn_one'):
                 self.model.learn_one(x_subset, y, **kwargs) # type: ignore

        if self._background_learner:
            self._background_learner.learn_one(x, y, w=w, n_samples_seen=n_samples_seen, **kwargs) # type: ignore

        if not self.is_background_learner:
            correctly_classifies = False
            if hasattr(self.model, 'predict_one'):
                y_pred_for_detector = self.model.predict_one(x_subset) # type: ignore
                if y_pred_for_detector is not None:
                    correctly_classifies = (y_pred_for_detector == y)
            
            drift_input = int(not correctly_classifies)

            if not self.disable_background_learner and self.warning_detector:
                self.warning_detector.update(drift_input)
                if self.warning_detector.drift_detected:
                    self.n_warnings_detected += 1
                    self._trigger_warning(all_features=list(x.keys()), n_samples_seen=n_samples_seen)

            if not self.disable_drift_detector and self.drift_detector:
                self.drift_detector.update(drift_input)
                if self.drift_detector.drift_detected:
                    self.n_drifts_detected += 1 # This count is used by the ensemble
                    self.reset(all_features=list(x.keys()), n_samples_seen=n_samples_seen)

    def predict_proba_one(self, x: dict, **kwargs) -> dict[base.typing.ClfTarget, float]:
        x_subset = {k: x[k] for k in self.features if k in x} if self.features is not None else x
        if hasattr(self.model, 'predict_proba_one'):
            return self.model.predict_proba_one(x_subset, **kwargs) # type: ignore
        return {}

    def predict_one(self, x: dict, **kwargs) -> base.typing.ClfTarget | None:
        x_subset = {k: x[k] for k in self.features if k in x} if self.features is not None else x
        if hasattr(self.model, 'predict_one'):
            return self.model.predict_one(x_subset, **kwargs) # type: ignore
        
        if hasattr(self.model, 'predict_proba_one'): # Fallback
            probas = self.model.predict_proba_one(x_subset, **kwargs) # type: ignore
            if probas: return max(probas, key=probas.get) # type: ignore
        return None


# ================================================================
# NEW Class: SRPClassifierDynamicWeights
# ================================================================

class SRPClassifierDynamicWeights(SRPClassifier):
    """
    Streaming Random Patches Classifier with dynamic tree weighting based on
    the 0.9/1.1 multiplicative update rule.

    This class modifies the weighting scheme of the standard SRPClassifier.
    Instead of using the configured metric (e.g., accuracy) of each base model
    for weighted voting, it maintains a separate performance score.
    This score is updated multiplicatively: multiplied by 0.9 if the base
    model's prediction for the current instance was correct, and by 1.1 if
    incorrect. Voting weights are derived inversely from these scores and
    normalized.

    The `disable_weighted_vote` parameter from the parent class is effectively
    ignored for the primary voting mechanism implemented here.
    The `metric` associated with individual base models is still updated
    (if provided) for monitoring or other potential uses, but it does not
    drive the voting weights in this class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._dynamic_perf_scores: list[float] = []
        self._dynamic_weights: list[float] = []

        # If ensemble was already initialized by super's init (e.g. from a pre-trained state,
        # though SRP typically isn't "pre-trained" like that), ensure dynamic weights are set up.
        if self.data: # self.data is the list of models in base.Ensemble
            self._init_dynamic_weights()

    def _init_dynamic_weights(self):
        """Initializes or resets the dynamic weight tracking lists."""
        # Use len(self.data) if models exist, otherwise self.n_models as a target
        num_models = len(self.data) if self.data else self.n_models
        if num_models == 0 and self.n_models > 0: # Should ideally not happen if _init_ensemble is called
            num_models = self.n_models

        self._dynamic_perf_scores = [1.0] * num_models
        equal_weight = 1.0 / num_models if num_models > 0 else 1.0 # Avoid division by zero
        self._dynamic_weights = [equal_weight] * num_models

    def _init_ensemble(self, features: list):
        """Overrides BaseSRPEnsemble._init_ensemble to include dynamic weights."""
        super()._init_ensemble(features)  # This populates self.data with BaseSRPClassifier instances
        self._init_dynamic_weights()      # Initialize scores/weights for these new models

    def _update_dynamic_weights(self):
        """Calculates normalized weights from performance scores."""
        if not self._dynamic_perf_scores: # Should not be empty if initialized
            if self.data: self._init_dynamic_weights() # Attempt re-init if data exists
            else: return # No models, no weights

        # Using 1.0 / (1.0 + score) for stability, ensures denominator > 0 if scores can be small.
        # Scores initialized at 1.0 and multiplied by 0.9/1.1 will stay positive.
        raw_weights = [1.0 / (1.0 + max(0, score)) for score in self._dynamic_perf_scores] # max(0,score) for safety
        total_weight = sum(raw_weights)

        if total_weight > 0:
            self._dynamic_weights = [w / total_weight for w in raw_weights]
        else:
            num_models = len(self._dynamic_perf_scores)
            equal_weight = 1.0 / num_models if num_models > 0 else 1.0
            self._dynamic_weights = [equal_weight] * num_models
            
    def learn_one(self, x: dict, y: base.typing.Target, **kwargs):
        """
        Learns from an instance: updates dynamic scores, lets base models learn
        and handle drift (resetting scores for drifted models), then recalculates weights.
        """
        self._n_samples_seen += 1 # Inherited from BaseSRPEnsemble logic

        if not self.data: # If ensemble is empty (first call to learn_one)
            self._init_ensemble(list(x.keys()))
            # self.data is now populated, _dynamic_perf_scores initialized.

        # Ensure dynamic weight structures are correctly sized, e.g. if predict was called first
        # or if model loading occurred.
        if len(self._dynamic_perf_scores) != len(self.data):
            self._init_dynamic_weights()


        # Stage 1: Get predictions from each model and update their dynamic performance scores.
        # Store predictions for updating the model's internal metric later.
        individual_predictions: list[base.typing.ClfTarget | None] = [None] * len(self.data)

        for i, model_member in enumerate(self.data): # model_member is a BaseSRPClassifier
            pred = model_member.predict_one(x) # Uses model_member's specific feature subset
            individual_predictions[i] = pred

            if pred is not None: # Check if the model could make a prediction
                if pred == y:
                    self._dynamic_perf_scores[i] *= 0.9
                else:
                    self._dynamic_perf_scores[i] *= 1.1
                # Optional: Clip scores, e.g., self._dynamic_perf_scores[i] = max(0.01, min(self._dynamic_perf_scores[i], 100.0))
            # If pred is None, the score remains unchanged.

        # Stage 2: Let each model learn and handle its internal drift/warning.
        # Reset dynamic score if a model detected drift and reset itself.
        for i, model_member in enumerate(self.data):
            # Update the model's own metric (e.g., Accuracy), as in standard SRP behavior.
            # This metric is not used for dynamic weighting here but kept for consistency/monitoring.
            if individual_predictions[i] is not None and model_member.metric is not None:
                model_member.metric.update(y_true=y, y_pred=individual_predictions[i])

            # Determine training intensity (Poisson k) for the current model_member
            k_train_weight = 0.0 # Must be float for BaseSRPClassifier.learn_one 'w' type hint
            if self.training_method == self._TRAIN_RANDOM_SUBSPACES:
                k_train_weight = 1.0
            else:
                k_train_weight = float(poisson(rate=self.lam, rng=self._rng))
            
            if k_train_weight == 0:
                continue # Skip learning for this model_member on this instance

            # Store pre-learn drift count to check if the model resets itself
            old_n_drifts = model_member.n_drifts_detected
            
            # Model learns. This call might trigger internal drift detection and model reset.
            model_member.learn_one(x=x, y=y, w=k_train_weight, n_samples_seen=self._n_samples_seen, **kwargs)

            # If the model reset itself (n_drifts_detected increased), reset its dynamic performance score.
            if model_member.n_drifts_detected > old_n_drifts:
                self._dynamic_perf_scores[i] = 1.0  # Reset score to neutral

        # Stage 3: Recalculate normalized dynamic weights for voting
        self._update_dynamic_weights()

    def predict_proba_one(self, x: dict, **kwargs) -> dict[base.typing.ClfTarget, float]:
        """
        Predicts class probabilities using the dynamically calculated weights.
        """
        y_pred_probas_aggregated = collections.Counter()

        if not self.data: # Ensemble not initialized
            self._init_ensemble(list(x.keys())) # Initializes models and dynamic weights setup
            return {} # Return empty probabilities for an unlearned/empty ensemble

        # Ensure dynamic weights are available and correctly sized
        if not self._dynamic_weights or len(self._dynamic_weights) != len(self.data):
            self._init_dynamic_weights() # (Re-)initialize based on current model count
            if not self.data : # if still no models (edge case)
                 return {}

        for i, model_member in enumerate(self.data):
            # model_member.predict_proba_one(x) correctly handles its feature subset
            tree_probas = model_member.predict_proba_one(x, **kwargs)
            
            # Use our dynamic weight for this model
            weight = self._dynamic_weights[i]

            if weight > 0.0: # Only consider models with positive weight
                for label, proba in tree_probas.items():
                    y_pred_probas_aggregated[label] += proba * weight
        
        # Normalize the summed probabilities. Although individual weights are normalized,
        # direct summation of weighted probas might not perfectly sum to 1.
        total_sum_probas = sum(y_pred_probas_aggregated.values())
        if total_sum_probas > 0:
            return {label: proba / total_sum_probas for label, proba in y_pred_probas_aggregated.items()}
        
        return {} # Return empty if no model contributed or sum is zero

    def reset(self):
        """Resets the ensemble and its dynamic weighting states."""
        super().reset() # Resets base SRP ensemble (clears models, _n_samples_seen, etc.)
        # Also reset dynamic weighting specific attributes
        self._dynamic_perf_scores = []
        self._dynamic_weights = []
        # _init_dynamic_weights will be called again by _init_ensemble on next learn/predict.