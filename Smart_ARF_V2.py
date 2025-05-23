from river import forest, tree, metrics, drift, stats
from river.utils.random import poisson
import matplotlib.pyplot as plt
from river.drift import ADWIN
import collections # Added for SmartARF's _reindex_tracker_dict
import math # Added for Hoeffding bound

class SmartARF_V2(forest.ARFClassifier):
    def __init__(self,
                 max_models=30,
                 accuracy_drop_threshold=0.5,
                 monitor_window=1000,
                 min_instances_for_pruning=100,
                 hoeffding_delta_prune=0.05,
                 min_samples_for_hoeffding=100,
                 pruning_strategy_at_capacity="accuracy_metric_only",
                 **kwargs): # <--- nominal_attributes will be caught here

        super().__init__(**kwargs) # <--- and passed to ARFClassifier
        self.max_models = max_models
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.monitor_window = monitor_window
        self.model_count_history = []
        self._accuracy_window = []
        self._warned_tree_ids = set()
        self._warning_step = {}
        self._warned_recent_acc = {}
        self.min_instances_for_pruning = min_instances_for_pruning
        self.hoeffding_delta_prune = hoeffding_delta_prune
        self.min_samples_for_hoeffding = min_samples_for_hoeffding
        self.pruning_strategy_at_capacity = pruning_strategy_at_capacity
        self._instance_counts = []

        if self.data: # If superclass initialized models
            self._accuracy_window = [[] for _ in self.data]
            self._instance_counts = [0 for _ in self.data]


    def _init_ensemble(self, features: list):
        super()._init_ensemble(features)
        self._accuracy_window = [[] for _ in self.data]
        self._instance_counts = [0 for _ in self.data]

    def _drift_detector_input(self, tree_id: int, y_true, y_pred) -> float | int:
        return super()._drift_detector_input(tree_id, y_true, y_pred)

    def learn_one(self, x, y, **kwargs):
        if not self.data:
            self._init_ensemble(sorted(x.keys()))

        current_step = len(self.model_count_history)
        self.model_count_history.append(len(self.data))

        model_errors_for_this_instance = {}

        for i_phase1 in range(len(self.data)):
            model = self.data[i_phase1]
            y_pred = model.predict_one(x)
            self._metrics[i_phase1].update(y_true=y, y_pred=y_pred)
            self._ensure_accuracy_window_exists(i_phase1)
            self._accuracy_window[i_phase1].append(int(y_pred == y))
            if len(self._accuracy_window[i_phase1]) > self.monitor_window:
                self._accuracy_window[i_phase1].pop(0)

            k = poisson(rate=self.lambda_value, rng=self._rng)
            if k > 0:
                if i_phase1 < len(self._instance_counts):
                    self._instance_counts[i_phase1] += 1
                else:
                    print(f"Warning: _instance_counts out of sync for index {i_phase1} in Phase 1")
                model.learn_one(x=x, y=y, w=k, **kwargs)

            error_val = self._drift_detector_input(i_phase1, y, y_pred)
            model_errors_for_this_instance[id(model)] = error_val

        i = 0
        while i < len(self.data):
            current_model_object = self.data[i]
            model_id = id(current_model_object)

            if model_id not in model_errors_for_this_instance:
                i += 1
                continue
            error_val = model_errors_for_this_instance[model_id]

            if not self._warning_detection_disabled:
                if i < len(self._warning_detectors):
                    self._warning_detectors[i].update(error_val)
                    if self._warning_detectors[i].drift_detected:
                        if i < len(self._background) and self._background[i] is None:
                            print(f"⚠️ Warning detected in tree {i}. Creating background learner.")
                            self._background[i] = self._new_base_model()
                        self._warning_detectors[i] = self.warning_detector.clone()
                        if self._warning_tracker is not None: self._warning_tracker[i] += 1
                        self._warned_tree_ids.add(i)
                        self._warning_step[i] = current_step
                        self._warned_recent_acc[i] = self._get_recent_accuracy(i)
                else: print(f"Warning: _warning_detectors out of sync for index {i}")

            perform_drift_action = False
            new_tree_from_background = None
            if not self._drift_detection_disabled:
                if i < len(self._drift_detectors):
                    self._drift_detectors[i].update(error_val)
                    if self._drift_detectors[i].drift_detected:
                        perform_drift_action = True
                        if i < len(self._background) and self._background[i] is not None:
                            new_tree_from_background = self._background[i]
                            print(f"🌊 Drift detected in tree {i}. Background learner is ready.")
                        else:
                            print(f"🌊 Drift detected in tree {i}, no background. Will reset/replace.")
                else: print(f"Warning: _drift_detectors out of sync for index {i}")

            if perform_drift_action:
                original_drifted_model_idx = i
                if new_tree_from_background:
                    idx_to_prune = -1
                    made_space = False
                    if len(self.data) >= self.max_models:
                        idx_to_prune = self._find_model_to_remove_at_capacity(new_tree_from_background)
                        if idx_to_prune != -1:
                            metric_val_pruned = self._metrics[idx_to_prune].get() if idx_to_prune < len(self._metrics) else "N/A"
                            print(f"Max capacity: Pruning tree {idx_to_prune} (score: {metric_val_pruned}) to make space.")
                            self._remove_model(idx_to_prune)
                            made_space = True
                            if idx_to_prune < original_drifted_model_idx: i -= 1
                        else: # No model pruned
                            print(f"Max capacity: No prune for strategy '{self.pruning_strategy_at_capacity}'. Resetting {original_drifted_model_idx}.")
                            self.data[i] = self._new_base_model()
                            self._reset_model_attributes(i)
                            self._background[i] = None
                            if self._drift_tracker is not None: self._drift_tracker[i] += 1
                            i += 1
                            continue
                    if len(self.data) < self.max_models or made_space:
                        print(f"Adding new tree from background of original slot {original_drifted_model_idx}.")
                        self.data.append(new_tree_from_background)
                        self._metrics.append(self.metric.clone())
                        self._drift_detectors.append(self.drift_detector.clone())
                        self._warning_detectors.append(self.warning_detector.clone())
                        self._background.append(None)
                        self._accuracy_window.append([])
                        self._instance_counts.append(0)
                        if i < len(self.data) and id(self.data[i]) == model_id:
                            self._background[i] = None
                            self._drift_detectors[i] = self.drift_detector.clone()
                            if self._drift_tracker is not None: self._drift_tracker[i] += 1
                else: # No background, reset current tree
                    print(f"Drift in tree {i}, no background. Resetting tree {i}.")
                    self.data[i] = self._new_base_model()
                    self._reset_model_attributes(i)
                    self._background[i] = None
                    if self._drift_tracker is not None: self._drift_tracker[i] += 1
                i += 1
            else: i += 1
        self._check_prune_on_accuracy_drop(current_step)
        return self

    def _reset_model_attributes(self, index):
        if index < len(self._metrics): self._metrics[index] = self.metric.clone()
        if index < len(self._drift_detectors): self._drift_detectors[index] = self.drift_detector.clone()
        if index < len(self._warning_detectors): self._warning_detectors[index] = self.warning_detector.clone()
        if index < len(self._accuracy_window): self._accuracy_window[index] = []
        if index < len(self._instance_counts): self._instance_counts[index] = 0

    def _ensure_accuracy_window_exists(self, i):
        while len(self._accuracy_window) <= i: self._accuracy_window.append([])
        while len(self._instance_counts) <= i: self._instance_counts.append(0)
        while len(self._metrics) <= i: self._metrics.append(self.metric.clone())
        while len(self._drift_detectors) <=i: self._drift_detectors.append(self.drift_detector.clone())
        while len(self._warning_detectors) <=i: self._warning_detectors.append(self.warning_detector.clone())
        while len(self._background) <=i: self._background.append(None)

    def _find_model_to_remove_at_capacity(self, new_tree_candidate=None):
        if not self.data: return -1
        candidate_indices = list(range(len(self.data)))
        if self.pruning_strategy_at_capacity in ["min_instances_only", "min_instances_then_hoeffding"]:
            eligible_indices = [
                idx for idx in candidate_indices if idx < len(self._instance_counts) and \
                                                self._instance_counts[idx] >= self.min_instances_for_pruning
            ]
            if not eligible_indices:
                print(f"Strategy '{self.pruning_strategy_at_capacity}': No trees meet min_instances ({self.min_instances_for_pruning}). No pruning.")
                return -1
            candidate_indices = eligible_indices
        if not candidate_indices: return -1

        if self.pruning_strategy_at_capacity in ["hoeffding_assisted", "min_instances_then_hoeffding"]:
            hoeffding_eligible_candidates = [
                idx for idx in candidate_indices if idx < len(self._accuracy_window) and \
                                                len(self._accuracy_window[idx]) >= self.min_samples_for_hoeffding
            ]
            if not hoeffding_eligible_candidates:
                print(f"Strategy '{self.pruning_strategy_at_capacity}': No trees meet min_samples_for_hoeffding ({self.min_samples_for_hoeffding}). Cannot apply Hoeffding.")
                return -1
            worst_hoeffding_candidate_idx = self._find_worst_model_by_metric(hoeffding_eligible_candidates)
            if worst_hoeffding_candidate_idx == -1: return -1

            error_candidate = 1.0 - self._get_recent_accuracy(worst_hoeffding_candidate_idx)
            n_candidate = len(self._accuracy_window[worst_hoeffding_candidate_idx])
            other_errors = []
            for idx in hoeffding_eligible_candidates:
                if idx == worst_hoeffding_candidate_idx: continue
                other_errors.append(1.0 - self._get_recent_accuracy(idx))
            if not other_errors:
                print(f"Strategy '{self.pruning_strategy_at_capacity}': Not enough other models for Hoeffding comparison with tree {worst_hoeffding_candidate_idx}.")
                return -1
            avg_other_error = sum(other_errors) / len(other_errors)
            epsilon = math.sqrt(math.log(1 / self.hoeffding_delta_prune) / (2 * n_candidate))
            if (error_candidate - epsilon) > avg_other_error:
                print(f"Hoeffding: Pruning tree {worst_hoeffding_candidate_idx} (error {error_candidate:.3f}, n={n_candidate}) "
                      f"significantly > avg error of others ({avg_other_error:.3f}). Epsilon: {epsilon:.3f}")
                return worst_hoeffding_candidate_idx
            else:
                print(f"Hoeffding: Not pruning tree {worst_hoeffding_candidate_idx} (error {error_candidate:.3f}, n={n_candidate}) "
                      f"not significantly > avg error of others ({avg_other_error:.3f}). Epsilon: {epsilon:.3f}")
                return -1
        if self.pruning_strategy_at_capacity == "accuracy_metric_only":
            return self._find_worst_model_by_metric(list(range(len(self.data))))
        elif self.pruning_strategy_at_capacity == "min_instances_only":
            return self._find_worst_model_by_metric(candidate_indices)
        return -1

    def _find_worst_model_by_metric(self, indices=None):
        if indices is None: indices = range(len(self.data))
        valid_indices = [i for i in indices if i < len(self._metrics)]
        if not valid_indices: return -1
        worst_idx = -1
        # metric.bigger_is_better can be None for some metrics, default to True if so.
        bigger_is_better = getattr(self.metric, 'bigger_is_better', True)
        if bigger_is_better:
            worst_score = float('inf')
            for i in valid_indices:
                score = self._metrics[i].get()
                if score < worst_score:
                    worst_score = score; worst_idx = i
        else:
            worst_score = float('-inf')
            for i in valid_indices:
                score = self._metrics[i].get()
                if score > worst_score:
                    worst_score = score; worst_idx = i
        return worst_idx

    def _check_prune_on_accuracy_drop(self, current_step):
        to_remove_indices = []
        for i in list(self._warned_tree_ids):
            if not (0 <= i < len(self.data)):
                self._warned_tree_ids.discard(i)
                self._warning_step.pop(i, None)
                self._warned_recent_acc.pop(i, None)
                continue
            age_since_warning = current_step - self._warning_step.get(i, current_step + 1)
            if age_since_warning > self.monitor_window:
                current_acc = self._get_recent_accuracy(i)
                past_acc = self._warned_recent_acc.get(i, 1.0)
                if current_acc < self.accuracy_drop_threshold * past_acc:
                    print(f"📉 Tree {i} accuracy drop: {past_acc:.2f} → {current_acc:.2f} (threshold {self.accuracy_drop_threshold:.2f}). Pruning.")
                    to_remove_indices.append(i)
                else:
                    print(f"Tree {i} accuracy stable: {past_acc:.2f} → {current_acc:.2f}. Removing from warning.")
                self._warned_tree_ids.discard(i)
                self._warning_step.pop(i, None)
                self._warned_recent_acc.pop(i, None)
        for i_rem in sorted(to_remove_indices, reverse=True):
            if 0 <= i_rem < len(self.data): self._remove_model(i_rem)

    def _get_recent_accuracy(self, i):
        if not (0 <= i < len(self._accuracy_window)) or not self._accuracy_window[i]:
            return 0.0
        accs = self._accuracy_window[i]
        return sum(accs) / len(accs)

    def _remove_model(self, index):
        if not (0 <= index < len(self.data)):
            print(f"Error: Attempted to remove model at invalid index {index} (len data: {len(self.data)}).")
            return

        metric_val = self._metrics[index].get() if index < len(self._metrics) else "N/A"
        recent_acc_val = self._get_recent_accuracy(index) if index < len(self._accuracy_window) else "N/A"
        instance_count_val = self._instance_counts[index] if index < len(self._instance_counts) else "N/A"
        print(f"🪓 Removing tree {index} (overall score: {metric_val}, recent acc: {recent_acc_val}, instances: {instance_count_val})")

        del self.data[index]
        if index < len(self._metrics): del self._metrics[index]
        if index < len(self._drift_detectors): del self._drift_detectors[index]
        if index < len(self._warning_detectors): del self._warning_detectors[index]
        if index < len(self._background): del self._background[index]
        if index < len(self._accuracy_window): del self._accuracy_window[index]
        if index < len(self._instance_counts): del self._instance_counts[index]

        new_warned_tree_ids = set()
        for tree_id in self._warned_tree_ids:
            if tree_id == index: continue
            new_warned_tree_ids.add(tree_id - 1 if tree_id > index else tree_id)
        self._warned_tree_ids = new_warned_tree_ids
        self._warning_step = { (k - 1 if k > index else k):v for k,v in self._warning_step.items() if k != index }
        self._warned_recent_acc = { (k - 1 if k > index else k):v for k,v in self._warned_recent_acc.items() if k != index }
        if self._warning_tracker is not None: self._warning_tracker = self._reindex_tracker_dict(self._warning_tracker, index)
        if self._drift_tracker is not None: self._drift_tracker = self._reindex_tracker_dict(self._drift_tracker, index)

    def _reindex_tracker_dict(self, tracker_dict, removed_index):
        new_tracker = collections.defaultdict(int)
        for tree_id, count in tracker_dict.items():
            if tree_id == removed_index: continue
            new_tracker[tree_id - 1 if tree_id > removed_index else tree_id] = count
        return new_tracker