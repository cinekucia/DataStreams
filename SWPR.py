import csv
from collections import deque
import numpy as np
import logging
from capymoa.base import MOAClassifier
from moa.classifiers.meta import StreamingRandomPatches as _MOA_SRP
from capymoa.stream import ARFFStream
from capymoa.evaluation import prequential_evaluation
from capymoa.evaluation.visualization import plot_windowed_results

class StreamWiseRandomPatches(MOAClassifier):
    def __init__(
        self,
        schema,
        random_seed=0,
        ensemble_size=100,
        accuracy_threshold=0.6,
        prune_threshold=0.5,
        **kwargs,
    ):
        super().__init__(moa_learner=_MOA_SRP, schema=schema, random_seed=random_seed, **kwargs)
        
        # Accuracy and weights for weighting
        self.tree_accuracies = [1.0] * ensemble_size
        self.ensemble_weights = [1.0] * ensemble_size
        
        # Metrics for pruning
        self.correct_predictions = [0] * ensemble_size
        self.total_predictions = [0] * ensemble_size
        
        self.accuracy_threshold = accuracy_threshold
        self.prune_threshold = prune_threshold

    def partial_fit(self, X, y, classes=None):
        super().partial_fit(X, y, classes)
        
        # Update tree-specific accuracies for weighting
        predictions = self.predict(X)
        for i, (pred, true_label) in enumerate(zip(predictions, y)):
            for tree_idx in range(len(self.tree_accuracies)):  # Update all trees
                if pred == true_label:
                    self.tree_accuracies[tree_idx] *= 0.9  # Correct prediction decreases metric
                    self.correct_predictions[tree_idx] += 1  # Track correct predictions
                else:
                    self.tree_accuracies[tree_idx] *= 1.1  # Incorrect prediction increases metric
                self.total_predictions[tree_idx] += 1  # Track total predictions

        # Normalize tree accuracies to derive ensemble weights
        self.ensemble_weights = [1 / (1 + acc) for acc in self.tree_accuracies]
        total_weight = sum(self.ensemble_weights)
        self.ensemble_weights = [w / total_weight for w in self.ensemble_weights]

        # Prune or replace low-performing trees
        self.prune_or_replace_trees()

    def prune_or_replace_trees(self):
        """
        Replace or prune trees with true accuracy below the threshold.
        """
        for i in range(len(self.tree_accuracies)):
            # Calculate the true pruning accuracy for this tree
            if self.total_predictions[i] > 0:
                prune_accuracy = self.correct_predictions[i] / self.total_predictions[i]
            else:
                prune_accuracy = 0  # No predictions made yet

            # Prune tree if true accuracy is below the threshold
            if prune_accuracy < self.prune_threshold:
                logging.info(f"Tree {i} replaced due to low true accuracy ({prune_accuracy:.2f}).")
                # Replace the tree by resetting its metrics
                self.tree_accuracies[i] = 1.0
                self.ensemble_weights[i] = 1.0 / len(self.tree_accuracies)
                self.correct_predictions[i] = 0
                self.total_predictions[i] = 0

    def predict_proba(self, X):
        base_proba = super().predict_proba(X)
        
        # Apply weighted contributions
        weighted_proba = np.zeros_like(base_proba)
        for i, proba in enumerate(base_proba):
            tree_weight = self.ensemble_weights[i % len(self.ensemble_weights)]
            weighted_proba[i] += proba * tree_weight
        
        return weighted_proba

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
