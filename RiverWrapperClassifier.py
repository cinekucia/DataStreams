from capymoa.base import Classifier, Instance, Schema # Re-iterate necessary imports

class RiverWrapperClassifier(Classifier):
    """
    A wrapper to make a River classifier compatible with the CapyMOA evaluation framework.
    Corrected: Uses fallback to get feature names in __init__.
    """
    def __init__(self, river_model: RiverClassifier, schema: Schema):
        super().__init__(schema=schema) # Pass schema to the base class initializer

        if not isinstance(river_model, RiverClassifier):
             raise TypeError("river_model must be an instance of river.base.Classifier")

        self.river_model = river_model
        self._schema = schema # Keep schema reference if needed

        # Extract class label info
        self._class_labels = list(schema.get_label_values())
        self._num_classes = len(self._class_labels)
        self._class_to_index = {label: i for i, label in enumerate(self._class_labels)}

        # --- Store feature names once using FALLBACK ---
        print("Attempting manual extraction of feature names for wrapper...")
        self._feature_names = []
        try:
            # Use methods confirmed to exist
            moa_header = schema.get_moa_header()
            num_features = schema.get_num_attributes() # Number of input features

            for i in range(num_features):
                # Access the i-th MOA attribute object (input feature)
                moa_attr = moa_header.attribute(i)
                # Get its name using its Java method
                self._feature_names.append(moa_attr.name())
            print(f"Manual extraction found feature names: {self._feature_names}")

        except AttributeError as e:
            print(f"Error during manual extraction of feature names: {e}")
            print("Could not determine feature names. Wrapper might fail.")
            # If this fails, the wrapper won't work correctly later
            self._feature_names = []
        # --- End store feature names ---

        if not self._feature_names:
             print("Warning: No feature names extracted from schema.")

    def _instance_to_river_dict(self, instance: Instance) -> Dict[str, Any]:
        """Helper function to convert CapyMOA instance features to River's dict format."""
        features_array = instance.x
        if len(self._feature_names) != len(features_array):
             print(f"Warning: Mismatch! Feature names count ({len(self._feature_names)}) "
                   f"!= feature values count ({len(features_array)}). Instance: {instance}")
        return dict(zip(self._feature_names, features_array))

    def get_schema(self) -> Schema:
        """Returns the schema associated with this classifier."""
        # Assuming base class provides this after schema is passed in __init__
        return super().get_schema()

    def train(self, instance: Instance):
        """
        Trains the internal River model on a single CapyMOA instance.
        CORRECTED: Uses instance.y_label to get the ground truth label.
        """
        x_river = self._instance_to_river_dict(instance)
        # --- CORRECTED LINE ---
        y_river = instance.y_label # Use the .y_label attribute for the ground truth
        # --- END CORRECTION ---
        self.river_model.learn_one(x=x_river, y=y_river)
        
    def predict(self, instance: Instance) -> np.ndarray:
        """
        Makes a prediction for a single CapyMOA instance using the River model.
        Returns the probability distribution array.
        """
        x_river = self._instance_to_river_dict(instance)
        proba_dict = self.river_model.predict_proba_one(x=x_river)
        proba_array = np.zeros(self._num_classes, dtype=float)
        if proba_dict:
            total_proba = sum(proba_dict.values())
            if total_proba > 0:
                 for label, proba in proba_dict.items():
                    if label in self._class_to_index:
                        class_index = self._class_to_index[label]
                        proba_array[class_index] = proba / total_proba
            elif self._num_classes > 0:
                 proba_array.fill(1.0 / self._num_classes)
        elif self._num_classes > 0:
             proba_array.fill(1.0 / self._num_classes)
        return proba_array

    def predict_proba(self, instance: Instance) -> np.ndarray:
        """
        Predicts class probabilities for a single instance. Required by ABC.
        """
        return self.predict(instance)

    def __str__(self) -> str:
        """
        Returns a string representation of the classifier. Required by ABC.
        """
        return f"RiverWrapper({str(self.river_model)})"

    def reset(self):
        print("Warning: RiverWrapperClassifier.reset() called. "
              "River models often handle resets internally (like ARF) or require re-instantiation.")
