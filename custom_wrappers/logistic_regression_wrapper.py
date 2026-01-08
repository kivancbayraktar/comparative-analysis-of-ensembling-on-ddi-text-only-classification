from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from typing import Optional
import os

from typing import Any, Callable
from ddi_fw.ml.model_wrapper import ModelWrapper
import tensorflow as tf
from tensorflow import keras
# from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import numpy as np
from tensorflow.keras import Model
from ddi_fw.ml.evaluation_helper import Metrics, evaluate

# import tf2onnx
# import onnx

from ddi_fw.ml.tracking_service import TrackingService
import ddi_fw.utils as utils
import os
import warnings

def convert_to_categorical(arr, num_classes):
    """
    This function takes an array of labels and converts them to one-hot encoding 
    if they are not binary-encoded. If the array is already in a 
    compatible format, it returns the original array.

    Parameters:
    - arr: numpy array with label data (could be binary-encoded or label-encoded)
    - num_classes: number of classes to be used in one-hot encoding

    Returns:
    - The one-hot encoded array if the original array was binary or label encoded
    - The original array if it doesn't require any conversion
    """

    try:
        # First, check if the array is binary-encoded
        if not utils.is_binary_encoded(arr):
            # If the arr labels are binary-encoded, convert them to one-hot encoding
            return tf.keras.utils.to_categorical(np.argmax(arr, axis=1), num_classes=num_classes)
        else:
            print("No conversion needed, returning original array.")
            return arr
    except Exception as e:
        # If binary encoding check raises an error, print it and continue to label encoding check
        print(f"Error while checking binary encoding: {e}")

    try:
        # Check if the array is label-encoded
        if utils.is_label_encoded(arr):
            # If the arr labels are label-encoded, convert them to one-hot encoding
            return tf.keras.utils.to_categorical(arr, num_classes=num_classes)
    except Exception as e:
        # If label encoding check raises an error, print it
        print(f"Error while checking label encoding: {e}")
        # If the arr labels don't match any of the known encodings, raise an error
        raise ValueError("Unknown label encoding format.")

    # If no conversion was needed, return the original array

    return arr


class LogisticRegressionModelWrapper(ModelWrapper):
    """
    Wrapper around sklearn.linear_model.LogisticRegression.
    Supports optional feature scaling and internal StratifiedKFold CV (cv_folds).
    """
    def __init__(self, date, descriptor, model_func=None, tracking_service: Optional[TrackingService] = None, **kwargs):
        super().__init__(date, descriptor, model_func, **kwargs)
        self.tracking_service = tracking_service
        # default params suitable for multinomial classification
        self.params = kwargs.get("params", {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "multi_class": "multinomial",
            "max_iter": 1000,
            "class_weight": None,
            "n_jobs": -1
        })
        self.scale = kwargs.get("scale", False)
        self.cv_folds: Optional[int] = kwargs.get("cv_folds", None)
        self.num_classes = None

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.size == 0:
            return X
        return X

    def _labels_to_indices(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        if y.ndim > 1:
            return np.argmax(y, axis=1)
        return y.astype(int)

    def fit_model(self, X_train, y_train, X_valid=None, y_valid=None):
        X_train = self._prepare_X(np.array(X_train))
        if X_valid is not None:
            X_valid = self._prepare_X(np.array(X_valid))

        scaler = None
        if self.scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            if X_valid is not None:
                X_valid = scaler.transform(X_valid)

        y_train_idx = self._labels_to_indices(y_train)
        y_valid_idx = self._labels_to_indices(y_valid) if y_valid is not None else None

        model = LogisticRegression(**self.params)
        model.fit(X_train, y_train_idx)

        if scaler is not None:
            setattr(model, "_scaler", scaler)

        evals_result = {}
        if X_valid is not None and y_valid_idx is not None:
            preds = model.predict(X_valid)
            val_acc = accuracy_score(y_valid_idx, preds)
            evals_result['validation_accuracy'] = val_acc

        return model, evals_result

    def fit(self):
        print(f"Training {self.descriptor} LogisticRegression model...")
        models = {}
        models_val_acc = {}

        # detect num_classes from train_label (one-hot or label-encoded)
        self.num_classes = int(self.train_label.shape[1]) if getattr(self.train_label, "ndim", 0) > 1 else int(np.unique(self.train_label).size)

        # external splits provided
        if self.train_idx_arr and self.val_idx_arr:
            for i, (train_idx, val_idx) in enumerate(zip(self.train_idx_arr, self.val_idx_arr)):
                print(f"Validation {i}")
                X_train_cv = self.train_data[train_idx]
                y_train_cv = self.train_label[train_idx]
                X_valid_cv = self.train_data[val_idx]
                y_valid_cv = self.train_label[val_idx]

                def fit_model_cv_func():
                    model, evals_result = self.fit_model(X_train_cv, y_train_cv, X_valid_cv, y_valid_cv)
                    return model, evals_result

                if self.tracking_service:
                    model, evals_result = self.tracking_service.run(
                        run_name=f'Validation {i}', description='CV models', nested_run=True, func=fit_model_cv_func)
                else:
                    model, evals_result = fit_model_cv_func()

                models[f'{self.descriptor}_validation_{i}'] = model
                models_val_acc[f'{self.descriptor}_validation_{i}'] = evals_result.get('validation_accuracy', 0.0)

        # internal Stratified K-Fold if requested and no external splits
        if (not self.train_idx_arr or not self.val_idx_arr) and self.cv_folds and self.cv_folds > 1:
            y_indices = self._labels_to_indices(self.train_label)
            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            for i, (train_idx, val_idx) in enumerate(skf.split(self.train_data, y_indices)):
                print(f"Internal CV fold {i+1}/{self.cv_folds}")
                X_train_cv = self.train_data[train_idx]
                y_train_cv = self.train_label[train_idx]
                X_valid_cv = self.train_data[val_idx]
                y_valid_cv = self.train_label[val_idx]

                def fit_model_cv_func():
                    model, evals_result = self.fit_model(X_train_cv, y_train_cv, X_valid_cv, y_valid_cv)
                    return model, evals_result

                if self.tracking_service:
                    model, evals_result = self.tracking_service.run(
                        run_name=f'Internal CV {i}', description='Internal CV', nested_run=True, func=fit_model_cv_func)
                else:
                    model, evals_result = fit_model_cv_func()

                models[f'{self.descriptor}_internal_cv_{i}'] = model
                models_val_acc[f'{self.descriptor}_internal_cv_{i}'] = evals_result.get('validation_accuracy', 0.0)

        # If no CV runs happened, perform single fit on full training set
        if not models:
            def fit_model_func():
                model, evals_result = self.fit_model(self.train_data, self.train_label, None, None)
                return model, evals_result

            if self.tracking_service:
                model, evals_result = self.tracking_service.run(
                    run_name=f'Training', description='Training', nested_run=True, func=fit_model_func)
            else:
                model, evals_result = fit_model_func()

            models[self.descriptor] = model
            models_val_acc[self.descriptor] = evals_result.get('validation_accuracy', 0.0)

        # select best model by validation accuracy if available
        if models_val_acc:
            best_model_key = max(models_val_acc, key=models_val_acc.get)
            best_model = models[best_model_key]
            print("best model key:", best_model_key)
            return best_model, best_model_key, None

        # fallback
        return models.get(self.descriptor), None, None

    def predict(self):
        if self.best_model is None:
            raise RuntimeError("Model not trained yet.")
        X_test = np.asarray(self.test_data)
        scaler = getattr(self.best_model, "_scaler", None)
        if scaler is not None:
            X_test = scaler.transform(X_test)
        preds = self.best_model.predict(X_test)
        return preds

    def fit_and_evaluate(self, print_detail=False) -> tuple[dict[str, Any], Metrics, Any]:
        """
        Fit the model, evaluate it, and log results using the tracking service.
        """
        self.best_model = None

        def evaluate_and_log(artifact_uri=None):
            best_model, best_model_key, _ = self.fit()
            self.best_model = best_model

            pred = self.predict()
            pred = convert_to_categorical(pred.flatten(), self.num_classes)
            actual = self.test_label

            logs, metrics = evaluate(
                actual=actual, pred=pred, info=self.descriptor, print_detail=print_detail)
            metrics.format_float()

            if self.tracking_service:
                self.tracking_service.log_metrics(logs)
                self.tracking_service.log_param('best_cv', best_model_key)
                if artifact_uri:
                    utils.compress_and_save_data(
                        metrics.__dict__, artifact_uri, f'{self.date}_metrics.gzip')
                    self.tracking_service.log_artifact(
                        f'{artifact_uri}/{self.date}_metrics.gzip')

            return logs, metrics, pred

        if self.tracking_service:
            return self.tracking_service.run(run_name=self.descriptor, description="Fit and evaluate the model", nested_run=True, func=evaluate_and_log)
        else:
            return evaluate_and_log()