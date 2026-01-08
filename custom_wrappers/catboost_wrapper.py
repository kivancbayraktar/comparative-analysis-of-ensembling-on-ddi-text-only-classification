import xgboost as xgb
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

from catboost import CatBoostClassifier, Pool

class CatBoostModelWrapper(ModelWrapper):

    def __init__(self, date, descriptor, model_func, tracking_service: Optional[TrackingService] = None, **kwargs):
        super().__init__(date, descriptor, model_func, **kwargs)
        print("wrapper")
        print(kwargs)
        print(self.kwargs)
        self.params = kwargs
        self.tracking_service = tracking_service
        self.num_rounds = kwargs.get('num_rounds', 5)
        self.early_stopping_rounds = kwargs.get('early_stopping_rounds', 10)
        # self.params = kwargs.get('params', {})
        if not self.params:
            print("Using default CatBoost parameters.")
            self.params = {
                'iterations': 100,              # Number of boosting rounds
                'learning_rate': 0.1,           # Learning rate
                'depth': 8,                     # Tree depth
                'loss_function': 'MultiClass',  # Multi-class classification
                'eval_metric': 'Accuracy',      # Evaluation metric
                'verbose': 50,                  # Print progress every 50 iterations
                'random_seed': 1,
                'task_type': 'GPU',             # Change to 'GPU' if you have GPU
            }

    def fit_model(self, X_train, y_train, X_valid, y_valid):
        # Create DMatrix for XGBoost

        X_train = np.array(X_train)
        X_valid = np.array(X_valid)

        # Convert one-hot encoded labels to class indices (should already be 1D)
        # Convert to 1D class indices
        y_train_indices = np.argmax(y_train, axis=1)
        # Convert to 1D class indices
        y_valid_indices = np.argmax(y_valid, axis=1)

        # Prepare CatBoost Pool (optimized dataset format)
        dtrain = Pool(X_train, label=y_train_indices)
        dvalid = Pool(X_valid, label=y_valid_indices) if X_valid is not None else None
        self.num_classes = y_train.shape[1]
        # # Set up the evaluation list
        # evals = [(dtrain, 'train')]
        # if dvalid is not None:
        #     evals.append((dvalid, 'valid'))
        evals_result = {}
            # # Number of classes (as per the one-hot encoding)
            # if not 'num_class' in self.params:
            #     self.num_classes = y_train.shape[1]
            #     self.params['num_class'] = self.num_classes
            # else:
            #     self.num_classes = self.params['num_class']

        # Train the model
        # Initialize the CatBoost model
        model = CatBoostClassifier(**self.params)

        # Train the model with early stopping
        model.fit(dtrain, eval_set=dvalid, early_stopping_rounds=20)
        evals_result = model.evals_result_
        return model, evals_result

    def fit(self):
        print(f"Training {self.descriptor} model...")
        models = {}
        models_val_acc = {}

        if self.train_idx_arr and self.val_idx_arr:
            for i, (train_idx, val_idx) in enumerate(zip(self.train_idx_arr, self.val_idx_arr)):
                print(f"Validation {i}")
                X_train_cv = self.train_data[train_idx]
                y_train_cv = self.train_label[train_idx]
                X_valid_cv = self.train_data[val_idx]
                y_valid_cv = self.train_label[val_idx]

                def fit_model_cv_func():
                    model, evals_result = self.fit_model(
                        X_train_cv, y_train_cv, X_valid_cv, y_valid_cv)
                    return model, evals_result

                if self.tracking_service:
                    model, evals_result = self.tracking_service.run(
                        run_name=f'Validation {i}', description='CV models', nested_run=True, func=fit_model_cv_func)
                else:
                    model, evals_result = fit_model_cv_func()

                models[f'{self.descriptor}_validation_{i}'] = model
                # Here you can extract validation metrics from the model
                # Assuming the validation set is provided, we get accuracy as an example:
                if 'validation' in evals_result:
                    # val_acc = evals_result['valid']['rmse'][self.early_stopping_rounds-1]
                    val_acc = evals_result['validation']['Accuracy'][-1]
                    models_val_acc[f'{self.descriptor}_validation_{i}'] = val_acc

        else:
            def fit_model_func():
                model, evals_result = self.fit_model(
                    self.train_data, self.train_label, None, None)
                return model, evals_result

            if self.tracking_service:
                model, evals_result = self.tracking_service.run(
                    run_name=f'Training', description='Training', nested_run=True, func=fit_model_func)
            else:
                model, evals_result = fit_model_func()

            models[self.descriptor] = model
            # Extract validation accuracy if available
            models_val_acc[self.descriptor] = model['learn']['Accuracy'][-1]  # or any metric you prefer
        # print(model.attributes())
        print(models_val_acc)
        # Return the best model based on validation accuracy
        if models_val_acc:
            best_model_key = max(models_val_acc, key=models_val_acc.get)
            print("best model key:", best_model_key)
            best_model = models[best_model_key]
            return best_model, best_model_key
        print(models.keys())
        return models[self.descriptor], None

    def predict(self):
        # Assuming the model is already trained
        # dtest = Pool(self.test_data, label=y_test_indices)
        y_test_indices = np.argmax(self.test_label, axis=1)
        dtest = Pool(self.test_data, label=y_test_indices)
        pred = self.best_model.predict(dtest)
        
        
        # y_pred = pred.flatten().astype(int)  # Flatten and ensure integer labels

        # # Evaluate accuracy
        # accuracy = accuracy_score(y_test_indices, y_pred)
        # print(f'Accuracy: {accuracy:.4f}')
        
        return pred

    def fit_and_evaluate(self, print_detail=False) -> tuple[dict[str, Any], Metrics, Any]:
        """
        Fit the model, evaluate it, and log results using the tracking service.

        Args:
            print_detail (bool): Whether to print detailed evaluation logs.

        Returns:
            tuple: A tuple containing logs, metrics, and predictions.
        """
        self.best_model: any = None

        def evaluate_and_log(artifact_uri=None):
            # Fit the model
            best_model, best_model_key = self.fit()
            self.best_model = best_model

            # Make predictions
            pred = self.predict()
            pred = convert_to_categorical(pred.flatten(), self.num_classes)
            actual = self.test_label

            # Evaluate the model
            logs, metrics = evaluate(
                actual=actual, pred=pred, info=self.descriptor, print_detail=print_detail
            )
            metrics.format_float()

            if self.tracking_service:
                # Log metrics and parameters
                self.tracking_service.log_metrics(logs)
                self.tracking_service.log_param('best_cv', best_model_key)

                # Save metrics to the artifact URI if provided
                if artifact_uri:
                    utils.compress_and_save_data(
                        metrics.__dict__, artifact_uri, f'{self.date}_metrics.gzip'
                    )
                    self.tracking_service.log_artifact(
                        f'{artifact_uri}/{self.date}_metrics.gzip'
                    )

            return logs, metrics, pred

        # Use the tracking service to run the evaluation
        if self.tracking_service:
            return self.tracking_service.run(
                run_name=self.descriptor,
                description="Fit and evaluate the model",
                nested_run=True,
                func=evaluate_and_log
            )
        else:
            # If no tracking service is provided, run the evaluation directly
            return evaluate_and_log()
