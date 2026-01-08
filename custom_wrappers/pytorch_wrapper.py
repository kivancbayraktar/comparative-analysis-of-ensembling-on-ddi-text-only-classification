import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow
from typing import Any, Dict, Tuple
from ddi_fw.ml.evaluation_helper import Metrics, evaluate
from ddi_fw.ml.model_wrapper import ModelWrapper
import ddi_fw.utils as utils
import numpy as np

import tensorflow as tf
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


class PTModelWrapper(ModelWrapper):
    def __init__(self, date, descriptor, model_func, **kwargs):
        super().__init__(date, descriptor, model_func, **kwargs)
        self.batch_size = kwargs.get('batch_size',128)
        self.epochs = kwargs.get('epochs',100)
        self.model_func = model_func
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # self.optimizer = kwargs['optimizer']
        
        # self.criterion = kwargs['criterion']
        self.loss_function = torch.nn.CrossEntropyLoss()

    def fit_model(self, model , X_train, y_train, X_valid, y_valid):

       
        criterion = self.loss_function.to(self.device)
        optimizer =torch.optim.Adam(model.parameters(), lr=0.001)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        valid_dataset = TensorDataset(torch.tensor(X_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)

        best_loss = float('inf')
        best_model = None

        for epoch in range(self.epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

            valid_loss = self._validate(model, criterion, valid_loader)
            print(f'Epoch {epoch+1}/{self.epochs}, Validation Loss: {valid_loss:.4f}')
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = model.state_dict()

        model.load_state_dict(best_model)
        return model, best_loss

    def _validate(self, model, criterion, valid_loader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in valid_loader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                output = model(batch_X)
                loss = criterion(output, batch_y)
                total_loss += loss.item()
        return total_loss / len(valid_loader)

    def fit(self):
        models = {}
        models_val_acc = {}
        
        self.kwargs['input_shape'] = self.train_data.shape
        self.num_classes = len(np.unique(self.test_label, axis=0))
        self.kwargs['num_classes'] = self.num_classes
        
        for i, (train_idx, val_idx) in enumerate(zip(self.train_idx_arr, self.val_idx_arr)):
            print(f"Validation {i}")
            model = self.model_func(**self.kwargs).to(self.device)
            with mlflow.start_run(run_name=f'Validation {i}', description='CV models', nested=True) as cv_fit:
                X_train_cv = self.train_data[train_idx]
                y_train_cv = self.train_label[train_idx]
                X_valid_cv = self.train_data[val_idx]
                y_valid_cv = self.train_label[val_idx]
                model, best_loss = self.fit_model(model, X_train_cv, y_train_cv, X_valid_cv, y_valid_cv)
                models[f'{self.descriptor}_validation_{i}'] = model
                models_val_acc[f'{self.descriptor}_validation_{i}'] = best_loss

        best_model_key = min(models_val_acc,  key=lambda k: models_val_acc[k])
        best_model = models[best_model_key]
        return best_model, best_model_key

    def predict(self):
        test_dataset = TensorDataset(torch.tensor(self.test_data, dtype=torch.float32), torch.tensor(self.test_label, dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        self.best_model.eval()
        preds = []
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                output = self.best_model(batch_X)
                preds.append(output)
        return torch.cat(preds, dim=0).cpu().numpy()

    def fit_and_evaluate(self) -> Tuple[Dict[str, Any], Metrics, Any]:
        with mlflow.start_run(run_name=self.descriptor, description="***", nested=True) as run:
            print(run.info.artifact_uri)
            best_model, best_model_key = self.fit()
            print(best_model_key)
            self.best_model = best_model
            pred = self.predict()
            pred = convert_to_categorical(pred.astype(np.int32), self.num_classes)
            logs, metrics = evaluate(actual=self.test_label, pred=pred, info=self.descriptor)
            metrics.format_float()
            mlflow.log_metrics(logs)
            mlflow.log_param('best_cv', best_model_key)
            utils.compress_and_save_data(metrics.__dict__, run.info.artifact_uri, f'{self.date}_metrics.gzip')
            mlflow.log_artifact(f'{run.info.artifact_uri}/{self.date}_metrics.gzip')

            return logs, metrics, pred

# from typing import Any
# import mlflow
# import torch
# from ddi_fw.ml.evaluation_helper import Metrics, evaluate
# from ddi_fw.ml.model_wrapper import ModelWrapper


# class PTModelWrapper(ModelWrapper):
#     def __init__(self, date, descriptor, model_func, batch_size=128, epochs=100, **kwargs):
#         super().__init__(date, descriptor, model_func, batch_size, epochs)
#         self.optimizer = kwargs['optimizer']
#         self.criterion = kwargs['criterion']

#     def _create_dataloader(self, data, labels):
#         dataset = torch.utils.data.TensorDataset(data, labels)
#         return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

#     def predict(self):
#         print(self.train_data.shape)

#         with mlflow.start_run(run_name=self.descriptor, description="***", nested=True) as run:
#             models = {}
#             # models_val_acc = {}

#             for i, (train_idx, val_idx) in enumerate(zip(self.train_idx_arr, self.val_idx_arr)):
#                 print(f"Validation {i}")

#                 with mlflow.start_run(run_name=f'Validation {i}', description='CV models', nested=True) as cv_fit:
#                     model = self.model_func(self.train_data.shape[1])
#                     models[f'validation_{i}'] = model

#                     # Create DataLoaders
#                     X_train_cv = torch.tensor(self.train_data[train_idx], dtype=torch.float16)
#                     y_train_cv = torch.tensor(self.train_label[train_idx], dtype=torch.float16)
#                     X_valid_cv = torch.tensor(self.train_data[val_idx], dtype=torch.float16)
#                     y_valid_cv = torch.tensor(self.train_label[val_idx], dtype=torch.float16)

#                     train_loader = self._create_dataloader(X_train_cv, y_train_cv)
#                     valid_loader = self._create_dataloader(X_valid_cv, y_valid_cv)

#                     optimizer = self.optimizer
#                     criterion = self.criterion
#                     best_val_loss = float('inf')

#                     for epoch in range(self.epochs):
#                         model.train()
#                         for batch_X, batch_y in train_loader:
#                             optimizer.zero_grad()
#                             output = model(batch_X)
#                             loss = criterion(output, batch_y)
#                             loss.backward()
#                             optimizer.step()

#                         model.eval()
#                         with torch.no_grad():
#                             val_loss = self._validate(model, valid_loader)

#                         # Callbacks after each epoch
#                         for callback in self.callbacks:
#                             callback.on_epoch_end(epoch, logs={'loss': loss.item(), 'val_loss': val_loss.item()})

#                         if val_loss < best_val_loss:
#                             best_val_loss = val_loss
#                             best_model = model

#                     # Evaluate on test data
#                     with torch.no_grad():
#                         pred = best_model(torch.tensor(self.test_data, dtype=torch.float16))
#                         logs, metrics = evaluate(
#                             actual=self.test_label, pred=pred.numpy(), info=self.descriptor)
#                         mlflow.log_metrics(logs)

#             return logs, metrics, pred.numpy()

#     def _validate(self, model, valid_loader):
#         total_loss = 0
#         criterion = self.criterion

#         for batch_X, batch_y in valid_loader:
#             output = model(batch_X)
#             loss = criterion(output, batch_y)
#             total_loss += loss.item()

#         return total_loss / len(valid_loader)
    
#     def fit_and_evaluate(self)  -> tuple[dict[str, Any], Metrics, Any]:
#         return None,None,None