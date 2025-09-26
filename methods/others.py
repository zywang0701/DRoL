"""
Date: 2025/03/18
Summary of the module:
This file defines three main functionalities:
1. ERM: A class implementing Empirical Risk Minimization method.
2. UDA: A class implementing Unsupervised Domain Adaptation method that accounts for covariate shift by estimating density ratios.
3. GroupDRO: A class implementing Group Distributionally Robust Optimization using Mirror Ascent for handling group weights in regression tasks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from methods.utils import *

class ERM:
    def __init__(self, data):
        self.data = data
        self.model = None
    
    def fit(self, outcome_learner='xgb'):
        X_merged = np.vstack(self.data.X_sources_list)
        Y_merged = np.concatenate(self.data.Y_sources_list)
        self.model = OutcomeModel(learner=outcome_learner, params=None)
        self.model.fit(X_merged, Y_merged)
    
    def predict(self, X_target):
        return self.model.predict(X_target)


class UDA:
    def __init__(self, data):
        self.data = data
        self.model = None
    
    def fit(self, outcome_learner='xgb', density_learner='logistic', params=None):
        # Learn density ratios
        density_models = [DensityModel(learner=density_learner, params=params) for _ in range(self.data.L)]
        sample_weights = []
        for l in range(self.data.L):
            density_models[l].fit(self.data.X_sources_list[l], self.data.X_target)
            omega_l = density_models[l].predict(self.data.X_sources_list[l])
            sample_weights.append(omega_l)
        sample_weights = np.concatenate(sample_weights)
        
        # Merge all source data
        X_merged = np.vstack(self.data.X_sources_list)
        Y_merged = np.concatenate(self.data.Y_sources_list)
        
        # Sample-weighted training
        self.model = OutcomeModel(learner=outcome_learner, params=None)
        self.model.fit(X_merged, Y_merged, sample_weight=sample_weights)
    
    def predict(self, X_target):
        return self.model.predict(X_target)


class GroupDRO:
    
    def __init__(self, data, hidden_dims):
        """_summary_

        Args:
            data (_type_): DataContainer
            input_dim (int): Number of features.
            hidden_dims (List[int], optional): Sizes of hidden layers.
        """
        self.data = data
        # Define the neural network architecture
        layers = []
        prev_dim = self.data.d
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))  # Output layer for regression
        self.model = nn.Sequential(*layers)
    
    def fit(self, lr_model=1e-3, eta=0.1, epochs=100, weight_decay=1e-5, early_stopping=True, 
            early_stopping_patience=5, min_delta=1e-4, verbose=False):
        """
        Trains the model using Group DRO with Mirror Ascent for group weights.
        Adds optional early stopping based on improvements in weighted loss.

        Args:
            lr_model (float, optional): Learning rate for the model optimizer. Defaults to 1e-3.
            eta (float, optional): Learning rate for mirror ascent. Defaults to 0.1.
            epochs (int, optional): Max number of training epochs. Defaults to 100.
            weight_decay (float, optional): Weight decay for optimizer. Defaults to 1e-5.
            early_stopping (bool, optional): Whether to use early stopping. Defaults to False.
            early_stopping_patience (int, optional): # epochs to wait for improvement. Defaults to 5.
            min_delta (float, optional): Minimum improvement to reset patience. Defaults to 1e-4.
            verbose (bool, optional): If True, prints training progress. Defaults to False.
        """
        # Optimizer for model parameters
        optimizer_model = optim.Adam(self.model.parameters(), lr=lr_model, weight_decay=weight_decay)
        # Loss function (Mean Squared Error)
        loss_fn = nn.MSELoss(reduction='mean')  # Compute mean loss per group
        # Initialize group weights
        weights = torch.ones(self.data.L) / self.data.L
        
        self.model.train()
        # Convert all data to PyTorch tensors
        X_groups = [torch.tensor(X, dtype=torch.float32) for X in self.data.X_sources_list]
        Y_groups = [torch.tensor(Y, dtype=torch.float32).unsqueeze(1) for Y in self.data.Y_sources_list]
        
        # Track best state for early stopping
        best_loss = float('inf')
        best_model_state = copy.deepcopy(self.model.state_dict())
        best_weights = weights.detach().clone()
        no_improve_epochs = 0
        
        for epoch in range(1, epochs + 1):
            group_losses = []
            
            # compute loss for each group
            for l in range(self.data.L):
                X = X_groups[l]
                Y = Y_groups[l]
                preds = self.model(X)
                loss = loss_fn(preds, Y) - torch.mean(Y**2) # negative reward
                group_losses.append(loss)
            
            group_losses_tensor = torch.stack(group_losses)  # shape: [num_groups]
            
            # Mirror Ascent: update group weights
            group_losses_tensor_clip = group_losses_tensor.detach()
            group_losses_tensor_clip = group_losses_tensor_clip - group_losses_tensor_clip.max()  # stabilize
            new_weights = weights * torch.exp(eta * group_losses_tensor_clip)
            new_weights = new_weights / new_weights.sum()
            weights = new_weights.detach()
            
            # Weighted loss
            weighted_loss = torch.dot(weights, group_losses_tensor)
            
            # Backprop and update model parameters
            optimizer_model.zero_grad()
            weighted_loss.backward()
            optimizer_model.step()
            
            # Convert to float
            current_loss = weighted_loss.item()
            # Early stopping logic
            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                best_weights = weights.detach().clone()
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if verbose and (epoch == 1 or epoch % max(1, epochs // 20) == 0):
                print(f"Epoch {epoch}/{epochs}, Weighted Loss: {current_loss:.6f}, Best: {best_loss:.6f}")

            if early_stopping and no_improve_epochs >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (no improvement in {early_stopping_patience} epochs).")
                break

        # Restore best model weights and group weights
        self.model.load_state_dict(best_model_state)
        print('model saved')

    def predict(self, X, batch_size=512):
        """
        Makes predictions using the trained model.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted continuous outcomes.
        """
        self.model.eval()
        print('model eval')
        preds_list = []
        with torch.no_grad():
            for i in range(0, X.shape[0], batch_size):
                X_batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32)
                preds_batch = self.model(X_batch).detach().numpy()
                preds_list.append(preds_batch)
        return np.vstack(preds_list).flatten()
