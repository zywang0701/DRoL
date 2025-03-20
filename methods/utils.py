import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

def reward(Y, Y_pred):
    return np.mean(Y**2 - (Y - Y_pred) ** 2)

class OutcomeModel:
    def __init__(self, learner='linear', params=None):
        """Initialize the OutcomeModel class with parameters (if applicable).
        """
        self.learner = learner
        self.params = params
        self.model = None
    
    def fit(self, X, Y, sample_weight=None, verbose=0, seed=None):
        """Fit the outcome model.
        
        Args:
            X (np.ndarray): features
            Y (np.ndarray): outcomes
        """
        if sample_weight is None:
            sample_weight = np.ones_like(Y)
 
        if self.learner == 'linear':
            self.model = LinearRegression().fit(X, Y, sample_weight=sample_weight)
            param_grid = None
        if self.learner == 'xgb':
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                eval_metric='rmse',
                n_estimators=200,
                n_jobs=-1
            )
            param_grid = {
                'learning_rate': [0.1], #[0.01, 0.05, 0.1],    # Step size shrinkage used in update to prevents overfitting
                'max_depth': [3,6],#[3, 6, 9],          # Maximum depth of a tree
                'subsample': [0.8], #, 1.0],         # Row fraction
                'colsample_bytree': [0.8] # [0.8, 1.0],  # Feature fraction
            }
            if self.params is not None:
                self.model.set_params(**self.params)
                self.model.fit(X, Y, sample_weight=sample_weight)
            else:
                kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
                grid = GridSearchCV(estimator = self.model, 
                                    param_grid = param_grid, 
                                    cv=kfold,
                                    scoring='neg_mean_squared_error',
                                    n_jobs=-1,
                                    verbose=verbose)
                grid.fit(X, Y, sample_weight=sample_weight)
                self.model = grid.best_estimator_
                self.params = grid.best_params_
    
    def predict(self, X):
        return self.model.predict(X)
    

class DensityModel:
    def __init__(self, learner='linear', params=None):
        """Initialize the DensityModel class with parameters (if applicable).
        """
        self.learner = learner
        self.params = params
        self.model = None
        self.sample_ratio = None
        
    def fit(self, X, X_target, verbose=0, seed=None):
        """Fit the density model.
        
        Args:
            X (np.ndarray): features
            X_target (np.ndarray): target features
        """
        self.sample_ratio = X.shape[0] / X_target.shape[0]
        X_concat = np.vstack((X, X_target))
        Y_concat = np.concatenate((np.zeros(X.shape[0]), np.ones(X_target.shape[0])))
        if self.learner == 'logistic':
            self.model = LogisticRegression(solver='lbfgs').fit(X_concat, Y_concat)
            param_grid = None
        if self.learner == 'xgb':
            self.model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                n_estimators=200,
                n_jobs=-1
            )
            param_grid = {
                'learning_rate': [0.1], #[0.01, 0.05, 0.1],    # Step size shrinkage used in update to prevents overfitting
                'max_depth': [3,6],#[3, 6, 9],          # Maximum depth of a tree
                'subsample': [0.8], #, 1.0],         # Row fraction
                'colsample_bytree': [0.8] # [0.8, 1.0],  # Feature fraction
            }
            # param_grid = {
            #     'learning_rate': [0.01, 0.05, 0.1],    # Step size shrinkage used in update to prevents overfitting
            #     'max_depth': [3, 6, 9],          # Maximum depth of a tree
            #     'subsample': [0.8, 1.0],         # Row fraction
            #     'colsample_bytree': [0.8, 1.0],  # Feature fraction
            # }
            if self.params is not None:
                self.model.set_params(**self.params)
                self.model.fit(X_concat, Y_concat)
            else:
                stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                grid = GridSearchCV(estimator = self.model, 
                                    param_grid = param_grid, 
                                    cv=stratified_kfold,
                                    scoring='neg_log_loss',
                                    n_jobs=-1,
                                    verbose=verbose)
                grid.fit(X_concat, Y_concat)
                self.model = grid.best_estimator_
                self.params = grid.best_params_
        
    
    def predict(self, X):
        proba_ratio = self.model.predict_proba(X)[:, 1] / self.model.predict_proba(X)[:, 0]
        omega = proba_ratio * self.sample_ratio
        return omega
