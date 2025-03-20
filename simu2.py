import numpy as np
from methods.data import DataContainerSimu2
from methods.drol import DRoL
from methods.utils import reward, OutcomeModel
import cvxpy as cp

num_simulations = 200
N_labels = [20, 50, 100]
rhos = np.arange(0.0, 1.0, 0.05)
results = {N_label: {} for N_label in N_labels}
for N_label in N_labels:
    data = DataContainerSimu2(n=2000, N=20000, N_label=N_label)
    data.generate_funcs_list(seed=0)
    rewards = {
        'DRoL_Default': np.zeros(num_simulations),
        'DRoL_Unif': np.zeros((num_simulations, len(rhos))),
        'DRoL_Label': np.zeros((num_simulations, len(rhos))),
        'Target_Only': np.zeros(num_simulations)
    }
    
    for sim in range(num_simulations):
        data.generate_data()
        drol = DRoL(data)
        drol.fit(outcome_learner='xgb', density_learner='logistic')
        
        # ------ DRoL Default -------
        pred_drol_default, weights_default = drol.predict(bias_correct=True, priors=None)
        rewards['DRoL_Default'][sim] = reward(data.Y_target, pred_drol_default)
        # ------ DRoL Unif -------
        for i_rho, rho in enumerate(rhos):
            unif_weight = np.ones(data.L) / data.L
            pred_drol_unif, weights_unif = drol.predict(bias_correct=True, priors=(unif_weight, rho))
            rewards['DRoL_Unif'][sim, i_rho] = reward(data.Y_target, pred_drol_unif)
        
        # ------ DRoL Label -------
        # Fit the prior weight from labeled data
        pred_label = np.zeros((N_label, data.L))
        for l in range(data.L):
            pred_label[:, l] = drol.source_full_models[l].predict(data.X_label)
        q_label = cp.Variable(data.L, nonneg=True)
        constraints = [cp.sum(q_label) == 1]
        objective = cp.Minimize(cp.sum_squares(data.Y_target_label - pred_label @ q_label) / N_label)
        prob = cp.Problem(objective, constraints)
        prob.solve()
        q_label = q_label.value
        # Evaluate the reward
        for i_rho, rho in enumerate(rhos):
            pred_drol_label, weights_label = drol.predict(bias_correct=True, priors=(q_label, rho))
            rewards['DRoL_Label'][sim, i_rho] = reward(data.Y_target, pred_drol_label)
        
        # ------ Target Only -------
        target_only = OutcomeModel(learner='xgb', params=None)
        target_only.fit(data.X_target_label, data.Y_target_label)
        pred_target_only = target_only.predict(data.X_target)
        rewards['Target_Only'][sim] = reward(data.Y_target, pred_target_only)