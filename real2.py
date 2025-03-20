import numpy as np
from methods.data import DataContainerReal2
from methods.drol import DRoL
from methods.utils import reward, OutcomeModel
import cvxpy as cp

years = [2013, 2014, 2015, 2016]
season = 'SU'
filename = 'air_pollution.csv'
source_sites = ['Aotizhongxin', 'Dongsi', 'Tiantan', 'Guanyuan', 'Wanliu']
target_sites = 'Nongzhanguan'
rhos = np.arange(0, 1.0, 0.05) # parameter controls size of prior
N_label = 100 # number of labeled data in target domains

rewards = {
        'DRoL_Default': np.zeros(len(years)),
        'DRoL_Unif': np.zeros((len(years), len(rhos))),
        'DRoL_Label': np.zeros((len(years), len(rhos))),
        'Target_Only': np.zeros(len(years))
    }

for i_year, year in enumerate(years):
    data = DataContainerReal2()
    data.load_data(filename, 
                   source_sites=source_sites, 
                   target_site=target_sites, 
                   season=season, 
                   year=year,
                   N_label=N_label)
    
    drol = DRoL(data)
    drol.fit(outcome_learner='xgb', density_learner='xgb')
    
    # ------ DRoL Default -------
    pred_drol_default, weights_default = drol.predict(bias_correct=True, priors=None)
    rewards['DRoL_Default'][i_year] = reward(data.Y_target, pred_drol_default)
    # ------ DRoL Unif -------
    for i_rho, rho in enumerate(rhos):
        unif_weight = np.ones(data.L) / data.L
        pred_drol_unif, weights_unif = drol.predict(bias_correct=True, priors=(unif_weight, rho))
        rewards['DRoL_Unif'][i_year, i_rho] = reward(data.Y_target, pred_drol_unif)
    
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
        rewards['DRoL_Label'][i_year, i_rho] = reward(data.Y_target, pred_drol_label)
    
    # ------ Target Only -------
    target_only = OutcomeModel(learner='xgb', params=None)
    target_only.fit(data.X_target_label, data.Y_target_label)
    pred_target_only = target_only.predict(data.X_target)
    rewards['Target_Only'][i_year] = reward(data.Y_target, pred_target_only)