import numpy as np
from methods.data import DataContainerReal1
from methods.drol import DRoL
from methods.utils import reward
from methods.others import ERM, UDA, GroupDRO

years = [2013, 2014, 2015, 2016]
seasons = ['SP', 'SU', 'AU', 'WI']
worst_rewards = {(year, season): {} for year in years for season in seasons}

filename = 'air_pollution.csv'
source_sites = ['Aotizhongxin', 'Dongsi', 'Tiantan', 'Guanyuan', 'Wanliu']
for year in years:
    for season in seasons:
        data = DataContainerReal1()
        data.load_data(filename, source_sites=source_sites, season=season, year=year)
        # ERM
        erm = ERM(data)
        erm.fit(outcome_learner='xgb')
        worst_rewards[(year, season)]['ERM'] = np.min([reward(Y_target, erm.predict(X_target)) for X_target, Y_target in zip(data.X_targets_list, data.Y_targets_list)])
        # UDA
        uda = UDA(data)
        uda.fit(outcome_learner='xgb', density_learner='xgb')
        worst_rewards[(year, season)]['UDA'] = np.min([reward(Y_target, uda.predict(X_target)) for X_target, Y_target in zip(data.X_targets_list, data.Y_targets_list)])
        # GroupDRO
        gdro = GroupDRO(data, hidden_dims=[4, 8, 16, 32, 8])
        gdro.fit(epochs=500)
        worst_rewards[(year, season)]['GroupDRO'] = np.min([reward(Y_target, gdro.predict(X_target)) for X_target, Y_target in zip(data.X_targets_list, data.Y_targets_list)])
        # DRoL
        drol = DRoL(data)
        drol.fit(outcome_learner='xgb', density_learner='xgb')
        _, weights = drol.predict(bias_correct=True, priors=None)
        rewards_drol = []
        for X_target, Y_target in zip(data.X_targets_list, data.Y_targets_list):
            pred_mat = np.zeros((X_target.shape[0], data.L))
            for l in range(data.L):
                pred_mat[:, l] = drol.source_full_models[l].predict(X_target)
            pred_drol = pred_mat @ weights
            rewards_drol.append(reward(Y_target, pred_drol))
        worst_rewards[(year, season)]['DRoL'] = np.min(rewards_drol)