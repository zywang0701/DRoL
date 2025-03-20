import numpy as np
from methods.data import DataContainerSimu1
from methods.drol import DRoL
from methods.utils import reward
from methods.others import ERM, UDA, GroupDRO

# ---------------------------------------------------------
# ------- Get the worst-case reward for each method -------
# ---------------------------------------------------------
num_simulations = 200
Ls = [3, 4, 5, 6, 7, 8, 9, 10] # number of source domains
results = {L: {} for L in Ls}
for L in Ls:
    data = DataContainerSimu1(n=2000, N=20000)
    data.generate_funcs_list(L=L, seed=0)
    worst_rewards = np.zeros((num_simulations, 4)) # ERM, UDA, GroupDRO, DRoL
    for sim in range(num_simulations):
        data.generate_data()
        
        # ERM
        erm = ERM(data)
        erm.fit(outcome_learner='xgb')
        pred_erm = erm.predict(data.X_target)
        
        # uda
        uda = UDA(data)
        uda.fit(outcome_learner='xgb', density_learner='logistic')
        pred_uda = uda.predict(data.X_target)
        
        # GroupDRO
        gdro = GroupDRO(data, hidden_dims=[4, 8, 16, 32, 8])
        gdro.fit(epochs=500)
        pred_gdro = gdro.predict(data.X_target)
        
        # DRoL
        drol = DRoL(data)
        drol.fit(outcome_learner='xgb', density_learner='logistic')
        pred_drol, weights = drol.predict(bias_correct=True, priors=None)
        
        # Evaluate the worst-case reward
        worst_rewards[sim, 0] = np.min([reward(Y_target, pred_erm) for Y_target in data.Y_target_potential_list])
        worst_rewards[sim, 1] = np.min([reward(Y_target, pred_uda) for Y_target in data.Y_target_potential_list])
        worst_rewards[sim, 2] = np.min([reward(Y_target, pred_gdro) for Y_target in data.Y_target_potential_list])
        worst_rewards[sim, 3] = np.min([reward(Y_target, pred_drol) for Y_target in data.Y_target_potential_list])

    results[L]['ERM'] = np.mean(worst_rewards[:, 0])
    results[L]['UDA'] = np.mean(worst_rewards[:, 1])
    results[L]['GroupDRO'] = np.mean(worst_rewards[:, 2])
    results[L]['DRoL'] = np.mean(worst_rewards[:, 3])

