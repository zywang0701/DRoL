import numpy as np
from methods.data import DataContainerSimu1
from methods.drol import DRoL
from methods.utils import reward
from methods.others import ERM, UDA, GroupDRO

# This script corresponds to the experiment in Figure 4 (left) of the paper
# for clarity, we only report a single run of simulation, 
# while the results in the paper are averaged over 200 runs.


# ------- Get the worst-case reward for each method -------
Ls = [3, 4, 5, 6, 7, 8, 9, 10] # number of source domains
results = {L: {} for L in Ls}
for L in Ls:
    data = DataContainerSimu1(n=1000, N=20000)
    # Fix the function generation at seed=0 for all runs
    data.generate_funcs_list(L=L, seed=0)
    # Generate data; seed varies for different runs
    data.generate_data(seed=0)
    
    # ERM
    print('Training ERM...')
    erm = ERM(data)
    erm.fit(outcome_learner='xgb')
    pred_erm = erm.predict(data.X_target)
    
    # uda (Importance weighting)
    print('Training UDA...')
    uda = UDA(data)
    uda.fit(outcome_learner='xgb', density_learner='logistic')
    pred_uda = uda.predict(data.X_target)
    
    # DRoL
    drol = DRoL(data)
    drol.fit(outcome_learner='xgb', density_learner='logistic')
    pred_drol, weights = drol.predict(bias_correct=True, priors=None)
    
    # GroupDRO
    gdro = GroupDRO(data, hidden_dims=[4, 8, 16, 32, 8])
    gdro.fit(epochs=100, early_stopping=True)
    pred_gdro = gdro.predict(data.X_target)
    
    # Evaluate the worst-case reward
    worst_rewards = np.zeros(4) # ERM, UDA, GroupDRO, DRoL
    worst_rewards[0] = np.min([reward(Y_target, pred_erm) for Y_target in data.Y_target_potential_list])
    worst_rewards[1] = np.min([reward(Y_target, pred_uda) for Y_target in data.Y_target_potential_list])
    worst_rewards[2] = np.min([reward(Y_target, pred_gdro) for Y_target in data.Y_target_potential_list])
    worst_rewards[3] = np.min([reward(Y_target, pred_drol) for Y_target in data.Y_target_potential_list])

    results[L]['ERM'] = worst_rewards[0]
    results[L]['UDA'] = worst_rewards[1]
    results[L]['GroupDRO'] = worst_rewards[2]
    results[L]['DRoL'] = worst_rewards[3]

if __name__ == '__main__':
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(results)


