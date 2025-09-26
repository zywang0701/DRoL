import numpy as np
from methods.data import DataContainerSimu3
from methods.drol import DRoL
from methods.utils import reward
import cvxpy as cp

L = 3 # number of source domains
ns = [1000, 2000]
results = {n: {} for n in ns}
for n in ns:
    data = DataContainerSimu3(n=n, N=20000)
    # Fix the function generation at seed=0 for all runs
    data.generate_funcs_list(L=L, seed=0)
    # Generate data; seed varies for different runs
    data.generate_data(seed=1)
    
    # ------- Population DRoL -------
    pred0_pop_mat = np.zeros((data.N, data.L))
    for l in range(data.L):
        pred0_pop_mat[:, l] = data.f_funcs[l](data.X_target)
    Gamma_pop = pred0_pop_mat.T @ pred0_pop_mat / data.N
    q = cp.Variable(data.L, nonneg=True)
    objective = cp.Minimize(cp.quad_form(q, Gamma_pop))
    constraints = [cp.sum(q) == 1]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    q_opt_pop =  q.value
    pred_pop = pred0_pop_mat @ q_opt_pop
    
    # -------- DRoL oracle --------
    drol = DRoL(data)
    drol.fit(density_learner='oracle', 
             params={'X_mean': data.mu, 'X_cov': data.Sigma,
                     'X0_mean': data.mu0, 'X0_cov': data.Sigma0})
    pred_plug, q_opt_plug = drol.predict(bias_correct=False)
    pred_oracle, q_opt_oracle = drol.predict(bias_correct=True)

    # -------- DRoL logistic --------
    drol = DRoL(data)
    drol.fit(density_learner='logistic')
    pred_logistic, q_opt_logistic = drol.predict(bias_correct=True)
    
    # Evaluate q distance and pred difference
    q_dists = [np.sqrt(np.mean((q_opt_plug - q_opt_pop)**2)),
               np.sqrt(np.mean((q_opt_logistic - q_opt_pop)**2)),
               np.sqrt(np.mean((q_opt_oracle - q_opt_pop)**2))]
    pred_diffs = [np.sqrt(np.mean((pred_plug - pred_pop)**2)),
                  np.sqrt(np.mean((pred_logistic - pred_pop)**2)),
                  np.sqrt(np.mean((pred_oracle - pred_pop)**2))]
    results[n]['q_dists'] = q_dists
    results[n]['pred_diffs'] = pred_diffs
    
if __name__ == '__main__':
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(results)
    
