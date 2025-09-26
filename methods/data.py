import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

class DataContainerSimu1:
    def __init__(self, n, N):
        self.n = n            # number of samples in each source domain
        self.N = N            # number of samples in the target domain
        self.d = 5            # number of features
        self.L = None         # number of source domains
        self.X_sources_list = []  # list of source covariate matrices
        self.Y_sources_list = []  # list of source outcome vectors
        self.X_target = None  # target covariate matrix
        self.Y_target_potential_list = []  # list of potential target outcome vectors
        self.f_funcs = []   # list of source conditional outcome functions
        self.mu0 = None       # target covariate distribution mean, used when generating data
        self.Sigma0 = None    # target covariate distribution covariance, used when generating data
        
    def generate_funcs_list(self, L, seed=None):
        np.random.seed(seed)
        self.L = L
        beta_list = []
        A_list = []
        self.mu0 = np.array([1, -1, 0.5, 0., 0.])
        X_sample = np.random.randn(20000, self.d) + self.mu0
        for l in range(L):
            # random beta in [-1,1]^d
            beta = np.random.uniform(-1, 1, self.d)
            beta_list.append(beta)
            
            # random symmetric matrix A
            B = np.random.uniform(-0.5, 0.5, size=(self.d, self.d))
            A = (B + B.T) / 2
            A_list.append(A)
            
            # compute c = trace(A) + mu^T A mu
            c = np.trace(A) + self.mu0.dot(A.dot(self.mu0))
            
            def f_func(x, beta=beta, A=A, c=c):
                return np.sin(x.dot(beta)) + np.sum(np.dot(x, A) * x, axis=1) - c
            self.f_funcs.append(lambda x, f_func=f_func: f_func(x) - np.mean(f_func(X_sample)))
    
    def generate_data(self, seed=None):
        np.random.seed(seed)
        self._reset_data_lists()
        
        # ------- Generate Source Data -------
        mu = np.zeros(self.d)
        Sigma = np.eye(self.d)
        for l in range(self.L):
            X = np.random.multivariate_normal(mu, Sigma, self.n)
            Y = self.f_funcs[l](X) + np.random.randn(self.n)
            self.X_sources_list.append(X)
            self.Y_sources_list.append(Y)
        
        # ------- Generate Target Data -------
        self.Sigma0 = np.eye(self.d)
        self.X_target = np.random.multivariate_normal(self.mu0, self.Sigma0, self.N)
        for l in range(self.L):
            Y_target =  self.f_funcs[l](self.X_target) + np.random.randn(self.N)
            self.Y_target_potential_list.append(Y_target)
    
    def _reset_data_lists(self):
        self.X_sources_list = []
        self.Y_sources_list = []
        self.Y_target_potential_list = []
        

class DataContainerSimu2:
    def __init__(self, n, N, N_label):
        self.n = n            # number of samples in each source domain
        self.N = N            # number of samples in the target domain
        self.N_label = N_label # number of labeled samples in the target domain
        self.d = 5            # number of features
        self.L = None         # number of source domains
        self.X_sources_list = []  # list of source covariate matrices
        self.Y_sources_list = []  # list of source outcome vectors
        self.X_target = None  # target covariate matrix
        self.Y_target = None  # target outcome vector
        self.X_target_label = None 
        self.X_target_label = None
        self.f_funcs = []   # list of source conditional outcome functions
        self.mu0 = None       # target covariate distribution mean, used when generating data
        self.Sigma0 = None    # target covariate distribution covariance, used when generating data
        
    def generate_funcs_list(self, L=4, seed=None):
        np.random.seed(seed)
        self.L = L
        beta_list = []
        A_list = []
        self.mu0 = np.array([1, -1, 0.5, 0., 0.])
        X_sample = np.random.randn(20000, self.d) + self.mu0
        for l in range(L):
            # random beta in [-1,1]^d
            beta = np.random.uniform(-1, 1, self.d)
            beta_list.append(beta)
            
            # random symmetric matrix A
            B = np.random.uniform(-0.5, 0.5, size=(self.d, self.d))
            A = (B + B.T) / 2
            A_list.append(A)
            
            # compute c = trace(A) + mu^T A mu
            c = np.trace(A) + self.mu0.dot(A.dot(self.mu0))
            
            def f_func(x, beta=beta, A=A, c=c):
                return np.sin(x.dot(beta)) + np.sum(np.dot(x, A) * x, axis=1) - c
            self.f_funcs.append(lambda x, f_func=f_func: f_func(x) - np.mean(f_func(X_sample)))
    
    def generate_data(self, seed=None):
        np.random.seed(seed)
        self._reset_data_lists()
        
        # ------- Generate Source Data -------
        mu = np.zeros(self.d)
        Sigma = np.eye(self.d)
        for l in range(self.L):
            X = np.random.multivariate_normal(mu, Sigma, self.n)
            Y = self.f_funcs[l](X) + np.random.randn(self.n)
            self.X_sources_list.append(X)
            self.Y_sources_list.append(Y)
        
        # ------- Generate Target Data -------
        self.Sigma0 = np.eye(self.d)
        self.X_target = np.random.multivariate_normal(self.mu0, self.Sigma0, self.N)
        weights = np.concatenate([[0.6], 0.4 / np.ones(self.L - 1)])
        self.Y_target = np.random.randn(self.N)
        for l in range(self.L):
            self.Y_target += weights[l] * self.f_funcs[l](self.X_target)
        self.X_target_label = self.X_target[:self.N_label]
        self.Y_target_label = self.Y_target[:self.N_label]
    
    def _reset_data_lists(self):
        self.X_sources_list = []
        self.Y_sources_list = []
        

class DataContainerSimu3:
    def __init__(self, n, N):
        self.n = n            # number of samples in each source domain
        self.N = N            # number of samples in the target domain
        self.d = 5            # number of features
        self.L = None         # number of source domains
        self.X_sources_list = []  # list of source covariate matrices
        self.Y_sources_list = []  # list of source outcome vectors
        self.X_target = None  # target covariate matrix
        self.f_funcs = []   # list of source conditional outcome functions
        self.mu = np.zeros(self.d) # source covariate distribution mean, used when generating data
        self.Sigma = np.eye(self.d) # source covariate distribution covariance, used when generating data
        self.mu0 = np.array([0.5, -1, 0.25, 0., 0.])       # target covariate distribution mean, used when generating data
        self.Sigma0 = np.eye(self.d)    # target covariate distribution covariance, used when generating data
        
    def generate_funcs_list(self, L=3, seed=None):
        assert L == 3, "This version is designed for exactly 3 source domains: easy, medium, hard"
        np.random.seed(seed)
        self.L = L
        self.f_funcs = []

        X_sample = np.random.randn(20000, self.d) + self.mu0  # for normalization

        # ---------- Source 1: Easy (linear) ----------
        beta_easy = np.random.uniform(-1, 1, self.d)
        def f_easy(x):
            return 0.8 * x @ beta_easy
        y_easy = f_easy(X_sample)
        f_easy_centered = lambda x: f_easy(x) - np.mean(y_easy)

        # ---------- Source 2: Medium (quadratic) ----------
        A_medium = np.random.randn(self.d, self.d)
        A_medium = (A_medium + A_medium.T) / 2  # symmetric
        def f_medium(x):
            return 0.3 * np.einsum('ij,ij->i', x @ A_medium, x)
        y_medium = f_medium(X_sample)
        f_medium_centered = lambda x: f_medium(x) - np.mean(y_medium)

        # ---------- Source 3: Hard (nonlinear) ----------
        beta_hard = np.random.uniform(-1, 1, self.d)
        def f_hard(x):
            linear = x @ beta_hard
            return linear ** 3 - 0.5 * linear  # Adds a small linear term to avoid perfect symmetry
        y_hard = f_hard(X_sample)
        f_hard_centered = lambda x: f_hard(x) - np.mean(y_hard)

        # Store
        self.f_funcs = [f_easy_centered, f_medium_centered, f_hard_centered]
    # def generate_funcs_list(self, L=3, seed=None):
    #     np.random.seed(seed)
    #     self.L = L
    #     beta_list = []
    #     A_list = []
        
    #     X_sample = np.random.randn(20000, self.d) + self.mu0
    #     for l in range(L):
    #         # random beta in [-1,1]^d
    #         beta = np.random.uniform(-1, 1, self.d)
    #         beta_list.append(beta)
            
    #         # random symmetric matrix A
    #         B = np.random.uniform(-0.5, 0.5, size=(self.d, self.d))
    #         A = (B + B.T) / 2
    #         A_list.append(A)
            
    #         # compute c = trace(A) + mu^T A mu
    #         c = np.trace(A) + self.mu0.dot(A.dot(self.mu0))
            
    #         def f_func(x, beta=beta, A=A, c=c):
    #             return 0.2 * np.sin(x.dot(beta)) + 0.2 * np.sum(np.dot(x, A) * x, axis=1) - 0.2 * c
    #         self.f_funcs.append(lambda x, f_func=f_func: f_func(x) - np.mean(f_func(X_sample)))
    
    def generate_data(self, seed=None):
        np.random.seed(seed)
        self._reset_data_lists()
        
        # ------- Generate Source Data -------
        
        for l in range(self.L):
            X = np.random.multivariate_normal(self.mu, self.Sigma, self.n)
            Y = self.f_funcs[l](X) + 0.5 * np.random.randn(self.n)
            self.X_sources_list.append(X)
            self.Y_sources_list.append(Y)
        
        self.X_target = np.random.multivariate_normal(self.mu0, self.Sigma0, self.N)
    
    def _reset_data_lists(self):
        self.X_sources_list = []
        self.Y_sources_list = []

        
class DataContainerReal1:
    def __init__(self):
        self.X_sources_list = []
        self.Y_sources_list = []
        self.X_target = None
        self.X_targets_list = []
        self.Y_targets_list = []
        self.L = None
    
    def load_data(self, filename, 
                  source_sites=['Aotizhongxin', 'Dongsi', 'Tiantan', 'Guanyuan', 'Wanliu'], 
                  season='SP', 
                  year=2013,
                  demean=True):
        df = pd.read_csv(filename)
        all_stations = df['station'].unique()
        target_sites = [station for station in all_stations if station not in source_sites]
        self.L = len(source_sites)
        
        for l in range(len(all_stations)):
            df_domain = df[(df['station'] == all_stations[l]) & 
                           (df['season'] == season) &
                           (df['year'] == year)]
            df_domain['log_PM25'] = np.log(df_domain['PM2.5'] + 1)
            if demean:
                df_domain['Y'] = df_domain['log_PM25'] - df_domain['log_PM25'].mean()
            else:
                df_domain['Y'] = df_domain['log_PM25']
            df_features = df_domain.drop(columns=['year', 'season', 'station', 'PM2.5', 'log_PM25'])
            Y = np.asarray(df_features['Y'].values, dtype=float)
            X = np.asarray(df_features.drop(columns=['Y']).values, dtype=float)
            
            if all_stations[l] in source_sites:
                self.X_sources_list.append(X)
                self.Y_sources_list.append(Y)
            else:
                self.X_targets_list.append(X)
                self.Y_targets_list.append(Y)

            
class DataContainerReal2:
    def __init__(self):
        self.X_sources_list = []
        self.Y_sources_list = []
        self.X_target = None
        self.Y_target = None
        self.X_target_label = None
        self.Y_target_label = None
        self.L = None
    
    def load_data(self, filename,
                  source_sites=['Aotizhongxin', 'Dongsi', 'Tiantan', 'Guanyuan', 'Wanliu'],
                  target_site=['Nongzhanguan'],
                  season='SP',
                  year=2013,
                  demean=True,
                  N_label=100):
        df = pd.read_csv(filename)
        self.L = len(source_sites)
        all_stations = source_sites + target_site
        
        for l in range(len(all_stations)):
            df_domain = df[(df['station'] == all_stations[l]) & 
                           (df['season'] == season) &
                           (df['year'] == year)]
            df_domain['log_PM25'] = np.log(df_domain['PM2.5'] + 1)
            if demean:
                df_domain['Y'] = df_domain['log_PM25'] - df_domain['log_PM25'].mean()
            else:
                df_domain['Y'] = df_domain['log_PM25']
            df_features = df_domain.drop(columns=['year', 'season', 'station', 'PM2.5', 'log_PM25'])
            Y = np.asarray(df_features['Y'].values, dtype=float)
            X = np.asarray(df_features.drop(columns=['Y']).values, dtype=float)
            
            if all_stations[l] in source_sites:
                self.X_sources_list.append(X)
                self.Y_sources_list.append(Y)
            else:
                self.X_target = X
                self.Y_target = Y
                all_indices = np.arange(len(Y))
                np.random.shuffle(all_indices)
                self.X_target_label = X[all_indices[:N_label]]
                self.Y_target_label = Y[all_indices[:N_label]]
                