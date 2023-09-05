library(ranger)
library(stats)
library(MASS)
library(CVXR)
source('funs.R')

sim.round = 1 # {1,...,5}
n.per = 1000 # {500,1000,2000,4000}
r = 0.5 # {0.2,0.5,0.8}
close(f)

print(paste0('mixture_ratio-L2-varyr&n-r',r,'-nper',n.per,'-sim.round',sim.round))
nsim = 100
reward.cluster = mse.cluster = rep(list(NA), nsim)
for(i.sim in 1:nsim){
  print(paste0('simulation--->',i.sim))
  ratio.tr = c(r, 1-r)
  
  ## variables not changed
  L = 2
  p = 5
  n = n.per*L ## sample size for the whole data group
  n0 = 5000
  
  ## conditional outcomes
  B = readRDS('matrixB.rds')
  f.list = rep(list(NA),L)
  for(l in 1:L){
    f <- function(x){
      x.p = poly(x,2,raw=T)
      value = as.vector(x.p%*%B[,l]) - sum(B[c(2,5,9,14,20),l])
      return(value)
    }
    f.list[[l]] = f
  }
  
  ## no covariate shift
  set.seed(NULL)
  mu = rep(0,p); Sigma = diag(p)
  mu0 = mu; Sigma0 = Sigma
  ind.group = sample(1:L, size=n, replace=TRUE, prob=ratio.tr)
  ind.AB = sample(1:2, size=n, replace=TRUE, prob=c(0.5,0.5))
  ## covariates and outcome
  X = mvrnorm(n, mu, Sigma)
  Y = rep(NA, n)
  for(l in 1:L){
    map = (ind.group == l)
    Y[map] = f.list[[l]](X[map,]) + rnorm(sum(map))
  }
  
  ## fitting models
  myrf = rf.funs()
  train.fun = myrf$train.fun
  pred.fun = myrf$pred.fun
  
  fit.erm = train.fun(X,Y)
  fit.A.list = fit.B.list = rep(list(NA), L)
  for(l in 1:L){
    map.A = (ind.group==l)#&(ind.AB==1)
    fit.A.list[[l]] = train.fun(X[map.A,], Y[map.A])
    map.B = (ind.group==l)#&(ind.AB==2)
    fit.B.list[[l]] = fit.A.list[[l]]#train.fun(X[map.B,], Y[map.B], verbose=F)
  }
  
  ## generate X0 
  X0 = mvrnorm(n0, mu0, Sigma0)
  pred0.erm = pred.fun(fit.erm, X0)
  ## identify mixture weights
  pred0.mat = matrix(NA, nrow=n0, ncol=L)
  for(l in 1:L) pred0.mat[,l] = 0.5*pred.fun(fit.A.list[[l]], X0) + 0.5*pred.fun(fit.B.list[[l]], X0)
  
  ## Magging
  tGamma = t(pred0.mat)%*%pred0.mat/n0
  q = Variable(L)
  obj = quad_form(q, tGamma)
  constraints = list(q>=0, sum(q)==1)
  prob = Problem(Minimize(obj), constraints)
  result = solve(prob)
  q.opt = result$getValue(q)
  
  hGamma = tGamma
  #### bias correction function ####
  bias_correct <- function(fk, fl, wl, Xl, Yl){
    nl = nrow(Xl)
    fkX = pred.fun(fk, Xl)
    flX = pred.fun(fl, Xl)
    return(mean(wl*fkX*(flX - Yl)))
  }
  for(k in 1:L){
    for(l in 1:L){
      ##### Gamma-A #####
      f.kA = fit.A.list[[k]]
      f.lA = fit.A.list[[l]]
      map.kB = (ind.group == k)#&(ind.AB == 2)
      map.lB = (ind.group == l)#&(ind.AB == 2)
      X.lB = X[map.lB,]; Y.lB = Y[map.lB]
      num1 = bias_correct(f.kA, f.lA, rep(1, sum(map.lB)), X[map.lB,], Y[map.lB])
      num2 = bias_correct(f.lA, f.kA, rep(1, sum(map.kB)), X[map.kB,], Y[map.kB])
      ##### Gamma-B #####
      f.kB = fit.B.list[[k]]
      f.lB = fit.B.list[[l]]
      map.kA = (ind.group == k)#&(ind.AB == 1)
      map.lA = (ind.group == l)#&(ind.AB == 1)
      num3 = bias_correct(f.kB, f.lB, rep(1, sum(map.lA)), X[map.lA,], Y[map.lA])
      num4 = bias_correct(f.lB, f.kB, rep(1, sum(map.kA)), X[map.kA,], Y[map.kA])
      ##### hGamma #####
      hGamma[k,l] = hGamma[k,l] - (num1 + num2 + num3 + num4)/2
    }
  }
  eig.val = eigen(hGamma)$values
  eig.vec = eigen(hGamma)$vectors
  eig.val = pmax(eig.val, 1e-6)
  hGamma = eig.vec %*% diag(eig.val) %*% t(eig.vec)
  
  compute_v <- function(Gamma, const=L, v0){
    v = Variable(L)
    obj = quad_form(v, Gamma)
    constraints = list(v>=0, sum(v)==1, cvxr_norm(v - v0)/sqrt(L)<= const)
    prob = Problem(Minimize(obj), constraints)
    result = solve(prob)
    v.opt = result$getValue(v)
    return(v.opt)
  }
  
  v.mag.0 = compute_v(hGamma,const=10, v0=ratio.tr)
  v.mag.1 = compute_v(hGamma,const=0.2, v0=ratio.tr)
  v.mag.2 = compute_v(hGamma,const=0.1, v0=ratio.tr)
  v.mag.3 = compute_v(hGamma,const=0.05, v0=ratio.tr)
  v.mag.4 = compute_v(hGamma,const=0.01, v0=ratio.tr)
  
  ## generate evaluation
  r0.vec = seq(0,1,by=0.05)
  mse.mat = matrix(NA, nrow=length(r0.vec), ncol=9)
  colnames(mse.mat) = c('r0','erm','sqloss','mag','mag0','mag1','mag2','mag3','mag4')
  reward.mat = matrix(NA, nrow=length(r0.vec), ncol=9)
  colnames(reward.mat) = c('r0','erm','sqloss','mag','mag0','mag1','mag2','mag3','mag4')
  
  Y0 = rep(NA, n0)
  for(i.r0 in 1:length(r0.vec)){
    r0 = r0.vec[i.r0]
    ratio.te = c(r0, 1-r0)
    ind.group.0 = sample(1:L, size=n0, replace=TRUE, prob=ratio.te)
    for(l in 1:L){
      map = (ind.group.0 == l)
      if(sum(map)>0){
        Y0[map] = f.list[[l]](X0[map,]) + rnorm(sum(map))
      }
    }
    mse.erm = mean( (Y0 - pred0.erm)^2)
    mse.sq = mean((Y0 - pred0.mat%*%rep(1/L,L))^2)
    mse.mag = mean((Y0 - pred0.mat%*%q.opt)^2)
    mse.0 = mean((Y0 - pred0.mat%*%v.mag.0)^2)
    mse.1 = mean((Y0 - pred0.mat%*%v.mag.1)^2)
    mse.2 = mean((Y0 - pred0.mat%*%v.mag.2)^2)
    mse.3 = mean((Y0 - pred0.mat%*%v.mag.3)^2)
    mse.4 = mean((Y0 - pred0.mat%*%v.mag.4)^2)
    mse.mat[i.r0,] = c(r0, mse.erm, mse.sq, mse.mag, mse.0, mse.1, mse.2, mse.3, mse.4)
    
    re.erm = mean(Y0^2 - (Y0 - pred0.erm)^2)
    re.sq = mean(Y0^2 - (Y0 - pred0.mat%*%rep(1/L,L))^2)
    re.mag = mean(Y0^2 - (Y0 - pred0.mat%*%q.opt)^2)
    re.0 = mean(Y0^2 - (Y0 - pred0.mat%*%v.mag.0)^2)
    re.1 = mean(Y0^2 - (Y0 - pred0.mat%*%v.mag.1)^2)
    re.2 = mean(Y0^2 - (Y0 - pred0.mat%*%v.mag.2)^2)
    re.3 = mean(Y0^2 - (Y0 - pred0.mat%*%v.mag.3)^2)
    re.4 = mean(Y0^2 - (Y0 - pred0.mat%*%v.mag.4)^2)
    reward.mat[i.r0,] = c(r0, re.erm,re.sq, re.mag, re.0,re.1,re.2,re.3,re.4)
  }
  reward.cluster[[i.sim]] = reward.mat
  mse.cluster[[i.sim]] = mse.mat
}
data = list(reward = reward.cluster,
            mse = mse.cluster)




