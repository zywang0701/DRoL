### objective: compare mixture ratios ###
## 1. under no covariate shift setting
## 2. compare the reward for ERM, squared-loss dro, magging, magging H

library(ranger)
library(stats)
library(MASS)
library(CVXR)
source('src/funs.R')

f <- file('stdin')
open(f)
sim.round = as.numeric(readLines(f, n=1))
n.per = as.numeric(readLines(f,n=1))
L = 4
close(f)

print(paste0('mixture_ratio-L4-magH','-nper',n.per,'-sim.round',sim.round))
nsim = 25
reward.cluster = rep(list(NA), nsim)
for(i.sim in 1:nsim){
  print(paste0('simulation--->',i.sim))
  ## variables changed
  ratio.tr = c(0.15, 0.15, 0.55, 0.15)
  v0 = ratio.tr
  
  ## variables not changed
  p = 5
  n = n.per*L ## sample size for the whole data group
  n0 = 10000
  
  ## conditional outcomes
  B = readRDS('src/matrixB.rds')
  f.list = rep(list(NA),L)
  for(l in 1:L){
    f <- function(x){
      x.p = poly(x,2,raw=T)
      value = as.vector(x.p%*%B[,l+2]) - sum(B[c(2,5,9,14,20),l+2])
      return(value)
    }
    f.list[[l]] = f
  }
  
  ## no covariate shift
  set.seed(NULL)
  mu = rep(0,p); Sigma = diag(p)
  mu0 = mu; Sigma0 = Sigma
  ind.group = sample(1:L, size=n, replace=TRUE, prob=ratio.tr)
  ## covariates and outcome
  X = mvrnorm(n, mu, Sigma)
  Y = rep(NA, n)
  for(l in 1:L){
    map = (ind.group == l)
    Y[map] = f.list[[l]](X[map,]) + rnorm(sum(map))
  }
  
  ## fitting models
  model=1
  if(model==1){
    myrf = rf.funs()
    train.fun = myrf$train.fun
    pred.fun = myrf$pred.fun
  }
  if(model==2){
    mypoly = poly.funs()
    train.fun = mypoly$train.fun
    pred.fun = mypoly$pred.fun
  }
  
  fit.erm = train.fun(X,Y)
  fit.sep.list = rep(list(NA), L)
  for(l in 1:L){
    map.l = (ind.group==l)
    fit.sep.list[[l]] = train.fun(X[map.l,], Y[map.l])
  }
  
  ## generate X0 
  X0 = mvrnorm(n0, mu0, Sigma0)
  pred0.erm = pred.fun(fit.erm, X0)
  ## identify mixture weights
  pred0.mat = matrix(NA, nrow=n0, ncol=L)
  for(l in 1:L) pred0.mat[,l] = pred.fun(fit.sep.list[[l]], X0)
  
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
      ##### Gamma-bias value #####
      f.k = fit.sep.list[[k]]
      f.l = fit.sep.list[[l]]
      map.k = (ind.group == k)
      map.l = (ind.group == l)
      num1 = bias_correct(f.k, f.l, rep(1, sum(map.l)), X[map.l,], Y[map.l])
      num2 = bias_correct(f.l, f.k, rep(1, sum(map.k)), X[map.k,], Y[map.k])
      ##### hGamma #####
      hGamma[k,l] = hGamma[k,l] - (num1 + num2)
    }
  }
  eig.val = eigen(hGamma)$values
  eig.vec = eigen(hGamma)$vectors
  eig.val = pmax(eig.val, 1e-6)
  hGamma = eig.vec %*% diag(eig.val) %*% t(eig.vec)
  
  compute_v <- function(Gamma, const=L, v0){
    v = Variable(L)
    obj = quad_form(v, Gamma)
    constraints = list(v>=0, sum(v)==1, cvxr_norm(v-v0)/sqrt(L)<= const)
    prob = Problem(Minimize(obj), constraints)
    result = solve(prob)
    v.opt = result$getValue(v)
    reward = result$value
    return(list(v = v.opt,
                reward = reward))
  }
  
  consts.set = c(1, 0.3, 0.28, 0.25, 0.22, 0.2, 0.18, 0.15, 0.12, 0.1, 0.08, 0.05, 0.02, 0.01, 0)
  v.mag.mat = matrix(NA, nrow=length(consts.set), ncol=L)
  for(i.const in 1:length(consts.set)){
    out = compute_v(hGamma, const=consts.set[i.const], v0=ratio.tr)
    v.mag.mat[i.const,] = out$v
  }
  
  group.reward.mat = matrix(NA, nrow=L, ncol=1+length(consts.set))
  colnames(group.reward.mat) = c('erm', consts.set)
  for(l in 1:L){
    Y0 = rnorm(n0)
    Y0 = Y0 + f.list[[l]](X0)
    group.reward.mat[l,1] = mean(Y0^2 - (Y0 - pred0.erm)^2)
    for(i.const in 1:length(consts.set)){
      group.reward.mat[l,i.const+1] = mean(Y0^2 - (Y0 - pred0.mat%*%v.mag.mat[i.const,])^2)
    }
  }

  compute_worst_reward <- function(a, delta, v0){
    q = Variable(L)
    obj = sum(a * q)
    constraints = list(q>=0, sum(q)==1, cvxr_norm(q-v0)/sqrt(L)<= delta)
    prob = Problem(Minimize(obj), constraints)
    result = solve(prob)
    q.opt = result$getValue(q)
    return(list(q = q.opt,
                value = result$value))
  }
  
  consts.set.eval = c(1, seq(0.4, 0, by=-0.01))
  worst.reward = matrix(NA, nrow=length(consts.set.eval), ncol=1+length(consts.set))
  rownames(worst.reward) = consts.set.eval
  colnames(worst.reward) = c('erm', paste0('H',consts.set))
  for(i in 1:length(consts.set.eval)){
    delta = consts.set.eval[i]
    # print(i)
    worst.reward[i,1] = compute_worst_reward(group.reward.mat[,1], delta, v0=ratio.tr)$value
    for(j in 1:length(consts.set)){
      worst.reward[i, j+1] = compute_worst_reward(group.reward.mat[,j+1], delta, v0=ratio.tr)$value
    }
  }
  # 
  # x1 = 1:(length(consts.set.eval)-10)
  # y1 = worst.reward[-c(1,2:10),colnames(worst.reward)=='H0.25']
  # y2 = worst.reward[-c(1,2:10), 1]
  # y3 = worst.reward[-c(1,2:10), 2]
  # ylim1 = min(cbind(y1,y2,y3)); ylim2 = max(cbind(y1,y2,y3))
  # plot(x1, y1, type='l', ylim=c(ylim1, ylim2), col='red') # DRL-H
  # lines(x1, y2, col='blue') # erm
  # lines(x1, y3, col='green') # mag
  reward.cluster[[i.sim]] = worst.reward
}
data = list(reward = reward.cluster)
saveRDS(data, paste0('mixture_ratio-L4-magH-v3','-nper',n.per,'-sim.round',sim.round, '.rds'))

