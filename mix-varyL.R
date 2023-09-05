library(ranger)
library(stats)
library(MASS)
library(CVXR)
source('funs.R')

ratio.setting = 1 # {1,2}
n.per = 1000 #{1000,2000,4000}
L = 4 # {2,...,9}

print(paste0('mixture_ratio-varyL-nper',n.per,'-L',L,'-ratio',ratio.setting))
nsim = 500
reward.cluster = rep(list(NA), nsim)
for(i.sim in 1:nsim){
  print(paste0('simulation--->',i.sim))
  ## variables changed
  if(ratio.setting==1){
    ratio.tr = rep(1/L, L)
  }else if(ratio.setting==2){
    ratio.tr = c(0.55, rep(0.45/(L-1), (L-1)))
  }
  v0 = ratio.tr

  ## variables not changed
  p = 5
  n = n.per*L ## sample size for the whole data group
  n0 = 10000
  
  ## conditional outcomes
  B = readRDS('matrixB.rds')
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
  ind.AB = sample(1:2, size=n, replace = T)
  
  ## fitting models
  myrf = rf.funs()
  train.fun = myrf$train.fun
  pred.fun = myrf$pred.fun
  
  fit.erm = train.fun(X,Y)
  fit.sep.list = fit.sep.A.list = fit.sep.B.list = rep(list(NA), L)
  for(l in 1:L){
    map.l = (ind.group==l)
    fit.sep.list[[l]] = train.fun(X[map.l,], Y[map.l])
    map.lA = as.logical((ind.group==l)*(ind.AB == 1))
    map.lB = as.logical((ind.group==l)*(ind.AB == 2))
    fit.sep.A.list[[l]] = train.fun(X[map.lA,], Y[map.lA])
    fit.sep.B.list[[l]] = train.fun(X[map.lB,], Y[map.lB])
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
  
  hGamma.sp = hGamma.nosp = tGamma
  #### bias correction function ####
  bias_correct <- function(fk, fl, wl, Xl, Yl){
    nl = nrow(Xl)
    fkX = pred.fun(fk, Xl)
    flX = pred.fun(fl, Xl)
    return(mean(wl*fkX*(flX - Yl)))
  }
  for(k in 1:L){
    for(l in 1:L){
      ##### no split hGamma #####
      f.k = fit.sep.list[[k]]
      f.l = fit.sep.list[[l]]
      map.k = (ind.group == k)
      map.l = (ind.group == l)
      num1 = bias_correct(f.k, f.l, rep(1, sum(map.l)), X[map.l,], Y[map.l])
      num2 = bias_correct(f.l, f.k, rep(1, sum(map.k)), X[map.k,], Y[map.k])
      hGamma.nosp[k,l] = hGamma.nosp[k,l] - (num1 + num2)
      
      ##### split hGamma #####
      f.kA = fit.sep.A.list[[k]]; f.kB = fit.sep.B.list[[k]]
      f.lA = fit.sep.A.list[[l]]; f.lB = fit.sep.B.list[[l]]
      map.kA = as.logical((ind.group == k)*(ind.AB == 1))
      map.kB = as.logical((ind.group == k)*(ind.AB == 2))
      map.lA = as.logical((ind.group == l)*(ind.AB == 1))
      map.lB = as.logical((ind.group == l)*(ind.AB == 2))
      num1.A = bias_correct(f.kA, f.lA, rep(1, sum(map.lB)), X[map.lB,], Y[map.lB])
      num2.A = bias_correct(f.lA, f.kA, rep(1, sum(map.kB)), X[map.kB,], Y[map.kB])
      num1.B = bias_correct(f.kB, f.lB, rep(1, sum(map.lA)), X[map.lA,], Y[map.lA])
      num2.B = bias_correct(f.lB, f.kB, rep(1, sum(map.kA)), X[map.kA,], Y[map.kA])
      hGamma.sp[k,l] = hGamma.sp[k,l] - (num1.A + num2.A + num1.B + num2.B)/2
    }
  }
  eig.val = eigen(hGamma.nosp)$values
  eig.vec = eigen(hGamma.nosp)$vectors
  eig.val = pmax(eig.val, 1e-6)
  hGamma.nosp = eig.vec %*% diag(eig.val) %*% t(eig.vec)
  
  eig.val = eigen(hGamma.sp)$values
  eig.vec = eigen(hGamma.sp)$vectors
  eig.val = pmax(eig.val, 1e-6)
  hGamma.sp = eig.vec %*% diag(eig.val) %*% t(eig.vec)
  
  compute_v <- function(Gamma, const=L, v0= rep(1/L,L)){
    v = Variable(L)
    obj = quad_form(v, Gamma)
    constraints = list(v>=0, sum(v)==1, cvxr_norm(v-v0)/sqrt(L)<= const)
    prob = Problem(Minimize(obj), constraints)
    result = solve(prob)
    v.opt = result$getValue(v)
    return(v.opt)
  }
  
  w.plug = compute_v(tGamma, const=10)
  w.debias.nosp = compute_v(hGamma.nosp, const=10)
  w.debias.sp = compute_v(hGamma.sp, const=10)
  
  ## generate evaluation
  group.reward.mat = matrix(NA, nrow=L, ncol=4)
  colnames(group.reward.mat) = c('erm','plug','nosplit','split')
  for(l in 1:L){
    Y0 = rnorm(n0)
    Y0 = Y0 + f.list[[l]](X0)

    re.erm = mean(Y0^2 - (Y0 - pred0.erm)^2)
    re.plug = mean(Y0^2 - (Y0 - pred0.mat%*%w.plug)^2)
    re.debias.nosp = mean(Y0^2 - (Y0 - pred0.mat%*%w.debias.nosp)^2)
    re.debias.sp = mean(Y0^2 - (Y0 - pred0.mat%*%w.debias.sp)^2)
    group.reward.mat[l,] = c(re.erm, re.plug, re.debias.nosp, re.debias.sp)
  }
  
  reward.cluster[[i.sim]] = group.reward.mat
}
data = list(reward = reward.cluster)
# saveRDS(data, paste0('mixture_ratio-varyL-split-nper',n.per,'-L',L,'-ratio',ratio.setting,'.rds'))
# result = rep(0, 4)
# for(i.sim in 1:nsim){
#   result = result+apply(reward.cluster[[i.sim]], MARGIN=2, FUN=min)
# }
# result = result/10
# result


