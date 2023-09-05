library(ranger)
library(stats)
library(MASS)
library(CVXR)
library(mvtnorm)
library(glmnet)
source('funs.R')

sim.round = 1 #{1,2,3,4,5}
n.per = 200 # 200,300,400,600,800,1000
setting = 1 # 1 2
var.scale =  4 # 1,2,4

L = 5
p = 4

nsim = 100
dist.f.mat = matrix(NA, nrow=nsim, ncol=5); colnames(dist.f.mat) = c('plug','orac','nocs','log','rf')
v.mag.list = rep(list(NA), nsim)
delta_n.mat = matrix(NA, nrow=nsim, ncol=L)

## generate X0 
n0 = 1000
if(setting == 1){
  mu0 = rep(0,p)
  Sigma0 = diag(p) 
  X0 = mvrnorm(n0, mu0, Sigma0)
}
if(setting == 2){
  mu0 = c(0,-1,2,1)/4
  Sigma0 = diag(p) 
  X0 = mvrnorm(n0, mu0, Sigma0)
}
# set.seed(0)
# B.train = matrix(0, nrow=40, ncol=L)
# for(i.B in 1:40){
#   for(i.L in 1:L){
#     B.train[i.B, i.L] = 2*sample(c(2,0.4,0,-1),size=1, replace=F)
#   }
# }
B.train = readRDS('rf-mat.rds')*var.scale

set.seed(NULL)
for(i.sim in 1:nsim){
  set.seed(i.sim+(sim.round-1)*nsim)
  print(paste0('simulation--->',i.sim))
  ## variables changed
  ratio.tr =rep(1/L, L)
  v0 = ratio.tr
  
  ## variables not changed 
  n = n.per*L ## sample size for the whole data group
  f.list = rep(list(NA),L)
  for(l in 1:L){
    f <- function(x){
      n.x = nrow(x)
      value.vec = rep(NA, n.x)
      for(i.x in 1:n.x){
        value = 0
        idx = 0
        for(i.p in 1:p){
          idx =idx + 1
          value = value + B.train[idx, l]*(x[i.x, i.p]>0)
        }
        for(i.p1 in 1:p){
          for(i.p2 in i.p1:p){
            idx = idx+1
            value = value + B.train[idx,l]*(x[i.x, i.p1]>0)*(x[i.x, i.p2]>0)
          }
        }
        for(i.p1 in 1:p){
          for(i.p2 in 1:p){
            idx = idx+1
            value = value + B.train[idx,l]*(x[i.x, i.p1]<2)*(x[i.x, i.p2]>-2)
          }
        }
        value.vec[i.x] = value
      }
      return(value.vec)
    }
    f.list[[l]] = f
  }
  
  ## covariate shift
  set.seed(NULL)
  mu = rep(0,p); Sigma = 1*diag(p)
  ind.group = sample(1:L, size=n, replace=TRUE, prob=ratio.tr)
  ## covariates and outcome
  X = mvrnorm(n, mu, Sigma)
  
  Y = rep(NA, n)
  for(l in 1:L){
    map = (ind.group == l)
    Y[map] = f.list[[l]](X[map,]) + rnorm(sum(map))
    Y[map] = Y[map] - mean(Y[map])
  }
  
  ## fitting models
  model=1
  if(model==1){
    myrf = rf.funs()
    train.fun = myrf$train.fun
    pred.fun = myrf$pred.fun
    
    myrf.class = rf.class.funs()
    train.fun.class = myrf.class$train.fun
    pred.fun.class = myrf.class$pred.fun
  }
  
  fit.erm = train.fun(X,Y)
  fit.sep.list = rep(list(NA), L)
  for(l in 1:L){
    map.l = (ind.group==l)
    fit.sep.list[[l]] = train.fun(X[map.l,], Y[map.l])
  }
  
  pred0.erm = pred.fun(fit.erm, X0)
  ## identify mixture weights
  pred0.mat = matrix(NA, nrow=n0, ncol=L)
  for(l in 1:L) pred0.mat[,l] = pred.fun(fit.sep.list[[l]], X0)
  
  ## Magging
  tGamma = t(pred0.mat)%*%pred0.mat/n0
  
  ##### learn weights #####
  ## oracle ##
  w.oracle.list = rep(list(NA), L)
  for(l in 1:L){
    map.l = (ind.group == l)
    w.oracle.list[[l]] = dmvnorm(X[map.l,], mean=mu0, sigma=Sigma0)/dmvnorm(X[map.l,], mean=mu, sigma=Sigma)
  }
  ## logistic ##
  log_classifier <- function(X, X0, highd=F){
    n = nrow(X); n0 = nrow(X0)
    X.merge = rbind(X,X0)
    Y.merge = as.factor(c(rep(0,n), rep(1, n0)))
    data = data.frame(Y=Y.merge, X=X.merge)
    colnames(data) = c('Y',paste0('X',1:p))
    if(highd){
      fit.log = train.fun.log(X.merge, Y.merge)
      prob = pred.fun.log(fit.log, X)
    }else{
      model = glm(Y~., data=data, family='binomial')
      prob = predict(model, type='response')[1:n]
    }
    prob = pmin(prob, 1-1e-4)
    class.ratio = prob/(1-prob)
    return(class.ratio)
  }
  w.log.list = rep(list(NA),L)
  for(l in 1:L){
    map.l = (ind.group==l)
    w.log.list[[l]] = log_classifier(X[map.l,], X0) * nrow(X[map.l,]) / n0
  }
  ## random forest ##
  rf_classifier <- function(X, X0){
    n = nrow(X); n0 = nrow(X0)
    X.merge = rbind(X, X0)
    Y.merge = as.factor(c(rep(0,n), rep(1, n0)))
    data = data.frame(Y=Y.merge, X=X.merge)
    colnames(data) = c('y',paste0('x',1:p))
    rf.model = train.fun.class(X.merge, Y.merge)
    prob = pred.fun.class(rf.model, data[,-1])[1:n]
    # model = ranger(y~., data=data, probability = T)
    # prob = predict(model$forest, data=data[,-1])$predictions[1:n,2]
    # model = randomForest(Y~., data=data)
    # prob = predict(model, data=data, type='prob')[1:n,2]
    prob = pmin(prob,1-1e-4)
    class.ratio = prob/ (1-prob)
    return(class.ratio)
  }
  w.rf.list = rep(list(NA), L)
  for(l in 1:L){
    map.l = (ind.group==l)
    w.rf.list[[l]] = rf_classifier(X[map.l,], X0)*nrow(X[map.l,])/n0
  }
  hGamma.orac = hGamma.log = hGamma.rf = hGamma.nocs = tGamma
  
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
      ## nocs ##
      num1 = bias_correct(f.k, f.l, rep(1, sum(map.l)), X[map.l,], Y[map.l])
      num2 = bias_correct(f.l, f.k, rep(1, sum(map.k)), X[map.k,], Y[map.k])
      hGamma.nocs[k,l] = hGamma.nocs[k,l] - (num1 + num2)
      ## orac ##
      num1 = bias_correct(f.k, f.l, w.oracle.list[[l]], X[map.l,], Y[map.l])
      num2 = bias_correct(f.l, f.k, w.oracle.list[[k]], X[map.k,], Y[map.k])
      hGamma.orac[k,l] = hGamma.orac[k,l] - (num1 + num2)
      ## log ##
      num1 = bias_correct(f.k, f.l, w.log.list[[l]], X[map.l,], Y[map.l])
      num2 = bias_correct(f.l, f.k, w.log.list[[k]], X[map.k,], Y[map.k])
      hGamma.log[k,l] = hGamma.log[k,l] - (num1 + num2)
      ## rf ##
      num1 = bias_correct(f.k, f.l, w.rf.list[[l]], X[map.l,], Y[map.l])
      num2 = bias_correct(f.l, f.k, w.rf.list[[k]], X[map.k,], Y[map.k])
      hGamma.rf[k,l] = hGamma.rf[k,l] - (num1 + num2)
    }
  }
  psd.Gamma <- function(hGamma){
    eig.val = eigen(hGamma)$values
    eig.vec = eigen(hGamma)$vectors
    eig.val = pmax(eig.val, 1e-6)
    hGamma = eig.vec %*% diag(eig.val) %*% t(eig.vec)
    return(hGamma)
  }
  hGamma.nocs = psd.Gamma(hGamma.nocs)
  hGamma.orac = psd.Gamma(hGamma.orac)
  hGamma.log = psd.Gamma(hGamma.log)
  hGamma.rf = psd.Gamma(hGamma.rf)
  
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
  
  ## truth ##
  truth0.mat = matrix(NA, nrow=n0, ncol=L)
  for(l in 1:L){
    truth0.mat[,l] = f.list[[l]](X0)
    truth0.mat[,l] = truth0.mat[,l] - mean(truth0.mat[,l])
  }
  
  delta_n = apply(pred0.mat - truth0.mat, MARGIN=2, FUN=function(x) sqrt(mean(x^2)))
  delta_n.mat[i.sim,] = delta_n
  
  Gamma.truth = t(truth0.mat)%*%truth0.mat/n0
  
  ## aggregation weights ##
  v.mag.truth = compute_v(Gamma.truth, const=2, v0=ratio.tr)$v
  v.mag.plug = compute_v(tGamma, const=2, v0=ratio.tr)$v
  v.mag.orac = compute_v(hGamma.orac, const=2, v0=ratio.tr)$v
  v.mag.nocs = compute_v(hGamma.nocs, const=2, v0=ratio.tr)$v
  v.mag.log = compute_v(hGamma.log, const=2, v0=ratio.tr)$v
  v.mag.rf = compute_v(hGamma.rf, const=2, v0=ratio.tr)$v
  v.mag.mat = cbind(v.mag.truth, v.mag.plug,v.mag.orac, v.mag.nocs, v.mag.log, v.mag.rf)
  colnames(v.mag.mat) = c('truth','plug','orac','nocs','log','rf')
  v.mag.list[[i.sim]] = v.mag.mat
  
  ## distance of f level ##
  drl.truth = truth0.mat%*%v.mag.truth
  drl.plug = pred0.mat%*%v.mag.plug
  drl.orac = pred0.mat%*%v.mag.orac
  drl.nocs = pred0.mat%*%v.mag.nocs
  drl.log = pred0.mat%*%v.mag.log
  drl.rf = pred0.mat%*%v.mag.rf
  
  dist.orac = mean((drl.orac - drl.truth)^2)
  dist.plug = mean((drl.plug - drl.truth)^2)
  dist.nocs = mean((drl.nocs - drl.truth)^2)
  dist.log = mean((drl.log - drl.truth)^2)
  dist.rf = mean((drl.rf - drl.truth)^2)
  dist.f.mat[i.sim,] = c(dist.plug, dist.orac, dist.nocs, dist.log, dist.rf)
}

data = list(v.mag.list=v.mag.list, dist.f.mat=dist.f.mat, delta_n.mat=delta_n.mat)
# saveRDS(data, paste0('Improve-rf','-nper',n.per,'-setting',setting,'-vars',var.scale,'-sim.round',sim.round, '.rds'))
# v.diff = function(i.sim, data){
#   vec = rep(0,5)
#   for(i.k in 1:5){
#     k = c(2,3,4,5,6)[i.k]
#     vec[i.k] = sum((data$v.mag.list[[i.sim]][,k] - data$v.mag.list[[i.sim]][,1])^2)
#   }
#   return(vec)
# }
# v.diff.mat = matrix(NA, nrow=nsim, ncol=5)
# colnames(v.diff.mat) = c('plug','orac','nocs','log','rf')
# for(i.sim in 1:nsim){
#   v.diff.mat[i.sim,] = (v.diff(i.sim,data))
# }
# apply(v.diff.mat, MARGIN=2, mean)
# apply(sqrt(data$dist.f.mat), MARGIN=2, mean)
# delta_n.mat
# v.mag.truth
