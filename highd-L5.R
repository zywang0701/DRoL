library(CVXR); library(glmnet); library(MASS)
source('funs.R')

#### generate training data ####
p = 200; L = 5
sim.round = 1 #{1,2,3,4,5}
n.per = 200 # 50 100 150 200 250 300 400 600 800 1000 1200
type = 2
vol = 1 

print(paste0('highd-L5-V2-nper',n.per,'-type',type,'-vol',vol,'-sim.round',sim.round))
nsim = 100
re.mat = matrix(NA, nrow=nsim, ncol=4); colnames(re.mat) = c('erm','mm','mmde','oracle')
bdiff.mat = matrix(NA, nrow=nsim, ncol=3); colnames(bdiff.mat) = c('erm','mm','mmde')
weight.mm.mat = weight.mmde.mat= matrix(NA, nrow=nsim, ncol=L)

##################### All Groups Kept #######################
for(i.sim in 1:nsim){
  if(i.sim %% 1 == 0) cat(paste0('----->i.sim = ', i.sim, '/', nsim, '\n'))
  #### generate training data ####
  n = n.per * L
  
  B.train = readRDS('rand-B-new.rds')
  X.train = mvrnorm(n, rep(0,p), diag(p))
  Y.train = rep(0,n)
  ### the group labels 
  l.train.vec = sample(1:L, size=n, replace=TRUE, prob=rep(1/L, L))
  
  for(l in 1:L){
    map.l = (l.train.vec==l)
    Y.train[map.l] = X.train[map.l,] %*% B.train[,l]
  }
  Y.train = Y.train + rnorm(n)
  
  #### fit ####
  mylm = lmhigh.funs(intercept=TRUE, model='linear')
  train.fun = mylm$train.fun
  pred.fun = mylm$pred.fun
  fit.erm = train.fun(X.train, Y.train)
  b.erm = fit.erm$lasso.est
  fit.each.list = rep(list(NA), L)
  for(l in 1:L){
    map.l = (l.train.vec == l)
    X.l = X.train[map.l,] 
    Y.l = Y.train[map.l]
    fit.each.list[[l]] = train.fun(X.l, Y.l)
  }
  Bhat.mat = do.call(cbind, lapply(fit.each.list, FUN=function(x) x$lasso.est))
  
  #### generate X0 ####
  n0 = 10000
  X0 = mvrnorm(n0, mu=rep(0,p), Sigma=diag(p))
  
  #### identify mixture weights ####
  pred0.mat = matrix(0, nrow=n0, ncol=L)
  for(l in 1:L){
    pred0.mat[,l] = pred.fun(fit.each.list[[l]], X0)
  }
  ### plguin Magging ###
  tGamma = t(pred0.mat)%*%pred0.mat / n0
  v = Variable(L)
  obj = quad_form(v, tGamma)
  constraints = list(v>=0, sum(v)==1)
  prob = Problem(Minimize(obj), constraints)
  result = solve(prob)
  v.opt = result$getValue(v)
  ### Debias noCS Magging ###
  bias_correct <- function(fk, fl, wl, Xl, Yl){
    nl = nrow(Xl)
    fkX = pred.fun(fk, Xl)
    flX = pred.fun(fl, Xl)
    return(mean(wl*fkX*(flX - Yl)))
  }
  hGamma = tGamma
  for(k in 1:L){
    for(l in 1:L){
      ##### Gamma-bias value #####
      f.k = fit.each.list[[k]]
      f.l = fit.each.list[[l]]
      map.k = (l.train.vec == k)
      map.l = (l.train.vec == l)
      ## nocs ##
      num1 = bias_correct(f.k, f.l, rep(1, sum(map.l)), X.train[map.l,], Y.train[map.l])
      num2 = bias_correct(f.l, f.k, rep(1, sum(map.k)), X.train[map.k,], Y.train[map.k])
      hGamma[k,l] = hGamma[k,l] - (num1 + num2)
    }
  }
  eig.val = eigen(hGamma)$values
  eig.vec = eigen(hGamma)$vectors
  eig.val = pmax(eig.val, 1e-4)
  hGamma = eig.vec %*% diag(eig.val) %*% t(eig.vec)
  q = Variable(L)
  obj = quad_form(q, hGamma)
  constraints = list(q>=0, sum(q)==1)
  prob = Problem(Minimize(obj), constraints)
  result = solve(prob)
  q.opt = result$getValue(q)
  
  #### generate Y0 with random effects ####
  b0 = rep(0, p)
  b0[1:10] = 0.5
  b0[11:15] = rnorm(5, mean=0, sd=vol)
  Y0 = X0%*%b0 + rnorm(n0)
  
  #### oracle ####
  b.oracle = rep(0, p)
  b.oracle[1:10] = 0.5
  
  #### reward ####
  pred0.erm = pred.fun(fit.erm, X0)
  pred0.mm = as.vector(pred0.mat %*% v.opt)
  pred0.mmde = as.vector(pred0.mat %*% q.opt)
  re.erm = mean(Y0^2-(Y0 - pred0.erm)^2)
  re.mm = mean(Y0^2-(Y0 - pred0.mm)^2)
  re.mmde = mean(Y0^2-(Y0 - pred0.mmde)^2)
  re.oracle = mean(Y0^2-(Y0 - X0%*%b.oracle)^2)
  
  #### distance ####
  b.mm = as.vector(Bhat.mat%*%v.opt)
  b.mmde = as.vector(Bhat.mat%*%q.opt)
  bdiff.mat[i.sim, ] = c(sum((b.erm - c(0, b.oracle))^2),
                         sum((b.mm - c(0, b.oracle))^2),
                         sum((b.mmde - c(0, b.oracle))^2))
  
  weight.mm.mat[i.sim,] = v.opt
  weight.mmde.mat[i.sim,] = q.opt
  re.mat[i.sim, ] = c(re.erm, re.mm, re.mmde, re.oracle)
}

##################### Drop one group #######################
re.mat.1 = re.mat.2 = re.mat.3 = re.mat.4 = re.mat.5 = matrix(NA, nrow=nsim, ncol=4);
colnames(re.mat.1) = colnames(re.mat.2) = colnames(re.mat.3) = colnames(re.mat.4) = c('erm','mm','mmde','oracle')
bdiff.mat.1 = bdiff.mat.2 = bdiff.mat.3 = bdiff.mat.4 = bdiff.mat.5 = matrix(NA, nrow=nsim, ncol=3);
colnames(bdiff.mat.1) = colnames(bdiff.mat.2) = colnames(bdiff.mat.3) =
  colnames(bdiff.mat.4) = colnames(bdiff.mat.5) = c('erm','mm','mmde')
for(i.sim in 1:nsim){
  for(drop.l in 1:5){
    if(i.sim %% 1 == 0) cat(paste0('----->i.sim = ', i.sim, '/', nsim, 'drop.l = ',drop.l,'\n'))
    L = 4
    n = n.per * L
    B.train = readRDS('src/rand-B-new.rds')
    B.train = B.train[,-drop.l]

    X.train = mvrnorm(n, rep(0,p), diag(p))
    Y.train = rep(0,n)
    ### the group labels
    l.train.vec = sample(1:L, size=n, replace=TRUE, prob=rep(1/L, L))

    for(l in 1:L){
      map.l = (l.train.vec==l)
      Y.train[map.l] = X.train[map.l,] %*% B.train[,l]
    }
    Y.train = Y.train + rnorm(n)

    #### fit ####
    mylm = lmhigh.funs(intercept=TRUE, model='linear')
    train.fun = mylm$train.fun
    pred.fun = mylm$pred.fun
    fit.erm = train.fun(X.train, Y.train)
    b.erm = fit.erm$lasso.est
    fit.each.list = rep(list(NA), L)
    for(l in 1:L){
      map.l = (l.train.vec == l)
      X.l = X.train[map.l,]
      Y.l = Y.train[map.l]
      fit.each.list[[l]] = train.fun(X.l, Y.l)
    }
    Bhat.mat = do.call(cbind, lapply(fit.each.list, FUN=function(x) x$lasso.est))

    #### generate X0 ####
    n0 = 10000
    X0 = mvrnorm(n0, mu=rep(0,p), Sigma=diag(p))

    #### identify mixture weights ####
    pred0.mat = matrix(0, nrow=n0, ncol=L)
    for(l in 1:L){
      pred0.mat[,l] = pred.fun(fit.each.list[[l]], X0)
    }
    ### plguin Magging ###
    tGamma = t(pred0.mat)%*%pred0.mat / n0
    v = Variable(L)
    obj = quad_form(v, tGamma)
    constraints = list(v>=0, sum(v)==1)
    prob = Problem(Minimize(obj), constraints)
    result = solve(prob)
    v.opt = result$getValue(v)
    ### Debias noCS Magging ###
    bias_correct <- function(fk, fl, wl, Xl, Yl){
      nl = nrow(Xl)
      fkX = pred.fun(fk, Xl)
      flX = pred.fun(fl, Xl)
      return(mean(wl*fkX*(flX - Yl)))
    }
    hGamma = tGamma
    for(k in 1:L){
      for(l in 1:L){
        ##### Gamma-bias value #####
        f.k = fit.each.list[[k]]
        f.l = fit.each.list[[l]]
        map.k = (l.train.vec == k)
        map.l = (l.train.vec == l)
        ## nocs ##
        num1 = bias_correct(f.k, f.l, rep(1, sum(map.l)), X.train[map.l,], Y.train[map.l])
        num2 = bias_correct(f.l, f.k, rep(1, sum(map.k)), X.train[map.k,], Y.train[map.k])
        hGamma[k,l] = hGamma[k,l] - (num1 + num2)
      }
    }
    eig.val = eigen(hGamma)$values
    eig.vec = eigen(hGamma)$vectors
    eig.val = pmax(eig.val, 1e-4)
    hGamma = eig.vec %*% diag(eig.val) %*% t(eig.vec)
    q = Variable(L)
    obj = quad_form(q, hGamma)
    constraints = list(q>=0, sum(q)==1)
    prob = Problem(Minimize(obj), constraints)
    result = solve(prob)
    q.opt = result$getValue(q)

    #### generate Y0 with random effects ####
    b0 = rep(0, p)
    b0[1:10] = 0.5
    b0[11:15] = rnorm(5, mean=0, sd=vol)
    Y0 = X0%*%b0 + rnorm(n0)

    #### oracle ####
    b.oracle = rep(0, p)
    b.oracle[1:10] = 0.5

    #### reward ####
    pred0.erm = pred.fun(fit.erm, X0)
    pred0.mm = as.vector(pred0.mat %*% v.opt)
    pred0.mmde = as.vector(pred0.mat %*% q.opt)
    re.erm = mean(Y0^2-(Y0 - pred0.erm)^2)
    re.mm = mean(Y0^2-(Y0 - pred0.mm)^2)
    re.mmde = mean(Y0^2-(Y0 - pred0.mmde)^2)
    re.oracle = mean(Y0^2-(Y0 - X0%*%b.oracle)^2)

    #### distance ####
    b.mm = as.vector(Bhat.mat%*%v.opt)
    b.mmde = as.vector(Bhat.mat%*%q.opt)
    bdiff.vec = c(sum((b.erm - c(0, b.oracle))^2),
                  sum((b.mm - c(0, b.oracle))^2),
                  sum((b.mmde - c(0, b.oracle))^2))
    re.vec = c(re.erm, re.mm, re.mmde, re.oracle)
    if(drop.l == 1){
      bdiff.mat.1[i.sim, ] = bdiff.vec
      re.mat.1[i.sim, ] = re.vec
    }
    if(drop.l == 2){
      bdiff.mat.2[i.sim, ] = bdiff.vec
      re.mat.2[i.sim, ] = re.vec
    }
    if(drop.l == 3){
      bdiff.mat.3[i.sim, ] = bdiff.vec
      re.mat.3[i.sim, ] = re.vec
    }
    if(drop.l == 4){
      bdiff.mat.4[i.sim, ] = bdiff.vec
      re.mat.4[i.sim, ] = re.vec
    }
    if(drop.l == 5){
      bdiff.mat.5[i.sim, ] = bdiff.vec
      re.mat.5[i.sim, ] = re.vec
    }
  }
}
bdiff.dropone = list(bdiff.mat.1, bdiff.mat.2, bdiff.mat.3, bdiff.mat.4, bdiff.mat.5)
re.mat.dropone = list(re.mat.1, re.mat.2, re.mat.3, re.mat.4, re.mat.5)

data = list(bdiff = bdiff.mat,
            w.mm = weight.mm.mat,
            w.mmde = weight.mmde.mat,
            re = re.mat,
            bdiff.dropone = bdiff.dropone,
            re.mat.dropone = re.mat.dropone)
# filename = paste0('highd-L5-V2-nper',n.per,'-sim.round',sim.round,'.rds')
# saveRDS(data, file=filename)

