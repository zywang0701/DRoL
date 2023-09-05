source('funs.R')
library(CVXR)
library(glmnet)
library(ranger)
data.all = readRDS("allgroups_split.rds")

################# Results with Y centralized ###################
myrf = rf.funs()
train.fun = myrf$train.fun
pred.fun = myrf$pred.fun
var.x = c('TEMP','PRES','DEWP','RAIN','wd','WSPM')
eval.seasons = c(13, 14, 15, 16)

# group = 3
# eval.season = 13

main_fun <- function(group, eval.season){
  train.seasons = sort(eval.season - c(4, 8, 12))
  L = length(train.seasons)
  X.pool = data.frame(); Y.pool = c()
  fit.sep = rep(list(NA), L)
  X.sep = Y.sep = rep(list(NA), L)
  for(l in 1:L){
    train.season = train.seasons[l]
    df = data.all[[group]][[train.season]]
    X = df[,colnames(df)%in%var.x]
    Y = log(df$PM2.5)
    Y = scale(Y, scale=F)
    fit = train.fun(X, Y, verbose = F)
    fit.sep[[l]] = fit
    X.pool = rbind(X.pool, X); Y.pool = c(Y.pool, Y)
    X.sep[[l]] = X; Y.sep[[l]] = Y
  }
  fit.erm = train.fun(X.pool, Y.pool, verbose=F)
  
  ### eval steps ###
  df0 = data.all[[group]][[eval.season]]
  X0 = df0[, colnames(df0)%in%var.x]
  Y0 = log(df0$PM2.5)
  Y0 = scale(Y0, scale=F)
  pred.mat = matrix(NA, nrow=length(Y0), ncol=L)
  re.sep = rep(NA, L)
  for(l in 1:L){
    pred.mat[,l] = pred.fun(fit.sep[[l]], X0)
    re.sep[l] = mean(Y0^2 - (Y0 - pred.mat[,l])^2)
  }
  
  ### plugin estimator ###
  tGamma = t(pred.mat)%*%pred.mat / length(Y0)
  
  compute_v <- function(Gamma, const=L, v0 = rep(1/L,L)){
    v = Variable(L)
    obj = quad_form(v, Gamma)
    constraints = list(v>=0, sum(v)==1, cvxr_norm(v - v0)/sqrt(L)<= const)
    prob = Problem(Minimize(obj), constraints)
    result = solve(prob)
    v.opt = result$getValue(v)
    return(v.opt)
  }
  q.opt = compute_v(tGamma, const = 2)
  pred.mm = pred.mat%*%q.opt; re.mm = mean(Y0^2 - (Y0 - pred.mm)^2)
  pred.erm = pred.fun(fit.erm, X0); re.erm = mean(Y0^2 - (Y0 - pred.erm)^2)
  
  ### debiased estimator - noCS / logistic###
  bias_correct <- function(fk, fl, wl, Xl, Yl){
    # print(paste0('length w', length(wl), 'vs length Yl', length(Yl)))
    nl = nrow(Xl)
    fkX = pred.fun(fk, Xl)
    flX = pred.fun(fl, Xl)
    return(mean(wl*fkX*(flX - Yl)))
  }
  log_classifier <- function(X, X0){
    n = nrow(X); n0 = nrow(X0)
    X.merge = rbind(X, X0)
    Y.merge = as.factor(c(rep(0,n), rep(1, n0)))
    data = data.frame(Y=Y.merge, X=X.merge)
    colnames(data) = c('Y',paste0('X',1:ncol(X)))
    model = glm(Y~., data=data, family='binomial')
    prob = predict(model, type='response')[1:n]
    prob = pmin(prob,0.999)
    class.ratio = prob / (1-prob)
    return(class.ratio)
  }
  w.log.list = rep(list(NA), L)
  for(l in 1:L){
    w.log.list[[l]] = log_classifier(X.sep[[l]], X0) * nrow(X.sep[[l]])/nrow(X0)
  }
  hGamma.nocs = tGamma
  hGamma.log = tGamma
  for(k in 1:L){
    for(l in 1:L){
      ##### Gamma-A #####
      f.k = fit.sep[[k]]
      f.l = fit.sep[[l]]
      num1 = bias_correct(f.k, f.l, 1, X.sep[[l]], Y.sep[[l]])
      num2= bias_correct(f.l, f.k, 1, X.sep[[k]], Y.sep[[k]])
      num1.log = bias_correct(f.k, f.l, w.log.list[[l]], X.sep[[l]], Y.sep[[l]])
      num2.log = bias_correct(f.l, f.k, w.log.list[[k]], X.sep[[k]], Y.sep[[k]])
      hGamma.nocs[k,l] = hGamma.nocs[k,l] - (num1 + num2)
      hGamma.log[k,l] = hGamma.log[k,l] - (num1.log + num2.log)
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
  hGamma.log = psd.Gamma(hGamma.log)
  v.opt.nocs = compute_v(hGamma.nocs, const = 2)
  v.opt.log = compute_v(hGamma.log, const=2)
  pred.mm.de.nocs = pred.mat%*%v.opt.nocs; re.mm.de.nocs = mean(Y0^2 - (Y0 - pred.mm.de.nocs)^2)
  pred.mm.de.log = pred.mat%*%v.opt.log; re.mm.de.log = mean(Y0^2 - (Y0 - pred.mm.de.log)^2)
  
  return(list(re.erm = re.erm, 
              re.mm = re.mm, re.mm.de.nocs = re.mm.de.nocs, re.mm.de.log = re.mm.de.log,
              re.sep = re.sep,
              w.mm = as.vector(q.opt), 
              w.mm.de.nocs = as.vector(v.opt.nocs), w.mm.de.log = as.vector(v.opt.log),
              tGamma = tGamma,
              hGamma.nocs = hGamma.nocs,
              hGamma.log = hGamma.log,
              pred.mat = pred.mat,
              Y0 = Y0))
}

train.groups = c(3,7,4)#c(7, 2, 6, 10, 9)
re.mat = matrix(NA, nrow=length(train.groups)*4, ncol=2+4+3)
colnames(re.mat) = c('group', 'season', 'erm', 'mm','mm.de.nocs','mm.de.log','sep13','sep14','sep15')
weight.mat = matrix(NA, nrow=length(train.groups)*4, ncol=2+3)
weight.denocs.mat = matrix(NA, nrow=length(train.groups)*4, ncol=2+3)
weight.delog.mat = matrix(NA, nrow=length(train.groups)*4, ncol=2+3)
tGamma.list = hGamma.nocs.list = hGamma.log.list = pred.mat.list = Y0.list = rep(list(NA), length(train.groups))

for(i.group in 1:length(train.groups)){
  tGamma.list[[i.group]] = hGamma.nocs.list[[i.group]] = 
    hGamma.log.list[[i.group]] = pred.mat.list[[i.group]] = Y0.list[[i.group]] = rep(list(NA), 4)
  for(i.season in 1:4){
    eval.season = eval.seasons[i.season]
    group = train.groups[i.group]
    print(paste0('group=',group,'; season', eval.season))
    out = main_fun(group, eval.season)
    idx = (i.group-1)*4 + i.season
    re.mat[idx,] = c(group, eval.season, out$re.erm, out$re.mm, out$re.mm.de.nocs, out$re.mm.de.log, out$re.sep)
    weight.mat[idx,] = c(group, eval.season, out$w.mm)
    weight.denocs.mat[idx,] = c(group, eval.season, out$w.mm.de.nocs)
    weight.delog.mat[idx,] = c(group, eval.season, out$w.mm.de.log)
    tGamma.list[[i.group]][[i.season]] = out$tGamma
    hGamma.nocs.list[[i.group]][[i.season]] = out$hGamma.nocs
    hGamma.log.list[[i.group]][[i.season]] = out$hGamma.log
    pred.mat.list[[i.group]][[i.season]] = out$pred.mat
    Y0.list[[i.group]][[i.season]] = out$Y0
  }
}

############# reward under multiple input H ##############
compute_v <- function(Gamma, const=L, v0 = rep(1/L,L)){
  v = Variable(L)
  obj = quad_form(v, Gamma)
  constraints = list(v>=0, sum(v)==1, cvxr_norm(v - v0)/sqrt(L)<= const)
  prob = Problem(Minimize(obj), constraints)
  result = solve(prob)
  v.opt = result$getValue(v)
  return(v.opt)
}

library(ggplot2)
library(reshape2)
library(ggpubr)

seasonnames = c("Spring", "Summer", "Autumn", "Winter")
##### Plot 1 #####
v0 = c(0.15, 0.35, 0.5)
consts.set = c(0.4, 0.3, 0.2, 0.1, 0.08, 0.05, 0.02, 0.01)
re.plug = re.deb = matrix(NA, nrow=length(train.groups)*4, ncol=2+length(consts.set))
L=3
for(i.group in 1:length(train.groups)){
  for(i.season in 1:4){
    group = train.groups[i.group]
    eval.season = eval.seasons[i.season]
    ## Gamma matrix
    tGamma = tGamma.list[[i.group]][[i.season]]
    hGamma = hGamma.log.list[[i.group]][[i.season]]
    ## pred.mat
    pred.mat = pred.mat.list[[i.group]][[i.season]]
    Y0 = Y0.list[[i.group]][[i.season]]
    ## weight
    idx = (i.group-1)*4 + i.season
    re.plug[idx,1:2] = re.deb[idx,1:2] = c(group, eval.season)
    for(i.const in 1:length(consts.set)){
      w.plug = compute_v(tGamma, const=consts.set[i.const], v0=v0)
      w.deb = compute_v(hGamma, const=consts.set[i.const], v0=v0)
      
      pred.mm.plug = pred.mat%*%w.plug; 
      pred.mm.deb = pred.mat%*%w.deb;
      re.plug[idx,i.const+2] = mean(Y0^2 - (Y0 - pred.mm.plug)^2)
      re.deb[idx,i.const+2] = mean(Y0^2 - (Y0 - pred.mm.deb)^2)
    }
  }
}

figs.list = rep(list(NA), 4)
for(i.season in 1:4){
  re.plot = data.frame(delta=consts.set, deb=re.deb[i.season,-c(1,2)], 
                       erm=rep(re.mat[i.season,3], length(consts.set)),
                       plug=re.plug[i.season,-c(1,2)])
  re.plot[,1] = factor(re.plot[,1], levels = sort(consts.set, decreasing=T)) #as.factor(re.plot[,1])
  colnames(re.plot) = c('delta','DRL','ERM','DRL-plug')
  re.plot.long = melt(re.plot, id='delta')
  
  figs.list[[i.season]] = ggplot(re.plot.long, aes(x=delta, y=value, colour=variable))+
    geom_line(linewidth=0.75, aes(group=variable))+
    geom_point(size=2,aes(shape=variable))+
    labs(x='Prior Info H: rho',title=seasonnames[i.season])+
    # scale_x_continuous(breaks=2:12)+
    scale_color_manual(values = c('#08519C','#DE2D26','#31A354'))+
    scale_shape_manual(values=c(16,15,17))+
    theme_light()+
    theme(plot.title = element_text(hjust=0.5, size=11,face = 'bold'),
          axis.text.x = element_text(size=9, angle = 90),
          #axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          legend.title = element_blank(),
          legend.text = element_text(size=10), legend.position = 'bottom')
}
p1 = ggarrange(figs.list[[1]], figs.list[[2]], figs.list[[3]], figs.list[[4]], ncol=4,
               common.legend = T, legend = 'none')
p1an = annotate_figure(p1, top = text_grob('Reward Evaluated at Year 2016 for Site "Dingling" with v0=(0.15,0.35,0.5)', color = '#08519C', face = "bold", size = 12))

##### Plot 2 #####
v0 = c(0.35, 0.15, 0.5)
consts.set = c(0.4, 0.3, 0.2, 0.1, 0.08, 0.05, 0.02, 0.01)
re.plug = re.deb = matrix(NA, nrow=length(train.groups)*4, ncol=2+length(consts.set))
L=3
for(i.group in 1:length(train.groups)){
  for(i.season in 1:4){
    group = train.groups[i.group]
    eval.season = eval.seasons[i.season]
    ## Gamma matrix
    tGamma = tGamma.list[[i.group]][[i.season]]
    hGamma = hGamma.log.list[[i.group]][[i.season]]
    ## pred.mat
    pred.mat = pred.mat.list[[i.group]][[i.season]]
    Y0 = Y0.list[[i.group]][[i.season]]
    ## weight
    idx = (i.group-1)*4 + i.season
    re.plug[idx,1:2] = re.deb[idx,1:2] = c(group, eval.season)
    for(i.const in 1:length(consts.set)){
      w.plug = compute_v(tGamma, const=consts.set[i.const], v0=v0)
      w.deb = compute_v(hGamma, const=consts.set[i.const], v0=v0)
      
      pred.mm.plug = pred.mat%*%w.plug; 
      pred.mm.deb = pred.mat%*%w.deb;
      re.plug[idx,i.const+2] = mean(Y0^2 - (Y0 - pred.mm.plug)^2)
      re.deb[idx,i.const+2] = mean(Y0^2 - (Y0 - pred.mm.deb)^2)
    }
  }
}

figs.list = rep(list(NA), 4)
for(i.season in 1:4){
  re.plot = data.frame(delta=consts.set, deb=re.deb[i.season,-c(1,2)], 
                       erm=rep(re.mat[i.season,3], length(consts.set)),
                       plug=re.plug[i.season,-c(1,2)])
  re.plot[,1] = factor(re.plot[,1], levels = sort(consts.set, decreasing=T)) #as.factor(re.plot[,1])
  colnames(re.plot) = c('delta','DRL','ERM','DRL-plug')
  re.plot.long = melt(re.plot, id='delta')
  
  figs.list[[i.season]] = ggplot(re.plot.long, aes(x=delta, y=value, colour=variable))+
    geom_line(linewidth=0.75, aes(group=variable))+
    geom_point(size=2,aes(shape=variable))+
    labs(x='Prior Info H: rho',title=seasonnames[i.season])+
    # scale_x_continuous(breaks=2:12)+
    scale_color_manual(values = c('#08519C','#DE2D26','#31A354'))+
    scale_shape_manual(values=c(16,15,17))+
    theme_light()+
    theme(plot.title = element_text(hjust=0.5, size=11,face = 'bold'),
          axis.text.x = element_text(size=9, angle = 90),
          #axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          legend.title = element_blank(),
          legend.text = element_text(size=10), legend.position = 'bottom')
}
p2 = ggarrange(figs.list[[1]], figs.list[[2]], figs.list[[3]], figs.list[[4]], ncol=4,
               common.legend = T, legend = 'bottom')
p2an = annotate_figure(p2, top = text_grob('Reward Evaluated at Year 2016 for Site "Dingling" with v0=(0.35, 0.15, 0.5)', color = '#08519C', face = "bold", size = 12))

ggarrange(p1an, p2an, nrow=2, heights = c(0.45, 0.55))
