lm.funs <- function(intercept=TRUE){
  train.fun = function(X,y){
    fit = lm(y~X)
    if(intercept==FALSE){
      theta = coef(fit)[-1]
    }else{
      theta = coef(fit)
    }
    return(list(theta = theta))
  }
  pred.fun = function(out, Xnew){
    if(intercept){
      return(as.vector(Xnew %*% out$theta[-1] + out$theta[1]))
    }else{
      return(as.vector(Xnew %*% out$theta))
    }
  }
  return(list(train.fun = train.fun,
              pred.fun = pred.fun))
}

poly.funs <- function(intercept=TRUE){
  train.fun = function(X,y){
    fit = lm(y~poly(X, degree=2, raw=T))
    if(intercept==FALSE){
      theta = coef(fit)[-1]
    }else{
      theta = coef(fit)
    }
    return(list(theta = theta))
  }
  pred.fun = function(out, Xnew){
    if(intercept){
      return(as.vector(poly(Xnew, degree=2, raw=T) %*% out$theta[-1] + out$theta[1]))
    }else{
      return(as.vector(poly(Xnew, degree=2, raw=T) %*% out$theta))
    }
  }
  return(list(train.fun = train.fun,
              pred.fun = pred.fun))
}

lmhigh.funs <- function(intercept=TRUE, model=c("linear","logistic")){
  model = match.arg(model)
  if(model=="linear"){
    train.fun <- function(X, y,lambda=NULL){
      if(is.null(lambda)) lambda = "CV.min"
      p = ncol(X)
      htheta <- if (lambda == "CV.min") {
        outLas <- cv.glmnet(X, y, family = "gaussian", alpha = 1,
                            intercept = intercept, standardize = T)
        as.vector(coef(outLas, s = outLas$lambda.min))
      } else if (lambda == "CV") {
        outLas <- cv.glmnet(X, y, family = "gaussian", alpha = 1,
                            intercept = intercept, standardize = T)
        as.vector(coef(outLas, s = outLas$lambda.1se))
      } else {
        outLas <- glmnet(X, y, family = "gaussian", alpha = 1,
                         intercept = intercept, standardize = T)
        as.vector(coef(outLas, s = lambda))
      }
      if(intercept==FALSE) htheta = htheta[2:(p+1)]
      
      return(list(lasso.est = htheta))
    }
    pred.fun = function(out, Xnew, intercept=TRUE){
      if(intercept){
        return(as.vector(Xnew%*%out$lasso.est[-1] + out$lasso.est[1]))
      }else{
        return(as.vector(Xnew%*%out$lasso.est))
      }
    }
  }else{
    train.fun <- function(X, y, lambda=NULL){
      if(is.null(lambda)) lambda = "CV.min"
      p = ncol(X)
      htheta <- if (lambda == "CV.min") {
        outLas <- cv.glmnet(X, y, family = "binomial", alpha = 1,
                            intercept = intercept, standardize = T)
        as.vector(coef(outLas, s = outLas$lambda.min))
      } else if (lambda == "CV") {
        outLas <- cv.glmnet(X, y, family = "binomial", alpha = 1,
                            intercept = intercept, standardize = T)
        as.vector(coef(outLas, s = outLas$lambda.1se))
      } else {
        outLas <- glmnet(X, y, family = "binomial", alpha = 1,
                         intercept = intercept, standardize = T)
        as.vector(coef(outLas, s = lambda))
      }
      if(intercept==FALSE) htheta = htheta[2:(p+1)]
      
      return(list(lasso.est = htheta))
    }
    pred.fun = function(out, Xnew){
      if(intercept){
        xval = as.vector(cbind(1,Xnew)%*%out$lasso.est)
      }else{
        xval = as.vector(Xnew%*%out$lasso.est)
      }
      return(exp(xval)/(1+exp(xval)))
    }
  }
  return(list(train.fun = train.fun,
              pred.fun = pred.fun))
}

rf.funs <- function(){
  ### training function ###
  train.fun = function(X, y, num.trees=200, mtry=NULL, max.depth=NULL, verbose=F){
    p = ncol(X)
    Data = data.frame(cbind(y, X))
    colnames(Data) = c('y', paste0('x', 1:p))
    n = nrow(X); p = ncol(X)
    if(is.null(mtry)) mtry = 2:p
    if(is.null(max.depth)) max.depth = c(0,3,5,8,10)
    
    # search grid
    params.grid = expand.grid(
      num.trees = num.trees,
      mtry = mtry,
      max.depth = max.depth
    )
    
    # use oob error to hyper-param tuning
    forest = NULL
    MSE.oob = Inf
    params = NULL
    for(i in 1:nrow(params.grid)){
      if(verbose) cat(sprintf('Training at the param ---> %s / %s.\n', i, nrow(params.grid)))
      temp.forest = ranger(y~., data=Data,
                           num.trees = params.grid$num.trees[i],
                           mtry = params.grid$mtry[i],
                           max.depth = params.grid$max.depth[i])
      if(temp.forest$prediction.error <= MSE.oob){
        forest = temp.forest
        params = params.grid[i,]
        MSE.oob = temp.forest$prediction.error
      }
    }
    out = list(forest = forest,
               params = params,
               MSE.oob = MSE.oob)
    return(out)
  }
  
  ### prediction function ###
  pred.fun = function(out, Xnew){
    if(is.vector(Xnew)) Xnew = t(as.matrix(Xnew))
    p = ncol(Xnew)
    colnames(Xnew) = paste0('x', 1:p)
    pred = predict(out$forest, data=Xnew, type='response')$predictions
    return(pred)
  }
  
  return(list(train.fun=train.fun,
              pred.fun =pred.fun))
}

rf.class.funs <- function(){
  ### training function ###
  train.fun = function(X, y, num.trees=200, mtry=NULL, max.depth=NULL, verbose=F){
    p = ncol(X)
    Data = data.frame(cbind(y, X))
    colnames(Data) = c('y', paste0('x', 1:p))
    n = nrow(X); p = ncol(X)
    if(is.null(mtry)) mtry = 1:p
    if(is.null(max.depth)) max.depth = c(0,3,5,8,10)
    
    # search grid
    params.grid = expand.grid(
      num.trees = num.trees,
      mtry = mtry,
      max.depth = max.depth
    )
    
    # use oob error to hyper-param tuning
    forest = NULL
    MSE.oob = Inf
    params = NULL
    for(i in 1:nrow(params.grid)){
      if(verbose) cat(sprintf('Training at the param ---> %s / %s.\n', i, nrow(params.grid)))
      temp.forest = ranger(y~., data=Data,
                           num.trees = params.grid$num.trees[i],
                           mtry = params.grid$mtry[i],
                           max.depth = params.grid$max.depth[i],
                           probability = T)
      if(temp.forest$prediction.error <= MSE.oob){
        forest = temp.forest
        params = params.grid[i,]
        MSE.oob = temp.forest$prediction.error
      }
    }
    out = list(forest = forest,
               params = params,
               MSE.oob = MSE.oob)
    return(out)
  }
  
  ### prediction function ###
  pred.fun = function(out, Xnew){
    if(is.vector(Xnew)) Xnew = t(as.matrix(Xnew))
    p = ncol(Xnew)
    colnames(Xnew) = paste0('x', 1:p)
    pred = predict(out$forest, data=Xnew, type='response')$predictions[,2]
    return(pred)
  }
  
  return(list(train.fun=train.fun,
              pred.fun =pred.fun))
}

