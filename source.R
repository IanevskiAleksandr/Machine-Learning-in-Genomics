# Machine-Learning-in-Genomics

setwd("/home/aianevsk/Desktop/ML in Genomics")

# load data
genotypeDT <- data.table::fread("genotype_data.csv")[-1:-4,];
phenotypeDT <- data.table::fread("phenotypes_s1.csv"); 

# convert to factor
genotypeDT[, names(genotypeDT) := lapply(.SD, as.factor)]; 

### perform Multiple Correspondence Analysis 
## aaa = FactoMineR::MCA(phenotypeDTtrain)

# convert factor to numeric (so we can feed them into models)
genotypeDT[, names(genotypeDT) := lapply(.SD, as.numeric)]; phenotypeDT[, names(phenotypeDT) := lapply(.SD, as.numeric)];
genotypeDT = as.data.frame(genotypeDT); phenotypeDT = as.data.frame(phenotypeDT); 

# merge to full set
fullSet = cbind.data.frame(phenotypeDT, genotypeDT)

# split into train and test (2/3 train, 1/3 test)
train_ind <- sample(seq_len(nrow(fullSet)), size = floor(2/3 * nrow(fullSet)))
phenotypeDTtrain = phenotypeDT[train_ind,]; phenotypeDTtest = phenotypeDT[-train_ind,]
genotypeDTtrain = genotypeDT[train_ind,]; genotypeDTtest = genotypeDT[-train_ind,]




# install keras, IntegratedMRF
install.packages(c("IntegratedMRF", "tensorflow")); devtools::install_github("rstudio/keras"); 
# Load packages
lapply(c("tensorflow", "keras", "IntegratedMRF", "parallel", "mlrMBO"), library, character.only = T)
# set up tensorflow
tensorflow::use_python("/home/aianevsk/.virtualenvs/envname/bin/python3.5"); 
tensorflow::use_virtualenv("/home/aianevsk/.virtualenvs/envname")
keras::use_python("/home/aianevsk/.virtualenvs/envname/bin/python3.5"); 
keras::use_virtualenv("/home/aianevsk/.virtualenvs/envname")




# # #####################################################################
# # ############ Multiple univariate linear regression models (MULRM)
# 
# predMURM_ =  sapply(1:ncol(phenotypeDTtrain), function(c_){
# 
#   # data
#   data = cbind.data.frame(genotypeDTtrain, phenotypeDTtrain[,c_]); colnames(data)[length(data)] = "V1";
# 
#   # train MMLR on 1 dependent variable
#   MLRmodel <- lm(V1 ~ ., data)
# 
#   # predict on test set
#   predict(MLRmodel, genotypeDTtest)
# })
# 
# # calculate test error
# multivariateLossMULRM = sapply(1:nrow(predMURM_), function(r_)
#   sum(1/2 * (as.numeric(phenotypeDTtest[r_,]) - as.numeric(predMURM_[r_,]))**2)
#)


##################################################################################################
######## Multivariate multiple regression

# train MMLR on 1 batch
MLRmodel <- lm(formula(paste('cbind(',
                             paste(colnames(genotypeDTtrain), collapse = ','),
                             ') ~ ',
                             ".")), data = as.data.frame(cbind.data.frame(phenotypeDTtrain, genotypeDTtrain)))

# predict on test set

predMMR_ = predict(MLRmodel, genotypeDTtest)

# calculate test error
multivariateLossMMR = sqrt(sapply(1:nrow(MMR_), function(r_)
  sum((as.numeric(phenotypeDTtest[r_,]) - as.numeric(MMR_[r_,]))**2)
)  / ncol(MMR_))



####################################
### Multivariate Random forest



obj.fun = makeSingleObjectiveFunction(
  name = "RF",
  fn = function(x) {
    
    # parameter set
    logNtree = x[1]; mtry = x[2]; nodesize = x[3]; 
    
    # repeated CV
    MAD_ <- mclapply(1:2, function(repCv){
      
      flds <- caret::createFolds(1:nrow(genotypeDTtrain), k = 3, list = !0, returnTrain = !1); MAD_i <- 0; 
      
      for(k in 1:length(flds)){
        validData <- genotypeDTtrain[flds[[k]], ]; validResp <- phenotypeDTtrain[flds[[k]], ]; 
        trainData <- genotypeDTtrain[-flds[[k]], ]; trainResp <- phenotypeDTtrain[-flds[[k]], ]
        
        fit =  randomForestSRC::rfsrc(formula(paste('cbind(',
                                                    paste(colnames(trainResp), collapse = ','),
                                                    ') ~ ',
                                                    ".")), data =  cbind.data.frame(trainData, trainResp),  
                                      ntree = round(2**logNtree)[[1]], mtry = mtry[[1]], nodesize = nodesize[[1]])
        
        # extract prediction results
        ypred = as.data.frame(lapply(predict(fit, validData)$regrOutput, function(x) x$predicted))
        
        # multivariate loss
        MAD_i <- MAD_i + sum(abs((ypred - validResp)))/(ncol(ypred) * nrow(ypred))
      }
      MAD_i
    }, mc.cores = 2)
    #RFRun <<- append(RFRun, list(StackedTrain))
    Reduce(sum, MAD_)
    
  },
  par.set = makeParamSet(
    makeNumericVectorParam("logNtree", len = 1, lower = 4, upper = 9),
    makeIntegerVectorParam("mtry", len = 1, lower = 1, upper = round(ncol(genotypeDTtrain) * (2/3))),
    makeIntegerVectorParam("nodesize", len = 1, lower = 1, upper = 50)
  ),
  minimize = !0
)

# generate 5 (latin hypercube base) low-discrepancy hyperparameter sets
des = generateDesign(n = 5, par.set = getParamSet(obj.fun), fun = lhs::randomLHS)

des$y = apply(des, 1, obj.fun)
surr.km = makeLearner("regr.km", predict.type = "se", covtype = "matern3_2", control = list(trace = F))

# after trying 5 latin hypercube based hyperparameter sets, identify next step using Expected Improvment Acquisition Function of (Bayesian Optimization)
modelsRF = mbo(obj.fun, design = des, learner = surr.km, show.info = !0,
               control = setMBOControlInfill(setMBOControlTermination(makeMBOControl(), iters = 30), crit = makeMBOInfillCritEI()))$opt.path$env[["path"]]

orderMAD = order(modelsRF$y); models = modelsRF[orderMAD, ]; 

# Fit the best perfroming model
fit =  randomForestSRC::rfsrc(formula(paste('cbind(',
                                            paste(colnames(genotypeDTtrain), collapse = ','),
                                            ') ~ ',
                                            ".")), data =  cbind.data.frame(genotypeDTtrain, phenotypeDTtrain),  
                              ntree = round(2**models[1,"logNtree"])[[1]], mtry = models[1,"mtry"][[1]], nodesize = models[1,"nodesize"][[1]])

predRF_ <- predict(fit, genotypeDTtest); 

# calculate test error
multivariateLossRF = sqrt(sapply(1:nrow(predRF_), function(r_)
  sum((as.numeric(phenotypeDTtest[r_,]) - as.numeric(predNNET_[r_,]))**2)
) / ncol(predRF_))



####################################
### keras NNet


obj.fun = makeSingleObjectiveFunction(
  name = "kerasNNet",
  fn = function(x) {
    
    # parameter set
    learningRate = c(1e-5,1e-4,1e-3,3e-3,1e-2,3e-2,.1,.3,1,3,10)[x[1][[1]]]; 
    activation = c("relu","tanh","sigmoid","linear","softmax")[x[2][[1]]]; 
    layerSize = x[3]; decay = c(1e-6,1e-5,1e-4,1e-3,1e-2)[x[4][[1]]]; beta_1 = x[5]; beta_2 = x[6]; dropout = x[7]; epoch = x[8]; 
    
    # repeated CV
    # MAD_ <- mclapply(1:2, function(repCv){
      
      flds <- caret::createFolds(1:nrow(genotypeDTtrain), k = 3, list = !0, returnTrain = !1); MAD_i <- 0; 
      
      for(k in 1:length(flds)){
        validData <- genotypeDTtrain[flds[[k]], ]; validResp <- phenotypeDTtrain[flds[[k]], ]; 
        trainData <- genotypeDTtrain[-flds[[k]], ]; trainResp <- phenotypeDTtrain[-flds[[k]], ]
        
        
        model <- keras_model_sequential() 
        model %>% 
          layer_dense(units = as.numeric(layerSize[[1]]), activation = 'relu', input_shape = ncol(trainData)) %>% layer_dropout(rate = as.numeric(dropout[[1]])) %>% 
          layer_dense(units = as.numeric(layerSize[[1]]), activation = as.character(activation[[1]])) %>% layer_dropout(rate = as.numeric(dropout[[1]])) %>% 
          layer_dense(units = as.numeric(layerSize[[1]]), activation = 'relu') %>% layer_dropout(rate = as.numeric(dropout[[1]])) %>%
          layer_dense(units = ncol(trainResp), activation = 'softmax')
        
        model %>% compile(
          loss = 'mean_absolute_error',
          optimizer = optimizer_adam( lr= as.numeric(learningRate[[1]]) , decay = as.numeric(decay[[1]]), beta_1 = as.numeric(beta_1[[1]]), 
                                                                                                          beta_2 = as.numeric(beta_2[[1]])),
          metrics = c('mae')
        )
        
        model %>% fit(
          x = data.matrix(trainData), y = data.matrix(trainResp), epochs = epoch[[1]], batch_size = 32, view_metrics = !1, verbose = !1)
        
        model_eval <- model %>% evaluate(x = data.matrix(validData), y = data.matrix(validResp), verbose = !1)
        
        # multivariate loss
        MAD_i <- MAD_i + model_eval$mean_absolute_error
      }
      MAD_i
   # }, mc.cores = 2)
    #RFRun <<- append(RFRun, list(StackedTrain))
    #Reduce(sum, MAD_)
    MAD_i
  },
  par.set = makeParamSet(
    makeIntegerVectorParam("learningRate", len = 1, lower = 1, upper = 11),
    makeIntegerVectorParam("activation", len = 1, lower = 1, upper = 5),
    makeIntegerVectorParam("layerSize", len = 1, lower = 5, upper = 50),
    makeIntegerVectorParam("decay", len = 1, lower = 1, upper = 5),
    makeNumericVectorParam("beta_1", len = 1, lower = 0, upper = 1),
    makeNumericVectorParam("beta_2", len = 1, lower = 0, upper = 1),
    makeNumericVectorParam("dropout", len = 1, lower = 0, upper = 0.7),
    makeIntegerVectorParam("epoch", len = 1, lower = 5, upper = 15)
  ),
  minimize = !0
)

# generate 10 (latin hypercube base) low-discrepancy hyperparameter sets
des = generateDesign(n = 10, par.set = getParamSet(obj.fun), fun = lhs::randomLHS)

des$y = apply(des, 1, obj.fun)
surr.km = makeLearner("regr.km", predict.type = "se", covtype = "matern3_2", control = list(trace = F))

# after trying 10 latin hypercube based hyperparameter sets, identify next step using Expected Improvment Acquisition Function of (Bayesian Optimization)
modelsNnet = mbo(obj.fun, design = des, learner = surr.km, show.info = !0,
               control = setMBOControlInfill(setMBOControlTermination(makeMBOControl(), iters = 30), crit = makeMBOInfillCritEI()))$opt.path$env[["path"]]

orderMAD = order(modelsNnet$y); models = modelsNnet[orderMAD, ]; 


# Fit with best performing model
  model <- keras_model_sequential() 
  model %>% 
    layer_dense(units = as.numeric(models[i,"layerSize"][[1]]), activation = 'relu', input_shape = ncol(genotypeDTtrain)) %>% layer_dropout(rate = as.numeric(models[i,"dropout"][[1]])) %>% 
    layer_dense(units = as.numeric(models[i,"layerSize"][[1]]), activation = as.character(c("relu","tanh","sigmoid","linear","softmax")[models[i,"activation"][[1]]])) %>% layer_dropout(rate = as.numeric(models[i,"dropout"][[1]])) %>% 
    layer_dense(units = as.numeric(models[i,"layerSize"][[1]]), activation = 'relu') %>% layer_dropout(rate = as.numeric(models[i,"dropout"][[1]])) %>%
    layer_dense(units = ncol(phenotypeDTtrain), activation = 'softmax')
  
  model %>% compile(
    loss = 'mean_absolute_error',
    optimizer = optimizer_adam( lr= as.numeric(models[i,"learningRate"][[1]]) , decay = as.numeric(models[i,"decay"][[1]]), beta_1 = as.numeric(models[i,"beta_1"][[1]]), 
                                beta_2 = as.numeric(models[i,"beta_2"][[1]])),
    metrics = c('mae')
  )
  
  hist <- model %>% fit(
    x = data.matrix(genotypeDTtrain), y = data.matrix(phenotypeDTtrain), epochs = models[i,"epoch"][[1]], batch_size = 32, view_metrics = !1, verbose = !1)
  #plot(hist)
  
  model_eval <- model %>% evaluate(x = data.matrix(genotypeDTtest), y = data.matrix(phenotypeDTtest), verbose = !1)
  
  predNNET_ <- model %>% predict(data.matrix(genotypeDTtrain))
  

  # calculate test error
  multivariateLossNNET = sqrt(sapply(1:nrow(predNNET_), function(r_)
    sum((as.numeric(phenotypeDTtest[r_,]) - as.numeric(predNNET_[r_,]))**2)
  ) / ncol(predNNET_))


  
####################################
### Visualization
  
  
  dt_ <- rbind(data.frame(err_ = multivariateLossNNET, model_ = "Neural network"), 
               data.frame(err_ = multivariateLossRF, model_ = "Multivariate Random forest"),
               data.frame(err_ = multivariateLossMULRM, model_ = "MURM"))

  pdf("compareMloss.pdf", width = 10, height = 10)
    yarrr::pirateplot(formula = err_ ~ model_, data = dt_, 
                      main = "Multivariate loss", theme = 2, 
                      inf.f.o = .4, inf.b.o = .5, inf.b.col = "#1FBED6",
                      point.o = .1, bar.f.o = .5, bean.f.o = .4, bean.b.o = .2, avg.line.o = 0, 
                      point.col = "black", bty="n", gl.lty = 3, gl.lwd = 0.5, cex.lab = 0.8, inf.p = .99)
  dev.off()
