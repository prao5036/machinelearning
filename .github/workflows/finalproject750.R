library(tidyverse)
library(ROCR)
library(randomForest)
library(boot)
library(gbm)
library(e1071)
library(caret)

heart.train <- read.csv("heart_train.csv")
heart.test <- read.csv("heart_test.csv")

###logistic regression/stepwise variable selection
all <- glm(condition~.,family=binomial,data=heart.train)
summary(all)
step <- step(all)
poststep <- glm(condition ~ sex + cp + trestbps + chol + fbs + thalach + exang + 
               slope + ca + thal-chol-thalach, family = binomial, data = heart.train)
summary(poststep)

pred.glm.train<-predict(poststep,type="response")
pred.step <- predict(poststep,heart.test,type = "response")

pred.glm <- prediction(pred.step, heart.test$condition)
perf.glm <- performance(pred.glm, "tpr", "fpr")
#AUC
auc.glm <- unlist(slot(performance(pred.glm, "auc"), "y.values"))

cost = rep(0, length(p.seq))
costfunc = function(obs, pred.p, pcut){
  weight1 = 3   # define the weight for "true=1 but pred=0" (FN)
  weight0 = 1    # define the weight for "true=0 but pred=1" (FP)
  c1 = (obs==1)&(pred.p<pcut)    # count for "true=1 but pred=0"   (FN)
  c0 = (obs==0)&(pred.p>=pcut)   # count for "true=0 but pred=1"   (FP)
  cost = mean(weight1*c1 + weight0*c0)  # misclassification with weight
  return(cost) # you have to return to a value when you write R functions
} # end of the function

p.seq = seq(0.01, 1, 0.01) 

for(i in 1:length(p.seq)){ 
  cost[i] = costfunc(obs = heart.train$condition, pred.p = pred.glm.train, pcut = p.seq[i])  
} # end of the loop
# draw a plot with X axis being all pcut and Y axis being associated cost
plot(p.seq, cost)

optimal.pcut.glm = p.seq[which(cost==min(cost))]

cost.glm <- costfunc(obs = heart.test$condition, pred.p = pred.step, pcut = optimal.pcut.glm)  
cost.glm
auc.glm

class.glm.test.opt<- (pred.step>optimal.pcut.glm)*1
# step 2. get confusion matrix, MR, FPR, FNR
table(heart.test$condition, class.glm.test.opt, dnn = c("True", "Predicted"))


MR.glm.out<- mean(heart.test$condition!= class.glm.test.opt)
FPR.glm.out<- sum(heart.test$condition==0 & class.glm.test.opt==1)/sum(heart.test$condition==0)
FNR.glm.out<- sum(heart.test$condition==1 & class.glm.test.opt==0)/sum(heart.test$condition==1)


################################### Random Forest ######################

heart.rf<- randomForest(as.factor(condition)~., data = heart.train, cutoff=c(3/4,1/4))
heart.rf
heart.rf$importance


plot(heart.rf, lwd=rep(2, 3))
legend("right", legend = c("OOB Error", "FPR", "FNR"), lwd=rep(2, 3), lty = c(1,2,3), col = c("black", "red", "green"))


heart.rf$mtry
#######################################
pred.heart.rf.train<- predict(heart.rf, type="prob")[,2]
#pred1 <- prediction(pred.heart.rf.train, as.factor(heart.train$condition))
#perf1 <- performance(pred1, "tpr", "fpr")
#plot(perf1, colorize=TRUE)
#in-sample AUC
#AUC.rf.in<-unlist(slot(performance(pred1, "auc"), "y.values"))


heart.rf.pred<- predict(heart.rf, newdata=heart.test, type = "prob")[,2]
pred <- prediction(heart.rf.pred, heart.test$condition)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
#out of sample AUC
AUC.rf.out<-unlist(slot(performance(pred, "auc"), "y.values"))



heart.rf.class.test<- predict(heart.rf, newdata=heart.test, type = "class")
table(heart.test$condition, heart.rf.class.test, dnn = c("True", "Pred"))


#missclassification rate for RF
mr.rf<- mean(heart.test$condition!=heart.rf.class.test)
RF<- cbind(mr.rf,AUC.rf.out)

###################################################
rf.trainPred<-predict(heart.rf,newdata=heart.train,type = "class")
rf.trainPred.prob<-predict(heart.rf,newdata=heart.train,type = "prob")

table(rf.trainPred,heart.train$condition,dnn=c("Prediction","Actual"))

rf.testPred<-predict(heart.rf, newdata=heart.test, type="class")
table(rf.testPred,heart.test$condition,dnn=c("Prediction","Actual"))

mr.rf<- mean(heart.test$condition!=rf.testPred)

costfunc = function(obs, pred.p, pcut){
  weight1 = 3   # define the weight for "true=1 but pred=0" (FN)
  weight0 = 1    # define the weight for "true=0 but pred=1" (FP)
  c1 = (obs==1)&(pred.p<pcut)    # count for "true=1 but pred=0"   (FN)
  c0 = (obs==0)&(pred.p>=pcut)   # count for "true=0 but pred=1"   (FP)
  cost = mean(weight1*c1 + weight0*c0)  # misclassification with weight
  return(cost) # you have to return to a value when you write R functions
} # end of the function

# define a sequence from 0.01 to 1 by 0.01
p.seq = seq(0.01, 1, 0.01) 

# write a loop for all p-cut to see which one provides the smallest cost
# first, need to define a 0 vector in order to save the value of cost from all pcut
cost = rep(0, length(p.seq))

for(i in 1:length(p.seq)){ 
  cost[i] = costfunc(obs = heart.train$condition, pred.p = rf.trainPred.prob[,2], pcut = p.seq[i])  
} # end of the loop
# draw a plot with X axis being all pcut and Y axis being associated cost
plot(p.seq, cost)

# find the optimal pcut (30)
optimal.pcut.rf = p.seq[which(cost==min(cost))]


# step 1. get binary classification
class.rf.train.opt<- (rf.trainPred.prob[,2]>optimal.pcut.rf)*1
# step 2. get confusion matrix, MR, FPR, FNR
table(heart.train$condition, class.rf.train.opt, dnn = c("True", "Predicted"))

MR.rf.in<- mean(heart.train$condition!= class.rf.train.opt)
FPR.rf.in<- sum(heart.train$condition==0 & class.rf.train.opt==1)/sum(heart.train$condition==0)
FNR.rf.in<- sum(heart.train$condition==1 & class.rf.train.opt==0)/sum(heart.train$condition==1)
cost.rf.in<- costfunc(obs = heart.train$condition, pred.p = rf.trainPred.prob[,2], pcut = optimal.pcut.rf)  

rf.Testp.prob<-predict(heart.rf, newdata=heart.test, type="prob")

class.rf.test.opt<- (rf.Testp.prob[,2]>optimal.pcut.rf)*1
table(heart.test$condition, class.rf.test.opt, dnn = c("True", "Predicted"))

pred <- prediction(rf.Testp.prob[,2], heart.test$condition)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)


MR.rf.out<- mean(heart.test$condition!= class.rf.test.opt)
FPR.rf.out<- sum(heart.test$condition==0 & class.rf.test.opt==1)/sum(heart.test$condition==0)
FNR.rf.out<- sum(heart.test$condition==1 & class.rf.test.opt==0)/sum(heart.test$condition==1)


rf.AUC<-unlist(slot(performance(pred, "auc"), "y.values"))

rf.cost<-costfunc(heart.test$condition,pred.p=rf.Testp.prob[,2],pcut=optimal.pcut.rf)


RF<- cbind(rf.AUC,rf.cost,FPR.rf.out,FNR.rf.out)

################################## Gradient Boosting ######################


# Find best number of trees
boost.model<- gbm(condition~., data = heart.train, 
                  distribution = "bernoulli", 
                  n.trees = 500, 
                  cv.folds = 5, 
                  n.cores = 4)
best.iter <- (gbm.perf(boost.model, method = "cv"))
cat("Optimal # of trees =", best.iter, "\n")

## Prediction

# Predicted probability
boost.trainPred<- predict(boost.model, newdata = heart.train,
                          n.trees = best.iter, type="response")
boost.trainPred.raw<- predict(boost.model, newdata = heart.train, 
                              n.trees = best.iter, type="link")

boost.testPred<- predict(boost.model, newdata = heart.test, type="response")

# define cost function
costfunc.boost<- function(obs, pred.p, pcut){
  weight1 = 3   # define the weight for "true=1 but pred=0" (FN)
  weight0 = 1   # define the weight for "true=0 but pred=1" (FP)
  c1 = (obs==1)&(pred.p<pcut)    # (FN)
  c0 = (obs==0)&(pred.p>=pcut)   # (FP)
  cost = mean(weight1*c1 + weight0*c0) 
  return(cost) 
} 

p.seq<- seq(0.01, 1, 0.01)
cost<- rep(0, length(p.seq))

for(i in 1:length(p.seq)){ 
  cost[i] = costfunc.boost(obs = heart.train$condition, 
                           pred.p = boost.trainPred.raw,
                           pcut = p.seq[i])  
} 
plot(p.seq, cost)

# Find the optimal p-cut
optimal.pcut.boost = min(p.seq[which(cost==min(cost))])
cat("Optimal p-cut (GBM) =", optimal.pcut.boost, "\n") 

# Step 1. get binary classification
class.boost.train.opt<- (boost.trainPred>optimal.pcut.boost)*1

boost.test.opt<- predict(boost.model, newdata = heart.test)
class.boost.test.opt<- (boost.test.opt>optimal.pcut.boost)*1

# Step 2. get confusion matrix, FPR, FNR, AUC & Cost using testing set
 
# Out-of-sample
table(heart.test$condition, class.boost.test.opt, dnn = c("True", "Predicted"))
pred.out <- prediction(boost.test.opt, heart.test$condition)
perf.out <- performance(pred.out, "tpr", "fpr")
plot(perf.out, colorize=TRUE)

FPR.boost.out.opt<- sum(heart.test$condition==0 & class.boost.test.opt==1)/sum(heart.test$condition==0)
FNR.boost.out.opt<- sum(heart.test$condition==1 & class.boost.test.opt==0)/sum(heart.test$condition==1)
cost.boost.out.opt<- costfunc.boost(obs = heart.test$condition, 
                                    pred.p = boost.testPred, 
                                    pcut = optimal.pcut.boost) 


boost.AUC.final<-unlist(slot(performance(pred.out, "auc"), "y.values"))
cat("AUC (GBM) =", boost.AUC.final, "\n")  

boost.cost.final<- costfunc.boost(heart.test$condition,
                                  pred.p=boost.test.opt,
                                  pcut=optimal.pcut.boost)
cat("Cost (GBM) =", boost.cost.final, "\n") 

# Results of using Optimal p-cut with a 3:1 weight for FNR:FPR for out-of-sample testing:
c(FPR = FPR.boost.out.opt, FNR = FNR.boost.out.opt, AUC = boost.AUC.final, Cost = boost.cost.final)

##################### Naive Bayes ##########################################

#install.packages("e1071")
library(e1071)
#install.packages("caret")
library(caret)

nb_model<- naiveBayes(as.factor(condition)~., data=heart.train)
nb_model
summary(nb_model)


nb.trainPred<-predict(nb_model,newdata=heart.train,type = "class")
nb.trainPred.raw<-predict(nb_model,newdata=heart.train,type = "raw")

table(nb.trainPred,heart.train$condition,dnn=c("Prediction","Actual"))

nb.testPred<-predict(nb_model, newdata=heart.test, type="class")
table(nb.testPred,heart.test$condition,dnn=c("Prediction","Actual"))

mr.nb<- mean(heart.test$condition!=nb.testPred)

costfunc = function(obs, pred.p, pcut){
  weight1 = 3   # define the weight for "true=1 but pred=0" (FN)
  weight0 = 1    # define the weight for "true=0 but pred=1" (FP)
  c1 = (obs==1)&(pred.p<pcut)    # count for "true=1 but pred=0"   (FN)
  c0 = (obs==0)&(pred.p>=pcut)   # count for "true=0 but pred=1"   (FP)
  cost = mean(weight1*c1 + weight0*c0)  # misclassification with weight
  return(cost) # you have to return to a value when you write R functions
} # end of the function

# define a sequence from 0.01 to 1 by 0.01
p.seq = seq(0.01, 1, 0.01) 

# write a loop for all p-cut to see which one provides the smallest cost
# first, need to define a 0 vector in order to save the value of cost from all pcut
cost = rep(0, length(p.seq))

for(i in 1:length(p.seq)){ 
  cost[i] = costfunc(obs = heart.train$condition, pred.p = nb.trainPred.raw[,2], pcut = p.seq[i])  
} # end of the loop
# draw a plot with X axis being all pcut and Y axis being associated cost
plot(p.seq, cost)

# find the optimal pcut
optimal.pcut.nb = p.seq[which(cost==min(cost))]

# step 1. get binary classification
class.nb.train.opt<- (nb.trainPred.raw[,2]>optimal.pcut.nb)*1
# step 2. get confusion matrix, MR, FPR, FNR
table(heart.train$condition, class.nb.train.opt, dnn = c("True", "Predicted"))

nb.Testp.raw<-predict(nb_model, newdata=heart.test, type="raw")

class.nb.test.opt<- (nb.Testp.raw[,2]>optimal.pcut.nb)*1
table(heart.test$condition, class.nb.test.opt, dnn = c("True", "Predicted"))

pred <- prediction(nb.Testp.raw[,2], heart.test$condition)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)

nb.AUC.final<-unlist(slot(performance(pred, "auc"), "y.values"))

nb.cost.final<-costfunc(heart.test$condition,pred.p=nb.Testp.raw[,2],pcut=optimal.pcut.nb)

nb.FPR.out.opt<- sum(heart.test$condition==0 & class.nb.test.opt==1)/sum(heart.test$condition==0)
nb.FNR.out.opt<- sum(heart.test$condition==1 & class.nb.test.opt==0)/sum(heart.test$condition==1)

######################################### Neural Network ######################

require(neuralnet)
nn=neuralnet(condition~.,data=heart.train, hidden=3,act.fct = "logistic",
             linear.output = FALSE)
plot(nn)
Predict=compute(nn,heart.test)
Predict$net.result

train.labels.nn <- heart.train$condition
train.predictions.nn <- predict(nn, heart.train, type="raw")
## computing a simple ROC curve (x-axis: fpr, y-axis: tpr)
detach(package:neuralnet)
pred.nn = prediction(Predict$net.result, heart.test$condition)
perf.nn = performance(pred.nn, "tpr", "fpr")
plot(perf.nn, lwd=2, col="blue", main="ROC - Title")
auc.nn <- unlist(slot(performance(pred.nn, "auc"), "y.values"))

cost = rep(0, length(p.seq))
costfunc = function(obs, pred.p, pcut){
  weight1 = 3   # define the weight for "true=1 but pred=0" (FN)
  weight0 = 1    # define the weight for "true=0 but pred=1" (FP)
  c1 = (obs==1)&(pred.p<pcut)    # count for "true=1 but pred=0"   (FN)
  c0 = (obs==0)&(pred.p>=pcut)   # count for "true=0 but pred=1"   (FP)
  cost = mean(weight1*c1 + weight0*c0)  # misclassification with weight
  return(cost) # you have to return to a value when you write R functions
} # end of the function

p.seq = seq(0.01, 1, 0.01) 

for(i in 1:length(p.seq)){ 
  cost[i] = costfunc(obs = heart.train$condition, pred.p = train.predictions.nn, pcut = p.seq[i])  
} # end of the loop

plot(p.seq, cost)

optimal.pcut.nn = p.seq[which(cost==min(cost))]
mean(optimal.pcut.nn)

prob.nn <- Predict$net.result
pred.nn <- ifelse(prob.nn>mean(optimal.pcut.nn), 1, 0)
cost.nn <- costfunc(obs = heart.test$condition, pred.p = Predict$net.result, pcut = optimal.pcut.nn)  
auc.nn
cost.nn
