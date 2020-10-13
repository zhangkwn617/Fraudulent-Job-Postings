wine <- read.csv('project/winequality-red.csv')
val <- sample(nrow(wine), 0.8*nrow(wine))
train <- wine[val,]
test <- wine[-val,]

############ get an idea of our variable of interest ###############
sd(train$quality)
summary(train$quality)

################## Multiple Regression ######################3

simp <- lm(quality~1, data=train)
comp <- lm(quality~., data=train)

regForward = step(simp,
                  scope=formula(comp),
                  direction="forward", 
                  k=log(length(train))) 
#quality ~ alcohol + volatile.acidity + sulphates + total.sulfur.dioxide + chlorides + pH + free.sulfur.dioxide
regBack = step(comp,
               direction="backward",
               k=log(length(train)))
#quality ~ volatile.acidity + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + pH + sulphates + alcohol
regMix = step(simp,
                  scope=formula(comp),
                  direction="both",
                  k=log(length(train)))
#quality ~ alcohol + volatile.acidity + sulphates + total.sulfur.dioxide + chlorides + pH + free.sulfur.dioxide

#conclude that fixed.acidity, residual.sugar, citric.acid, and density are not good predictors of quality

#all returned same formula:
M <- lm(formula = quality ~ alcohol + volatile.acidity + sulphates + 
     total.sulfur.dioxide + chlorides + pH + free.sulfur.dioxide, 
   data = train)
summary(M)

predlm <- predict(M, test)
mean((test$quality - predlm)^2)

#MSE: 0.4863969

######################## Lasso and Ridge Regression #################################3

library(glmnet)
lasso = glmnet(as.matrix(wine[,1:(ncol(wine)-1)]),as.matrix(wine[,ncol(wine)]),alpha=1)
ridge = glmnet(as.matrix(wine[,1:(ncol(wine)-1)]),as.matrix(wine[,ncol(wine)]),alpha=0)

plot(lasso)
plot(ridge)

CV.L = cv.glmnet(as.matrix(train[,1:(ncol(train)-1)]),as.matrix(train[,ncol(train)]),alpha=1)
CV.R = cv.glmnet(as.matrix(train[,1:(ncol(train)-1)]),as.matrix(train[,ncol(train)]),alpha=0)

LamR = CV.R$lambda.1se
LamL = CV.L$lambda.1se

coef.R = predict(CV.R,type="coefficients",s=LamR)
coef.L = predict(CV.L,type="coefficients",s=LamL)

plot(abs(coef.R[2:20]),abs(coef.L[2:20]),
     ylim=c(0,1),xlim=c(0,1))
abline(0,1)

plot(log(CV.R$lambda),sqrt(CV.R$cvm),
     main="Ridge CV (k=10)",
     xlab="log(lambda)",
     ylab = "RMSE",
     col=4,
     cex.lab=1.2)
abline(v=log(LamR),lty=2,col=2,lwd=2)

plot(log(CV.L$lambda),sqrt(CV.L$cvm),
     main="LASSO CV (k=10)",xlab="log(lambda)",
     ylab = "RMSE",col=4,type="b",cex.lab=1.2)
abline(v=log(LamL),lty=2,col=2,lwd=2)

#Lasso excluded: citric.acid, residual.sugar, free.sulfur.dioxide, density

ridgePred <- predict(CV.R, s=LamR, newx = as.matrix(test[,1:(ncol(test)-1)]))

mean((test$quality - ridgePred)^2)

#MSE: 0.5202214

lassoPred <- predict(CV.L, s=LamL, newx = as.matrix(test[,1:(ncol(test)-1)]))

mean((test$quality - lassoPred)^2)

#MSE:  0.517347

#################################### KNN? #####################################

near = kknn(quality~., #Formula (a point means that all covariates are used)
            train = train, #Train matrix/df
            test = test, #Test matrix/df
            k=10, #Number of neighbors
            kernel = "rectangular")

mean((test$quality - near$fitted)^2)

#MSE: 0.5280938

################################# Tree ######################################
library(rpart)
library(tree)
library(rpart.plot)

#Grow big tree first
big.tree = rpart(quality~.,
                 method="anova", #split maximizes the sum-of-squares between the new partitions
                 data=train, #data frame
                 control=rpart.control(minsplit=5, #the minimum number of obs in each leaf
                                       cp=.0001)) #complexity parameter (see rpart.control help)
nbig = length(unique(big.tree$where))
cat('Number of leaf nodes: ',nbig,'\n')

#Fit on train, predict on val for vector of cp.
cpvec = big.tree$cptable[,"CP"] #cp values to try
ntree = length(cpvec) #number of cv values = number of trees fit.
iltree = rep(0,ntree) #in-sample loss
oltree = rep(0,ntree) #out-of-sample loss
sztree = rep(0,ntree) #size of each tree
for(i in 1:ntree) {
   if((i %% 10)==0) cat('tree i: ',i, "out of", ntree, '\n')
   temptree = prune(big.tree,cp=cpvec[i]) #Pruned tree by cp
   sztree[i] = length(unique(temptree$where)) #Number of leaves
   iltree[i] = sum((train$quality-predict(temptree))^2) #in-sample loss
   ofit = predict(temptree,test) #Validation prediction
   oltree[i] = sum((test$quality-ofit)^2) #out-of-sample loss
}

oltree=sqrt(oltree/nrow(test)) #RMSE out-of-sample
iltree = sqrt(iltree/nrow(train)) #RMSE in-sample

#Plot losses
rgl = range(c(iltree,oltree)) #Range of the values for the plot
plot(range(sztree),rgl,
     type='n', #Type = n removes points from plot
     xlab='Number of Leaves',ylab='Loss')
points(sztree,iltree,
       pch=15, #Type of point
       col='red')
points(sztree,oltree,
       pch=16, #Type of point
       col='blue')
legend("topright", #Position of the legend
       legend=c('in-sample','out-of-sample'), #Text in the legend
       pch=c(15,16), #Types of points
       col=c('red','blue')) #Color of points


#Write val preds
iitree = which.min(oltree) #Tree which minimizes out-of-sample loss
thetree = prune(big.tree,cp=cpvec[iitree]) #Pruning your big tree using the minimum oos loss
thetreepred = predict(thetree,test) #Getting prediction of your validation set

#If you want, you can save the predictions in a file (.txt for example)
#using the write function
#write(thetreepred, file='thetreepred.txt',ncol=1)

#create a graphic of the tree
prp(thetree)

#################################### Random Forest #####################################
library(randomForest)
rffit = randomForest(quality~., #Formula (. means all variables are included)
                     data=test,#Data frame
                     mtry=5, #Number candidates variables at each split
                     ntree=500)#Number of trees in the forest

#Variance importance for Random Forest model
varImpPlot(rffit)

rfpred <- predict(rffit, test)
mean((test$quality - rfpred)^2)

#MSE: 0.09440834



