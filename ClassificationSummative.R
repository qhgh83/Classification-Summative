install.packages(("ISLR"))
install.packages(("corrplot"))
install.packages(("randomForest"))
install.packages(("class"))
install.packages(("Amelia"))
install.packages(("ggplot2"))
install.packages(("GGally"))
install.packages(("e1071"))
install.packages(("neuralnet"))
install.packages(("rgl"))
library(ISLR)
library(corrplot)
library(randomForest)
library(class)
library(Amelia)
library(ggplot2)
library(GGally)
library(e1071)
library(neuralnet)
library(rgl)
set.seed(100)
data = read.csv(file = "C:/Users/owenm/OneDrive/Documents/University/Masters/ASML/heart_failure.csv")
#Check data over for odd, extreme or missing values
print(summary(data))
print(sapply(data,class))
correlations = cor(data)
corrplot(correlations, method="circle")
missmap(data)
contData = data[,c(1,3,5,7,8,9,12)]
for(i in 1:7)
{
  contData[,i] = (contData[,i] - mean(contData[,i]))/sd(contData[,i])
}
boxplot(contData)
#Low levels of co-linearity between each of the predictor variables
#Only four of the predictors are correlated with fata_mi these are:
#age, ejection_fraction, serum_creatinine, time
#Select these
cols = c(1,5,8,12,13)
reducedData = as.data.frame(data[cols])
for(i in 1:4)
{
  reducedData[,i] = (reducedData[,i] - mean(reducedData[,i]))/sd(reducedData[,i])
}
correlations = cor(reducedData)
corrplot(correlations, method="circle")
#Split data into test and train subsets
N = length(reducedData[,1])
shuffledData = reducedData[sample(1:N),]
#Start with crossfold validation
lRMSE = c(0,0,0,0,0,0,0,0,0,0)
rfMSE = c(0,0,0,0,0,0,0,0,0,0)
nbMSE = c(0,0,0,0,0,0,0,0,0,0)
nnMSE = c(0,0,0,0,0,0,0,0,0,0)
coefs = matrix(0, nrow = 10, ncol = 5)
for(i in 1:10)
{
  #Split into train and test
  sub = (30*(i-1)+1):min(30*i,299)
  train = shuffledData[-sub,]
  test = shuffledData[sub,]
  #Logistic regression
  logReg = glm(train[,5] ~ age + ejection_fraction + serum_creatinine + time, family=binomial(link='logit'),data=train[,-5])
  logPred = round(predict(logReg, test[,-5], type="response"))
  lRMSE[i] = sum((logPred-test[,5])^2)
  coefs[i,] = logReg$coefficients
  #Random Forest
  forest = randomForest(as.factor(train[,5]) ~ ., data=train[,-5], ntree=56, mrty = 4)
  rfPred = as.integer(predict(forest, test[,-5], type = "response"))-1
  rfMSE[i] = sum((rfPred - test[,5])^2)
  #Naive Bayes
  nbModel = naiveBayes(train[,5] ~ age + ejection_fraction + serum_creatinine + time, data=train[,-5])
  nbPred = as.integer(predict(nbModel, test[,-5]))-1
  nbMSE[i] = sum((nbPred - test[,5])^2)
  #Neural Network
  nnModel=neuralnet(train[,5] ~ age + ejection_fraction + serum_creatinine + time, data=train, hidden=c(4,4),act.fct = "logistic", linear.output = FALSE, threshold = 0.05, stepmax = 10^6)
  nnPred = round(compute(nnModel,test)$net.result)
  nnMSE[i] = sum((nnPred - test[,5])^2)
}
print(coefs)
print(colMeans(coefs))
print(apply(coefs, 2, sd))
print(mean(lRMSE))
print(mean(rfMSE))
print(mean(nbMSE))
print(mean(nnMSE))

plot(nnModel)

logRegData = data.frame(
  name=c("Intercept", names(reducedData)[-5]),
  value=colMeans(coefs),
  sd=apply(coefs, 2, sd)
)

ggplot(logRegData) +
  geom_bar( aes(x=name, y=value), stat="identity", fill="skyblue", alpha=0.7) +
  geom_errorbar( aes(x=name, ymin=value-sd, ymax=value+sd), width=0.4, colour="orange", alpha=0.9, size=1.3)

#Optimise hyperparameter for k
bestKNMSE = c(100000000000000000,0,0,0,0,0,0,0,0,0)
bestK = 0
bestRfMSE = c(100000000000000000,0,0,0,0,0,0,0,0,0)
bestTrees = 0
for(k in 1:100)
{
  kNMSE = c(0,0,0,0,0,0,0,0,0)
  rfMSE = c(0,0,0,0,0,0,0,0,0)
  for(i in 1:10)
  {
    #Split into train and test
    sub = (30*(i-1)+1):min(30*i,299)
    train = shuffledData[-sub,]
    test = shuffledData[sub,]
    #K Neighbours
    kNPred = as.integer(knn(train[,c(-5, -3)],test[,c(-5, -3)],cl=train[,5],k=k))-1
    kNMSE[i] = sum((kNPred-test[,5])^2)
    #Random Forest
    forest = randomForest(as.factor(train[,5]) ~ ., data=train[,-5], ntree=k)
    rfPred = as.integer(predict(forest, test[,-5], type = "response"))-1
    rfMSE[i] = sum((rfPred - test[,5])^2)
  }
  
  if(mean(kNMSE) < mean(bestKNMSE))
  {
    bestK = k
    bestKNMSE = kNMSE
  }
  
  if(mean(rfMSE) < mean(bestRfMSE))
  {
    bestTrees = k
    bestRfMSE = rfMSE
  }
}
print(bestK)
print(mean(bestKNMSE))
print(bestTrees)
print(mean(bestRfMSE))

plot3d( 
  x=shuffledData[,1], y=shuffledData[,2], z=shuffledData[,4], 
  col = c("red","blue")[shuffledData[,5]+1], 
  type = 's', 
  radius = .1,
  xlab="Age", ylab="Ejection Fraction", zlab="Time")
FP = c(0,0,0,0,0,0,0,0,0,0)
FN = c(0,0,0,0,0,0,0,0,0,0)
#Further Nearest Neighbour analysis
for(i in 1:10)
{
  #Split into train and test
  sub = (30*(i-1)+1):min(30*i,299)
  train = shuffledData[-sub,]
  test = shuffledData[sub,]
  #K Neighbours
  kNPred = as.integer(knn(train[,c(-5, -3)],test[,c(-5, -3)],cl=train[,5],k=6))-1
  kNMSE[i] = sum((kNPred-test[,5])^2)
  t = table(kNPred, test[,5])
  FN[i] = t[1,2]
  FP[i] = t[2,1]
}
print(mean(FP))
print(mean(FN))

