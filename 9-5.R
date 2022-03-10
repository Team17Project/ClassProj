###Question 9.5 ####

#We have seen that we can fit an SVM with a non-linear kernel 
#in order to perform classification using a non-linear decision boundary.
#We will now see that we can also obtain a non-linear decision boundary by performing 
#logistic regression using non-linear transformations of the features

rm(list=ls())

#a
#Generate a data set with n = 500 and p = 2
#this way there's a quadratic decision boundary between them
set.seed(421)
x1 = runif(500) - 0.5  
x2 = runif(500) - 0.5  
y = 1 * (x1^2 - x2^2 > 0) 

#b
#plot the observations, colored according to their class labels
#X1 on the x-axis and X2 on the y-axis
plot(x1[y == 0], x2[y == 0], col = "red", xlab = "X1", ylab = "X2", pch = "+")
points(x1[y == 1], x2[y == 1], col = "blue", pch = 4)

#the above plot shows a non-linear decision boundary

#c
#Fit a logistic regression model to the data using X1 and X2 as predictors
lm.fit = glm(y ~ x1 + x2, family = binomial)
summary(lm.fit)

#d
#Apply this model to the training data in order to obtain
# a predicted class label for each training observation. 
#Plot the observations 
data = data.frame(x1 = x1, x2 = x2, y = y) #build a dataframe
lm.prob = predict(lm.fit, data, type = "response") #predict response
lm.pred = ifelse(lm.prob > 0.52, 1, 0) #transform the probability values from above to 1s or 0s.
data.pos = data[lm.pred == 1, ] #separate into positives and negatives
data.neg = data[lm.pred == 0, ]
plot(data.pos$x1, data.pos$x2, col = "blue", xlab = "X1", ylab = "X2", pch = "+")
points(data.neg$x1, data.neg$x2, col = "red", pch = 4)

#e
#fit a logistic regression model to the data using non-linear functions of X1, X2 as predictors
lm.fit = glm(y ~ poly(x1, 2) + poly(x2, 2) + I(x1 * x2), data = data, family = binomial) #use generalized linear model

#f
#Apply this model to the training data in order to obtain a predicted class label for each training observation
#The decision boundary should be non-linear.
lm.prob = predict(lm.fit, data, type = "response")  #predict lm.fit using the data
lm.pred = ifelse(lm.prob > 0.5, 1, 0) #transform to 1's and 0s
data.pos = data[lm.pred == 1, ]
data.neg = data[lm.pred == 0, ]
plot(data.pos$x1, data.pos$x2, col = "blue", xlab = "X1", ylab = "X2", pch = "+")
points(data.neg$x1, data.neg$x2, col = "red", pch = 4)

#g
#Fit a support vector classifier to the data with X1 and X2 as predictors.
library(e1071)
svm.fit = svm(as.factor(y) ~ x1 + x2, data, kernel = "linear", cost = 0.1) #make sure to use as.factor set the kernel to linear
svm.pred = predict(svm.fit, data)
data.pos = data[svm.pred == 1, ]
data.neg = data[svm.pred == 0, ]
plot(data.pos$x1, data.pos$x2, col = "blue", xlab = "X1", ylab = "X2", pch = "+")
points(data.neg$x1, data.neg$x2, col = "red", pch = 4)

#h
#Fit a SVM using non-linear kernel to the data. Obtain a class prediction for each training observation.
svm.fit = svm(as.factor(y) ~ x1 + x2, data, gamma = 1) #here we leave the kernel part out
svm.pred = predict(svm.fit, data)
data.pos = data[svm.pred == 1, ]
data.neg = data[svm.pred == 0, ]
plot(data.pos$x1, data.pos$x2, col = "blue", xlab = "X1", ylab = "X2", pch = "+")
points(data.neg$x1, data.neg$x2, col = "red", pch = 4)

#i
#This experiment enforces the idea that SVMs with non-linear kernel are extremely powerful in finding non-linear boundary.
#Both, logistic regression with non-interactions and SVMs with linear kernels fail to find the decision boundary. 
#Adding interaction terms to logistic regression seems to give them same power as radial-basis kernels.
#However, there is some manual efforts and tuning involved in picking right interaction terms.
#This effort can become prohibitive with large number of features.
#Radial basis kernels, on the other hand, only require tuning of one parameter - gamma - which can be easily done using cross-validation.
