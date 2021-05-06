#########
# Setup #
#########

# Clear workspace
rm(list = ls())

# Set working directory
setwd("D:\\Users\\Nicolas\\Desktop\\CSC 597 Statistical Learning\\Project")

# Includes
library(class)        # KNN
library(MASS)         # LDA, QDA
library(tree)         # Decision Trees
library(gbm)          # Boosting
library(randomForest) # Random Forests, Bagging
library(e1071)        # SVC, SVM

# Set Seed
set.seed(12)

##############
# Input Data #
##############

input_data = read.csv("D:\\Users\\Nicolas\\Desktop\\CSC 597 Statistical Learning\\Datasets\\Project\\Epileptic Seizure Recognition.csv")

# Check for empty rows
sum(is.na(input_data))

# Remove empty rows
input_data = na.omit(input_data)

# Validate data is clean
sum(is.na(input_data))

last_col = ncol(input_data)

# Convert class variable to an enum and rename it
# Convert the class to a factor type labeled "benign" and "malignant"
mental_state = as.factor(
  ifelse(
    input_data[,last_col] == 1, 
    "recording_of_seizure_activity", 
    ifelse(
      input_data[,last_col] == 2,
      "eeg_recorded_from_tumor_location",
      ifelse(
        input_data[,last_col] == 3,
        "eeg_recorded_from_healthy_location",
        ifelse(
          input_data[,last_col] == 4,
          "eyes_closed",
          "eyes_open"
        )
      )
    )
  )
)

# Bind the input data with the labeled class and remove the unnamed column
input_df = data.frame(input_data[,c(2:(last_col - 1))], mental_state)

last_col = ncol(input_df)
num_samples = nrow(input_df)

# Split data into training and testing at 0.8 (4/5) ratio
train = sample(c(TRUE, TRUE, TRUE, TRUE, FALSE), num_samples, replace=TRUE)

# Information about data
names(input_df)
summary(input_df)
cor((input_df[,-last_col])[,-1])
contrasts(input_df$mental_state)

# Check specifically if y classes are balanced
summary(input_df$mental_state)

#######
# KNN #
#######

# Number of folds for CV
fold_k = 10

# Variable K values
k_val = c(1, 2, 5, 10, 25, 50)

# Place each sample into a fold
folds = sample(1:fold_k, nrow(input_df), replace=TRUE)

# Create empty error matrices
cv.errors.knn.train = matrix(NA, fold_k, length(k_val))
cv.errors.knn.test = matrix(NA, fold_k, length(k_val))

for (i in 1:length(k_val)) {
  for (j in 1:fold_k) {
    # Build knn predictions (training and testing)
    # exclude non-numeric columns and class column
    # train.x = input_df[folds != j, c(1:(last_col - 1))]
    # test.x  = input_df[folds == j, c(1:(last_col - 1))]
    # train.y = input_df$mental_state[folds != j]
    knn.train.pred = knn(
      input_df[folds != j & train, c(1:(last_col - 1))],
      input_df[folds == j & train, c(1:(last_col - 1))],
      input_df$mental_state[folds != j & train],
      k=k_val[i]
    )
    # Predict on the testing set
    knn.test.pred = knn(
      input_df[folds != j & train, c(1:(last_col - 1))],
      input_df[folds == j & !train, c(1:(last_col - 1))],
      input_df$mental_state[folds != j & train],
      k=k_val[i]
    )
    
    # Record the training error for each value of k
    cv.errors.knn.train[j, i] = 
      sum(
        (knn.train.pred != input_df$mental_state[folds == j & train]),
        na.rm = TRUE
      ) / length(knn.train.pred)
    # Record the testing error for each value of k
    cv.errors.knn.test[j, i] = 
      sum(
        (knn.test.pred != input_df$mental_state[folds == j & !train]),
        na.rm = TRUE
      ) / length(knn.test.pred)
  }
}

# Print error rates
cv.errors.knn.train
cv.errors.knn.test

# Compute averages across CV runs
avgs_train = colMeans(cv.errors.knn.train)
avgs_train
avgs_test = colMeans(cv.errors.knn.test)
avgs_test

# Plot the averages against the values chosen for k
par(mfrow=c(1,2))
# Equalize the ymax and ymin limits so the graphs are comparable side by side
ymax = max(c(max(avgs_train), max(avgs_test))) + 0.03
ymin = min(c(min(avgs_train), min(avgs_test))) - 0.03
plot(k_val, avgs_train, ylim=c(ymin, ymax))
plot(k_val, avgs_test, ylim=c(ymin, ymax))


#######
# LDA #
#######

# Number of folds for CV
fold_k = 10

# Place each sample into a fold
folds = sample(1:fold_k, nrow(input_df), replace=TRUE)

# Create empty error matrices
cv.errors.lda.train = matrix(NA, fold_k, 1)
cv.errors.lda.test = matrix(NA, fold_k, 1)

for (j in 1:fold_k) {
  # Build LDA using all folds except for 1 and the training set
  lda.fit = lda(mental_state~., data = input_df[folds != j & train,])
  # Prediction on the one unused fold (training and testing)
  lda.pred.train = predict(lda.fit, input_df[folds == j & train,])
  lda.pred.test = predict(lda.fit, input_df[folds == j & !train,])
  
  # Get the class (training and testing)
  lda.train.class = lda.pred.train$class
  lda.test.class = lda.pred.test$class
  # Record the training error
  cv.errors.lda.train[j, 1] =
    sum(
      (lda.train.class != input_df$mental_state[folds == j & train]),
      na.rm = TRUE
    ) / length(lda.train.class)
  # Record the testing error
  cv.errors.lda.test[j, 1] =
    sum(
      (lda.test.class != input_df$mental_state[folds == j & !train]),
      na.rm = TRUE
    ) / length(lda.test.class)
}

# Print errors
cv.errors.lda.train
cv.errors.lda.test

# Find average train error
avg_error_train = mean(cv.errors.lda.train[, 1])
avg_error_train

# Find average test error
avg_error_test = mean(cv.errors.lda.test[, 1])
avg_error_test

# Plot errors across CV runs
par(mfrow=c(1,2))
# Equalize the ymax and ymin limits so the graphs are comparable side by side
ymax = max(c(max(cv.errors.lda.train[, 1]), max(cv.errors.lda.test[, 1]))) + 0.03
ymin = min(c(min(cv.errors.lda.train[, 1]), min(cv.errors.lda.test[, 1]))) - 0.03
plot(c(1:fold_k), cv.errors.lda.train[, 1], ylim=c(ymin, ymax))
plot(c(1:fold_k), cv.errors.lda.test[, 1], ylim=c(ymin, ymax))


#######
# QDA #
#######

# Number of folds for CV
fold_k = 10

# Place each sample into a fold
folds = sample(1:fold_k, nrow(input_df), replace=TRUE)

# Create empty error matrices
cv.errors.qda.train = matrix(NA, fold_k, 1)
cv.errors.qda.test = matrix(NA, fold_k, 1)

for (j in 1:fold_k) {
  # Build QDA using all folds except for 1
  qda.fit = qda(mental_state~., data = input_df[folds != j & train,])
  # Predict on the one unused fold (training and testing)
  qda.train.pred = predict(qda.fit, input_df[folds == j & train,])
  qda.test.pred = predict(qda.fit, input_df[folds == j & !train,])
  
  # Get the class and the error (training and testing)
  qda.train.class = qda.train.pred$class
  qda.test.class = qda.test.pred$class
  # Record the training error
  cv.errors.qda.train[j, 1] = 
    sum(
      (qda.train.class != input_df$mental_state[folds == j & train]),
      na.rm = TRUE
    ) / length(qda.train.class)
  # Record the testing error
  cv.errors.qda.test[j, 1] = 
    sum(
      (qda.test.class != input_df$mental_state[folds == j & !train]),
      na.rm = TRUE
    ) / length(qda.test.class)
}

# Print errors
cv.errors.qda.train
cv.errors.qda.test

# Find average train error
avg_error_train = mean(cv.errors.qda.train[, 1])
avg_error_train

# Find average test error
avg_error_test = mean(cv.errors.qda.test[, 1])
avg_error_test

# Plot errors across CV runs
par(mfrow=c(1,2))
# Equalize the ymax and ymin limits so the graphs are comparable side by side
ymax = max(c(max(cv.errors.qda.train[, 1]), max(cv.errors.qda.test[, 1]))) + 0.03
ymin = min(c(min(cv.errors.qda.train[, 1]), min(cv.errors.qda.test[, 1]))) - 0.03
plot(c(1:fold_k), cv.errors.qda.train[, 1], ylim=c(ymin, ymax))
plot(c(1:fold_k), cv.errors.qda.test[, 1], ylim=c(ymin, ymax))


##################
# Decision Trees #
##################

# Number of folds for CV
fold_k = 10

# Place each sample into a fold
folds = sample(1:fold_k, nrow(input_df), replace=TRUE)

# Create empty error matrices
cv.errors.tree.train = matrix(NA, fold_k, 1)
cv.errors.tree.test = matrix(NA, fold_k, 1)

for (j in 1:fold_k) {
  # Fit the tree using all the folds except for j
  tree.fit = tree(mental_state~., data=input_df[folds != j & train,])
  
  # predict on fold j
  tree.train.pred = predict(tree.fit, input_df[folds == j & train,],
                            type="class")
  tree.test.pred = predict(tree.fit, input_df[folds == j & !train,],
                           type="class")
  # Record training error
  cv.errors.tree.train[j, 1] = 
    sum(
      (tree.train.pred != input_df$mental_state[folds == j & train]),
      na.rm = TRUE
    ) / length(tree.train.pred)
  # Record testing error
  cv.errors.tree.test[j, 1] = 
    sum(
      (tree.test.pred != input_df$mental_state[folds == j & !train]),
      na.rm = TRUE
    ) / length(tree.test.pred)
}

# Print a sample tree
par(mfrow=c(1,1))
plot(tree.fit)
text(tree.fit, pretty=0, cex=0.75)

# Print errors
cv.errors.tree.train
cv.errors.tree.test

# Find average train error
avg_error_train = mean(cv.errors.tree.train[, 1])
avg_error_train

# Find average test error
avg_error_test = mean(cv.errors.tree.test[, 1])
avg_error_test

# Plot errors across CV runs
par(mfrow=c(1,2))
# Equalize the ymax and ymin limits so the graphs are comparable side by side
ymax = max(c(max(cv.errors.tree.train[, 1]), max(cv.errors.tree.test[, 1]))) + 0.03
ymin = min(c(min(cv.errors.tree.train[, 1]), min(cv.errors.tree.test[, 1]))) - 0.03
plot(c(1:fold_k), cv.errors.tree.train[, 1], ylim=c(ymin, ymax))
plot(c(1:fold_k), cv.errors.tree.test[, 1], ylim=c(ymin, ymax))


########################
# Pruned Decision Tree #
########################

# Number of folds for CV
fold_k = 10

# Place each sample into a fold
folds = sample(1:fold_k, nrow(input_df), replace=TRUE)

# Create empty error matrix
cv.errors.pdt.train = matrix(NA, fold_k, 1)
cv.errors.pdt.test = matrix(NA, fold_k, 1)

# Loop to perform the 10-fold CV
for (j in 1:fold_k) {
  # Fit the tree using all the folds except for j
  pdt.fit = tree(mental_state~., data=input_df[folds != j & train,])
  # cross validate for best sized tree
  cv.pdt = cv.tree(pdt.fit, FUN=prune.misclass, K=fold_k)
  # pick out best error
  size_error_df = data.frame(cv.pdt$size, cv.pdt$dev)
  best_error = (size_error_df[which.min(size_error_df$cv.pdt.dev),])$cv.pdt.size
  # use that tree with the best error
  prune.pdt = prune.misclass(pdt.fit, best=best_error)
  # predict on fold j (training)
  pdt.train.pred = predict(prune.pdt, input_df[folds == j & train,],
                           type="class")
  # predict on fold j (testing)
  pdt.test.pred = predict(prune.pdt, input_df[folds == j & !train,],
                           type="class")
  # Record training error
  cv.errors.pdt.train[j, 1] = 
    sum(
      (pdt.train.pred != input_df$mental_state[folds == j & train]),
      na.rm = TRUE
    ) / length(pdt.train.pred)
  # Record testing error
  cv.errors.pdt.test[j, 1] = 
    sum(
      (pdt.test.pred != input_df$mental_state[folds == j & !train]),
      na.rm = TRUE
    ) / length(pdt.test.pred)
}

# Print a sample tree
par(mfrow=c(1,1))
plot(pdt.fit)
text(pdt.fit, pretty=0, cex=0.75)

# Print errors
cv.errors.pdt.train
cv.errors.pdt.test

# Display average error
avg_error_train = mean(cv.errors.pdt.train[, 1])
avg_error_train
avg_error_test = mean(cv.errors.pdt.test[, 1])
avg_error_test

# Plot errors from CV runs
par(mfrow=c(1,2))
# Equalize the ymax and ymin limits so the graphs are comparable side by side
ymax = max(c(max(cv.errors.pdt.train[, 1]), max(cv.errors.pdt.test[, 1]))) + 0.03
ymin = min(c(min(cv.errors.pdt.train[, 1]), min(cv.errors.pdt.test[, 1]))) - 0.03
plot(c(1:fold_k), cv.errors.pdt.train[, 1], ylim=c(ymin, ymax))
plot(c(1:fold_k), cv.errors.pdt.test[, 1], ylim=c(ymin, ymax))


########################
# Multinomial Boosting #
########################

# Tree and depth values to test
trees = c(50, 100, 1000, 5000)
depths = c(1:4)

# Number of folds for CV
fold_k = 5

# Place each sample into a fold
folds = sample(1:fold_k, nrow(input_df), replace=TRUE)

# Create a matrix in which we will store the training results
cv.errors.boost.train.temp = matrix(NA, fold_k, length(trees))
cv.errors.boost.train = matrix(NA, length(depths), length(trees))

# Create a matrix in which we will store the testing results
cv.errors.boost.test.temp = matrix(NA, fold_k, length(trees))
cv.errors.boost.test = matrix(NA, length(depths), length(trees))

# Loop to perform the 10-fold CV
for (k in depths) {
  for (i in 1:length(trees)) {
    for (j in 1:fold_k) {
      # Since this is a multi-class problem, we set the distribution to 
      # "multinomial"
      # For binary classification, we would use "bernoulli"
      # n.trees: number of trees we want
      # interaction.depth: limits the depth of each tree
      boost.fit = gbm(mental_state~., data=input_df[folds != j & train,], 
                      distribution="multinomial", n.trees=trees[i],
                      interaction.depth=k)
      # Predict on remaining training data (fold == j)
      boost.train.pred = predict(boost.fit, newdata=input_df[folds == j & train,],
                                 n.trees=trees[i], type="response")
      # Predict on testing data (fold == j)
      boost.test.pred = predict(boost.fit, newdata=input_df[folds == j & !train,],
                                 n.trees=trees[i], type="response")
      # Get the training classification into the proper factor format
      boost.train.pred.class.int = apply(boost.train.pred, 1, which.max)
      boost.train.pred.class = as.factor(
        ifelse(
          boost.train.pred.class.int == 1, 
          "recording_of_seizure_activity", 
          ifelse(
            boost.train.pred.class.int == 2,
            "eeg_recorded_from_tumor_location",
            ifelse(
              boost.train.pred.class.int == 3,
              "eeg_recorded_from_healthy_location",
              ifelse(
                boost.train.pred.class.int == 4,
                "eyes_closed",
                "eyes_open"
              )
            )
          )
        )
      )
      # Get the testing classification into the proper factor format
      boost.test.pred.class.int = apply(boost.test.pred, 1, which.max)
      boost.test.pred.class = as.factor(
        ifelse(
          boost.test.pred.class.int == 1, 
          "recording_of_seizure_activity", 
          ifelse(
            boost.test.pred.class.int == 2,
            "eeg_recorded_from_tumor_location",
            ifelse(
              boost.test.pred.class.int == 3,
              "eeg_recorded_from_healthy_location",
              ifelse(
                boost.test.pred.class.int == 4,
                "eyes_closed",
                "eyes_open"
              )
            )
          )
        )
      )
      # Record training errors
      cv.errors.boost.train.temp[j, i] = 
        sum(
          (boost.train.pred.class != input_df$mental_state[folds == j & train]),
          na.rm = TRUE
        ) / length(boost.train.pred.class)
      # Record testing errors
      cv.errors.boost.test.temp[j, i] = 
        sum(
          (boost.test.pred.class != input_df$mental_state[folds == j & !train]),
          na.rm = TRUE
        ) / length(boost.test.pred.class)
    }
  }
  means_train = colMeans(cv.errors.boost.train.temp)
  means_test = colMeans(cv.errors.boost.test.temp)
  for (l in 1:length(means_train)) {
    cv.errors.boost.train[k, l] = means_train[l]
    cv.errors.boost.test[k, l] = means_test[l]
  }
}

# Print the errors
cv.errors.boost.train
cv.errors.boost.test

# Plot the train errors per depth
par(mfrow=c(2,2))
plot(trees, cv.errors.boost.train[1,])
plot(trees, cv.errors.boost.train[2,])
plot(trees, cv.errors.boost.train[3,])
plot(trees, cv.errors.boost.train[4,])

# Plot the test errors per depth
par(mfrow=c(2,2))
plot(trees, cv.errors.boost.test[1,])
plot(trees, cv.errors.boost.test[2,])
plot(trees, cv.errors.boost.test[3,])
plot(trees, cv.errors.boost.test[4,])

# Plot the errors for the depths (training and testing)
mins_train = c(
  cv.errors.boost.train[1,which.min(cv.errors.boost.train[1,])],
  cv.errors.boost.train[2,which.min(cv.errors.boost.train[2,])],
  cv.errors.boost.train[3,which.min(cv.errors.boost.train[3,])],
  cv.errors.boost.train[4,which.min(cv.errors.boost.train[4,])]
)
mins_test = c(
  cv.errors.boost.test[1,which.min(cv.errors.boost.test[1,])],
  cv.errors.boost.test[2,which.min(cv.errors.boost.test[2,])],
  cv.errors.boost.test[3,which.min(cv.errors.boost.test[3,])],
  cv.errors.boost.test[4,which.min(cv.errors.boost.test[4,])]
)

par(mfrow=c(1,2))
plot(depths, mins_train)
plot(depths, mins_test)

# Plot errors (3D)
par(mfrow=c(1,2))
persp(depths, trees, cv.errors.boost.train, theta=135, phi=0)
persp(depths, trees, cv.errors.boost.test, theta=135, phi=0)

######
# RF #
######

# Number of predictors
p = ncol(input_df) - 1

# Number of predictors to use (sqrt p = 178 ~= 13, p = 178 => bagging)
tries = c(5, 10, as.integer(sqrt(p)), 50, p)

# Number of trees to build in the forest
trees = c(50, 100, 1000, 5000)

# Number of folds for CV
fold_k = 5

# Place each sample into a fold
folds = sample(1:fold_k, nrow(input_df), replace=TRUE)

# Create a matrix in which we will store the results (training and testing)
# Temp matrix exists to hold intermediately computed results
cv.errors.rf.train.temp = matrix(NA, fold_k, length(tries))
cv.errors.rf.train = matrix(NA, length(trees), length(tries))
cv.errors.rf.test.temp = matrix(NA, fold_k, length(tries))
cv.errors.rf.test = matrix(NA, length(trees), length(tries))

# Loop to perform the 10-fold CV
for (k in 1:length(trees)) {
  for (i in 1:length(tries)) {
    for (j in 1:fold_k) {
      # Fit the model on the 9 training folds (folds != j) and the training set
      # (train == TRUE) with the given mtry and ntree values
      rf.fit = randomForest(mental_state~., data=input_df[folds != j & train,],
                            mtry=tries[i], ntree=trees[k])
      # Predict on the 1 testing fold (folds == j) and the training set (train
      # == TRUE)
      rf.train.pred = predict(rf.fit, newdata=input_df[folds == j & train,])
      # Predict on the 1 testing fold (folds == j) and the testing set (train ==
      # FALSE)
      rf.test.pred = predict(rf.fit, newdata=input_df[folds == j & !train,])
      # Get the training errors for the 10 folds
      cv.errors.rf.train.temp[j, i] =  
        sum(
          (rf.train.pred != input_df$mental_state[folds == j & train]),
          na.rm = TRUE
        ) / length(rf.train.pred)
      # Get the testing errors for the 10 folds
      cv.errors.rf.test.temp[j, i] =  
        sum(
          (rf.test.pred != input_df$mental_state[folds == j & !train]),
          na.rm = TRUE
        ) / length(rf.test.pred)
    }
  }
  # Average across the folds per tree length for training and testing
  means_train = colMeans(cv.errors.rf.train.temp)
  means_test = colMeans(cv.errors.rf.test.temp)
  for (l in 1:length(means_train)) {
    cv.errors.rf.train[k, l] = means_train[l]
    cv.errors.rf.test[k, l] = means_test[l]
  }
}

# Show Errors
cv.errors.rf.train
cv.errors.rf.test

# Plot errors
par(mfrow=c(1,2))
persp(trees, tries, cv.errors.rf.train, theta=135, phi=0)
persp(trees, tries, cv.errors.rf.test, theta=135, phi=0)


#######
# SVM #
#######

# Costs and gammas to test
costs  = c(0.001, 0.01, 0.1, 1, 5, 10, 100)
gammas = c(0.5, 1, 5, 10, 50)

# Support Vector Classifier
cv.errors.svc = tune(svm, mental_state~., data=input_df, kernel="linear",
                     ranges=list(cost=costs, gamma=gammas))

# CV error for each model
summary(cv.errors.svc)

# Obtain the best model automatically
svc.bestmod = cv.errors.svc$best.model
summary(svc.bestmod)

# Support Vector Machine
cv.errors.svm = tune(svm, mental_state~., data=input_df, kernel="radial",
                     ranges=list(cost=costs, gamma=gammas))

# CV error for each model
summary(cv.errors.svm)

# Obtain the best model automatically
svm.bestmod = cv.errors.svm$best.model
summary(svm.bestmod)