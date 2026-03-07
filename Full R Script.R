#Loading the required libraries

library(ggplot2)
library(tidyr)
library(dplyr)
library(scales)
library(FactoMineR)
library(factoextra)
library(Rtsne)
library(uwot)
library(h2o)
library(caret)
library(rBayesianOptimization)
library(MLmetrics)
library(adabag)
library(rpart)
library(reshape2)
library(xgboost)
library(e1071)
library(randomForest)
library(nnet)

#Importing the dataset

dataset <- read.csv("SPHALERITE_CHEMISTRY2025.csv")

#Remove rows with missing values

dataset <- na.omit(dataset)

############### Part 1: Box Plots #######################

#Selecting the relevant columns

dataset_new <- dataset[4:16]

unique_categories <- dataset_new %>% distinct(dataset_new$Deposit.type)
print(unique_categories)

#Convert data from wide to long format

dataset_long <- dataset_new %>% gather(key = "Element", value = "Concentration", -Deposit.type)

#Remove non-finite values (NA, NaN, Inf) and zero values (log10 can't handle 0)

dataset_long <- dataset_long %>% filter(is.finite(Concentration) & Concentration > 0)

elements <- c("Fe", "Mn", "Co", "Cu", "Ga", "Ge", "Ag", "Cd", "In", "Sn", "Sb", "Pb")

#Modify element names to include "(ppm)" and format correctly for facet labels

dataset_long$Element <- factor(dataset_long$Element, levels = elements, labels = paste0(elements, " (ppm)"))

#Box and whisker plots grouped by deposit type

ggplot(dataset_long, aes(x = Deposit.type, y = Concentration, fill = Deposit.type)) +
  stat_boxplot(geom = "errorbar", width = 0.3, linewidth = 0.2) + # Whiskers show min/max
  geom_boxplot(outlier.shape = NA, color = "black", width = 0.6, linewidth = 0.3) + # Thin median line
  stat_summary(fun = mean, geom = "point", shape = 21, size = 2, fill = "gray") + # Small mean point
  facet_wrap(~Element, scales = "free_y", ncol = 3, strip.position = "left") + # Exactly 3 plots per row
  scale_y_log10(labels = trans_format("log10", math_format(10^.x))) + # Exponent format (10^-1, 10^2, etc.)
  scale_x_discrete(expand = expansion(mult = 0.2)) + # Centers boxplots
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 9, color = "black", angle = 45, hjust = 1), # Set x-axis labels size 9, black
    axis.text.y = element_text(size = 9, color = "black"), # Set y-axis labels size 9, black
    axis.title = element_blank(), # Remove overall x and y labels
    strip.text.y.left = element_text(face = "bold", size = 9, angle = 90, hjust = 0.5, vjust = 1), # Element labels further left
    strip.placement = "outside", # Ensure element names appear outside the tick labels
    panel.spacing = unit(0.4, "lines"), # Reduce space between subplots
    plot.title = element_blank(), # Remove title
    legend.position = "none",
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.3), # Black border around each plot
    panel.grid.major = element_blank(), # Remove major grid lines
    panel.grid.minor = element_blank(),# Remove minor grid lines
    axis.ticks.y = element_line(color = "black"),# Ensure y-axis tick marks are visible for each subplot
    aspect.ratio = 1 # Prevent excessive elongation
  )

############ Part 2: Dimensionality reduction analysis

############ Pre-processing 

dataset_ml <- dataset[, 4:16]
dataset_ml$Deposit.type <- as.factor(dataset_ml$Deposit.type)

features <- dataset_ml[,1:12]
labels <- dataset_ml$Deposit.type

############ Outlier Removal Functions ############

#Used before dimensionality reduction


remove_outliers_strict <- function(df){
  
  Q1 <- apply(df,2,quantile,0.25)
  Q3 <- apply(df,2,quantile,0.75)
  IQR <- Q3 - Q1
  
  lower <- Q1 - 1.5*IQR
  upper <- Q3 + 1.5*IQR
  
  keep <- apply(df,1,function(row) all(row >= lower & row <= upper))
  
  list(data = df[keep,], indices = which(keep))
}

# Used before ML models

remove_outliers_ml <- function(df, max_outlier_cols = 2){
  
  Q1 <- apply(df,2,quantile,0.25)
  Q3 <- apply(df,2,quantile,0.75)
  IQR <- Q3 - Q1
  
  lower <- Q1 - 1.5*IQR
  upper <- Q3 + 1.5*IQR
  
  keep <- apply(df,1,function(row){
    sum(row < lower | row > upper) <= max_outlier_cols
  })
  
  list(data = df[keep,], indices = which(keep))
}

outlier_result <- remove_outliers_strict(features)
features_clean <- outlier_result$data
labels_clean <- labels[outlier_result$indices]

# CLR Transformation

clr_transform <- function(x){
  x <- as.matrix(x)
  if(any(x<=0)) stop("CLR requires positive values")
  log_x <- log(x)
  gm <- rowMeans(log_x)
  sweep(log_x,1,gm,"-")
}

features_clr <- clr_transform(features_clean + 1e-6)

# Standardize

features_scaled <- scale(features_clr)

############ A: PCA

set.seed(123)

pca_result <- PCA(features_scaled, scale.unit = TRUE, ncp = 3, graph = FALSE)

pca_df <- data.frame(
  PC1 = pca_result$ind$coord[,1],
  PC2 = pca_result$ind$coord[,2],
  Deposit.type = labels_clean
)

ggplot(pca_df,aes(PC1,PC2,color=Deposit.type))+
  geom_point(size=1)+
  labs(
    title="PCA: Outlier-filtered CLR-transformed data",
    x=paste0("PC1 (",round(pca_result$eig[1,2],1),"%)"),
    y=paste0("PC2 (",round(pca_result$eig[2,2],1),"%)")
  )+
  xlim(-4,4)+
  ylim(-4,4)+
  theme_minimal(base_size=9)+
  theme(
    panel.grid=element_blank(),
    panel.border=element_rect(color="black",fill=NA),
    axis.ticks=element_line(color="black"),
    axis.ticks.length=unit(0.15,"cm"),
    legend.position="none",
    plot.title=element_text(hjust=0.5)
  )

############ B: t-SNE

# Remove duplicate rows (required for Rtsne)

unique_rows <- !duplicated(features_scaled)
features_tsne <- features_scaled[unique_rows,]
labels_tsne <- labels_clean[unique_rows]

set.seed(123)

tsne_result <- Rtsne(
  features_tsne,
  dims=2,
  perplexity=40,
  verbose=TRUE,
  max_iter=1000
)

tsne_df <- data.frame(
  X = tsne_result$Y[,1],
  Y = tsne_result$Y[,2],
  Deposit.type = labels_tsne
)

ggplot(tsne_df,aes(X,Y,color=Deposit.type))+
  geom_point(size=1)+
  labs(x="t-SNE1",y="t-SNE2")+
  theme_minimal(base_size=9)+
  theme(
    legend.title=element_blank(),
    panel.border=element_rect(color="black",fill=NA),
    axis.ticks=element_line(color="black"),
    axis.ticks.length=unit(0.15,"cm"),
    panel.grid=element_blank(),
    legend.position="none"
  )

############ C: UMAP

set.seed(123)

umap_result <- umap(features_scaled)

umap_df <- data.frame(
  UMAP1 = umap_result[, 1],
  UMAP2 = umap_result[, 2],
  Deposit.type = labels_clean
)

ggplot(umap_df,aes(UMAP1,UMAP2,color=Deposit.type))+
  geom_point(size=1)+
  labs(
    title="UMAP: Outlier-filtered CLR-transformed data",
    x="UMAP Dimension 1",
    y="UMAP Dimension 2"
  )+
  theme_minimal(base_size=9)+
  theme(
    legend.title=element_blank(),
    panel.border=element_rect(color="black",fill=NA),
    axis.ticks.length=unit(0.15,"cm"),
    axis.ticks=element_line(color="black"),
    panel.grid=element_blank(),
    legend.position="none"
  )

############ D: Autoencoder

# Start H2O

h2o.init(
  nthreads = 1,
  max_mem_size = "4G"
)

# Initialize H2O

if (!h2o::h2o.clusterIsUp()) {
  h2o.init(
    nthreads = 1,
    max_mem_size = "4G",
    enable_assertions = TRUE
  )
}

# Convert to H2O frame

h2o_data <- as.h2o(features_scaled)

# Train autoencoder

autoencoder <- h2o.deeplearning(
  x = colnames(h2o_data),
  training_frame = h2o_data,
  autoencoder = TRUE,
  hidden = c(8,4,8),
  activation = "Tanh",
  epochs = 200,
  l1 = 1e-5,
  seed = 123,
  reproducible = TRUE
)

# Extract latent features

ae_features <- as.data.frame(
  h2o.deepfeatures(autoencoder, h2o_data, layer = 2)
)

ae_features$Deposit.type <- labels_clean

# Plot Autoencoder projection

ggplot(ae_features, aes(x = DF.L2.C1, y = DF.L2.C2, color = Deposit.type)) +
  geom_point(size = 0.8) +
  theme_minimal(base_size = 9) +
  labs(
    x = "Autoencoder Dimension 1",
    y = "Autoencoder Dimension 2"
  ) +
  theme(
    legend.position = "none",
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8),
    axis.ticks = element_line(color = "black"),
    panel.grid = element_blank()
  )

############ E: Autoencoder+t-SNE

# Remove duplicates

ae_unique <- !duplicated(ae_features[,1:2])
ae_features_filtered <- ae_features[ae_unique, ]

labels_filtered <- ae_features_filtered$Deposit.type

# Matrix input for t-SNE

tsne_input <- as.matrix(ae_features_filtered[,1:2])

set.seed(123)

# Run t-SNE on Autoencoder

tsne_result_ae <- Rtsne(
  tsne_input,
  dims = 2,
  perplexity = 50,
  verbose = TRUE,
  max_iter = 1000
)

# Prepare data for plotting

tsne_df_ae <- data.frame(
  X = tsne_result_ae$Y[,1],
  Y = tsne_result_ae$Y[,2],
  Deposit.type = labels_filtered
)

# Plot Hybrid Model

ggplot(tsne_df_ae, aes(X, Y, color = Deposit.type)) +
  geom_point(size = 1) +
  theme_minimal(base_size = 9) +
  labs(
    title = "t-SNE on Autoencoder Latent Features",
    x = "t-SNE Dimension 1",
    y = "t-SNE Dimension 2"
  ) +
  theme(
    legend.title = element_blank(),
    plot.title = element_text(hjust = 0.5),
    panel.border = element_rect(color = "black", fill = NA),
    axis.ticks = element_line(color = "black"),
    axis.ticks.length = unit(0.15, "cm"),
    legend.position = "none",
    panel.grid = element_blank()
  )

############ F: Autoencoder + UMAP Hybrid

# Run UMAP on Autoencoder

set.seed(123)

umap_result_ae <- uwot::umap(ae_features[, 1:2])

# Prepare data for plotting

umap_df_ae <- data.frame(
  UMAP1 = umap_result_ae[, 1],
  UMAP2 = umap_result_ae[, 2],
  Deposit.type = ae_features$Deposit.type
)

# Plot Hybrid Model

ggplot(umap_df_ae, aes(UMAP1,UMAP2,color=Deposit.type))+
  geom_point(size=1)+
  theme_minimal(base_size=9)+
  labs(
    title="UMAP on Autoencoder Latent Features",
    x="UMAP Dimension 1",
    y="UMAP Dimension 2"
  )+
  theme(
    legend.title=element_blank(),
    plot.title=element_text(hjust=0.5),
    panel.border=element_rect(color="black",fill=NA),
    axis.ticks=element_line(color="black"),
    axis.ticks.length=unit(0.15,"cm"),
    legend.position="none",
    panel.grid=element_blank()
  )

############ Part 4: Machine Learning Models############

# **LESS STRICT OUTLIER REMOVAL BEFORE ML**

outlier_result_ml <- remove_outliers_ml(features)

features_ml <- outlier_result_ml$data
labels_ml <- labels[outlier_result_ml$indices]


# CLR Transformation
features_clr_ml <- clr_transform(features_ml + 1e-6)


# Standardization
features_scaled_ml <- scale(features_clr_ml)


############ Train/Test Split

set.seed(123)

train_idx <- createDataPartition(labels_ml, p = 0.8, list = FALSE)

X_train <- features_scaled_ml[train_idx, ]
y_train <- labels_ml[train_idx]

X_test <- features_scaled_ml[-train_idx, ]
y_test <- labels_ml[-train_idx]

num_class <- length(levels(y_train))

############ A: XGboost

# Bayesian Optimization

xgb_cv_bayes <- function(eta, max_depth, subsample, colsample_bytree){
  
  param <- list(
    objective = "multi:softmax",
    num_class = num_class,
    eval_metric = "merror",
    eta = eta,
    max_depth = as.integer(max_depth),
    subsample = subsample,
    colsample_bytree = colsample_bytree
  )
  
  dtrain <- xgb.DMatrix(data = X_train, label = as.integer(y_train)-1)
  
  cv <- xgb.cv(
    params = param,
    data = dtrain,
    nrounds = 100,
    nfold = 5,
    verbose = 0,
    early_stopping_rounds = 10
  )
  
  list(Score = -min(cv$evaluation_log$test_merror_mean), Pred = 0)
}

set.seed(123)

bayes_opt_result <- BayesianOptimization(
  FUN = xgb_cv_bayes,
  bounds = list(
    eta = c(0.01,0.3),
    max_depth = c(3L,10L),
    subsample = c(0.5,1),
    colsample_bytree = c(0.5,1)
  ),
  init_points = 10,
  n_iter = 20
)

best <- bayes_opt_result$Best_Par

dtrain <- xgb.DMatrix(data = X_train, label = as.integer(y_train)-1)

final_model <- xgboost(
  data = dtrain,
  objective = "multi:softmax",
  num_class = num_class,
  nrounds = 100,
  eta = best["eta"],
  max_depth = as.integer(best["max_depth"]),
  subsample = best["subsample"],
  colsample_bytree = best["colsample_bytree"],
  verbose = 0
)

# Prediction

dtest <- xgb.DMatrix(data = X_test)

preds <- predict(final_model, dtest)

preds_factor <- factor(preds,
                       levels = 0:(num_class-1),
                       labels = levels(y_train))

# Evaluation Metrics

conf <- caret::confusionMatrix(preds_factor, y_test)

acc <- as.numeric(conf$overall["Accuracy"])
kappa <- as.numeric(conf$overall["Kappa"])

precision <- MLmetrics::Precision(y_pred = preds_factor, y_true = y_test, positive = NULL)
recall <- MLmetrics::Recall(y_pred = preds_factor, y_true = y_test, positive = NULL)
f1 <- MLmetrics::F1_Score(y_pred = preds_factor, y_true = y_test, positive = NULL)

specificity <- mean(conf$byClass[, "Specificity"], na.rm = TRUE)

compute_multiclass_mcc <- function(conf_matrix) {
  n <- sum(conf_matrix)
  rowsums <- rowSums(conf_matrix)
  colsums <- colSums(conf_matrix)
  diag_vals <- diag(conf_matrix)
  s <- sum(diag_vals)
  p0 <- sum(rowsums * colsums)
  numerator <- (n * s) - p0
  denominator <- sqrt((n^2 - sum(colsums^2)) * (n^2 - sum(rowsums^2)))
  if (denominator == 0) return(NA)
  numerator / denominator
}

conf_mat_table <- table(y_test, preds_factor)
mcc <- compute_multiclass_mcc(as.matrix(conf_mat_table))

cat("Evaluation Metrics:\n")
cat(sprintf("Accuracy       : %.4f\n", acc))
cat(sprintf("Precision      : %.4f\n", precision))
cat(sprintf("Recall         : %.4f\n", recall))
cat(sprintf("F1 Score       : %.4f\n", f1))
cat(sprintf("Cohen's Kappa  : %.4f\n", kappa))
cat(sprintf("Specificity    : %.4f\n", specificity))
cat(sprintf("MCC            : %.4f\n", mcc))

# Confusion Matrix

cm <- table(y_test, preds_factor)

cm_melted <- as.data.frame.table(cm)
colnames(cm_melted) <- c("True","Predicted","Count")

ggplot(cm_melted,aes(Predicted,True,fill=Count))+
  geom_tile(color="black")+
  geom_text(aes(label=Count),size=3)+
  scale_fill_gradient(low="white",high="steelblue")+
  labs(title="Confusion Matrix – XGBoost")+
  theme_minimal(base_size=9)+
  theme(
    plot.title=element_text(hjust=0.5),
    axis.title=element_blank(),
    axis.text=element_blank(),
    axis.ticks=element_blank(),
    panel.border=element_rect(color="black",fill=NA),
    panel.grid=element_blank(),
    legend.position="none"
  )

############ B: Bagged Decision Trees

train_df <- data.frame(X_train)
train_df$Deposit.type <- y_train

# Bayesian Optimization
bagged_cv_bayes <- function(nbagg, cp, maxdepth){
  
  nbagg <- as.integer(nbagg)
  maxdepth <- as.integer(maxdepth)
  
  folds <- createFolds(train_df$Deposit.type, k = 5)
  
  accs <- sapply(folds, function(val_idx){
    
    model <- bagging(
      Deposit.type ~ .,
      data = train_df[-val_idx, ],
      nbagg = nbagg,
      control = rpart.control(cp = cp, maxdepth = maxdepth)
    )
    
    preds <- predict(model, newdata = train_df[val_idx, ], type = "class")
    
    mean(preds == train_df$Deposit.type[val_idx])
  })
  
  list(Score = mean(accs), Pred = 0)
}

set.seed(123)

bayes_opt_result <- BayesianOptimization(
  FUN = bagged_cv_bayes,
  bounds = list(
    nbagg = c(50L,150L),
    cp = c(0.001,0.01),
    maxdepth = c(5L,15L)
  ),
  init_points = 10,
  n_iter = 20,
  acq = "ucb",
  kappa = 2.576
)

best_nbagg <- as.integer(bayes_opt_result$Best_Par["nbagg"])
best_cp <- bayes_opt_result$Best_Par["cp"]
best_maxdepth <- as.integer(bayes_opt_result$Best_Par["maxdepth"])

# Train Final Bagging Model

final_model <- bagging(
  Deposit.type ~ .,
  data = train_df,
  nbagg = best_nbagg,
  control = rpart.control(cp = best_cp, maxdepth = best_maxdepth)
)

# Predictions

test_df <- data.frame(X_test)

preds <- predict(final_model, newdata = test_df, type = "class")

y_pred <- factor(preds, levels = levels(y_test))

# Evaluation Metrics

conf <- confusionMatrix(y_pred, y_test)

acc <- as.numeric(conf$overall["Accuracy"])
kappa <- as.numeric(conf$overall["Kappa"])

precision <- Precision(y_pred = y_pred, y_true = y_test, positive = NULL)
recall <- Recall(y_pred = y_pred, y_true = y_test, positive = NULL)
f1 <- F1_Score(y_pred = y_pred, y_true = y_test, positive = NULL)

specificity <- mean(conf$byClass[, "Specificity"], na.rm = TRUE)

compute_multiclass_mcc <- function(conf_matrix) {
  n <- sum(conf_matrix)
  rowsums <- rowSums(conf_matrix)
  colsums <- colSums(conf_matrix)
  diag_vals <- diag(conf_matrix)
  s <- sum(diag_vals)
  p0 <- sum(rowsums * colsums)
  numerator <- (n * s) - p0
  denominator <- sqrt((n^2 - sum(colsums^2)) * (n^2 - sum(rowsums^2)))
  if (denominator == 0) return(NA)
  return(numerator / denominator)
}

conf_mat_table <- table(y_test, y_pred)
mcc <- compute_multiclass_mcc(as.matrix(conf_mat_table))

cat("Evaluation Metrics:\n")
cat(sprintf("Accuracy       : %.4f\n", acc))
cat(sprintf("Precision      : %.4f\n", precision))
cat(sprintf("Recall         : %.4f\n", recall))
cat(sprintf("F1 Score       : %.4f\n", f1))
cat(sprintf("Cohen's Kappa  : %.4f\n", kappa))
cat(sprintf("Specificity    : %.4f\n", specificity))
cat(sprintf("MCC            : %.4f\n", mcc))

# Confusion Matrix Plot

conf_mat_table <- table(y_test, y_pred)

cm_melted <- as.data.frame.table(conf_mat_table)
colnames(cm_melted) <- c("True","Predicted","Count")

ggplot(cm_melted,aes(Predicted,True,fill=Count))+
  geom_tile(color="black")+
  geom_text(aes(label=Count),size=3)+
  scale_fill_gradient(low="white",high="steelblue")+
  labs(title="Confusion Matrix – Bagged Trees")+
  theme_minimal(base_size=9)+
  theme(
    axis.title=element_blank(),
    axis.text=element_blank(),
    axis.ticks=element_blank(),
    panel.border=element_rect(color="black",fill=NA),
    panel.grid=element_blank(),
    legend.position="none"
  )

############ C: Adaboost

train_df <- data.frame(X_train)
train_df$Deposit.type <- y_train

# Bayesian Optimization for AdaBoost

adaboost_cv_bayes <- function(mfinal, maxdepth, cp){
  
  mfinal <- floor(mfinal)
  maxdepth <- floor(maxdepth)
  
  folds <- createFolds(train_df$Deposit.type, k = 5)
  
  accs <- sapply(folds, function(val_idx){
    
    model <- boosting(
      Deposit.type ~ .,
      data = train_df[-val_idx, ],
      mfinal = mfinal,
      control = rpart.control(maxdepth = maxdepth, cp = cp)
    )
    
    preds <- predict.boosting(model, newdata = train_df[val_idx, ])
    
    mean(preds$class == train_df$Deposit.type[val_idx])
  })
  
  list(Score = mean(accs), Pred = 0)
}

set.seed(123)

bayes_opt_result <- BayesianOptimization(
  FUN = adaboost_cv_bayes,
  bounds = list(
    mfinal = c(50L,150L),
    maxdepth = c(3L,15L),
    cp = c(0.001,0.01)
  ),
  init_points = 10,
  n_iter = 20,
  acq = "ucb",
  kappa = 2.576,
  verbose = TRUE
)

best_mfinal <- floor(bayes_opt_result$Best_Par["mfinal"])
best_maxdepth <- floor(bayes_opt_result$Best_Par["maxdepth"])
best_cp <- bayes_opt_result$Best_Par["cp"]

# Train Final AdaBoost Model

final_model <- boosting(
  Deposit.type ~ .,
  data = train_df,
  mfinal = best_mfinal,
  control = rpart.control(maxdepth = best_maxdepth, cp = best_cp)
)

# Predictions on Test Set

test_df <- data.frame(X_test)

preds <- predict.boosting(final_model, newdata = test_df)

y_pred <- factor(preds$class, levels = levels(y_test))

# Evaluation Metrics

conf <- confusionMatrix(y_pred, y_test)

acc <- as.numeric(conf$overall["Accuracy"])
kappa <- as.numeric(conf$overall["Kappa"])

f1 <- F1_Score(y_pred = y_pred, y_true = y_test, positive = NULL)
precision <- Precision(y_pred = y_pred, y_true = y_test, positive = NULL)
recall <- Recall(y_pred = y_pred, y_true = y_test, positive = NULL)

specificity <- mean(conf$byClass[,"Specificity"], na.rm = TRUE)


# MCC calculation

compute_multiclass_mcc <- function(conf_matrix){
  
  n <- sum(conf_matrix)
  
  rowsums <- rowSums(conf_matrix)
  colsums <- colSums(conf_matrix)
  
  diag_vals <- diag(conf_matrix)
  
  s <- sum(diag_vals)
  
  p0 <- sum(rowsums * colsums)
  
  numerator <- (n*s) - p0
  
  denominator <- sqrt((n^2 - sum(colsums^2)) * (n^2 - sum(rowsums^2)))
  
  if(denominator == 0) return(NA)
  
  numerator / denominator
}

conf_mat_table <- table(y_test, y_pred)

mcc <- compute_multiclass_mcc(as.matrix(conf_mat_table))


cat("\nEvaluation Metrics\n")

cat(sprintf("Accuracy      : %.4f\n", acc))
cat(sprintf("Precision     : %.4f\n", precision))
cat(sprintf("Recall        : %.4f\n", recall))
cat(sprintf("F1 Score      : %.4f\n", f1))
cat(sprintf("Cohen Kappa   : %.4f\n", kappa))
cat(sprintf("Specificity   : %.4f\n", specificity))
cat(sprintf("MCC           : %.4f\n", mcc))

# Confusion Matrix

cm_melted <- melt(as.matrix(conf_mat_table))

colnames(cm_melted) <- c("True","Predicted","Count")

ggplot(cm_melted, aes(Predicted,True,fill=Count))+
  geom_tile(color="black")+
  geom_text(aes(label=Count),size=3)+
  scale_fill_gradient(low="white",high="steelblue")+
  labs(title = "Confusion Matrix – AdaBoost") +
  theme_minimal(base_size=9)+
  theme(
    axis.title = element_blank(),
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    panel.border = element_rect(color="black",fill=NA),
    panel.grid = element_blank(),
    legend.position = "none"
  )

############ D: SVM

# Bayesian Optimization for SVM

svm_cv_bayes <- function(cost, gamma){
  
  cost <- 2^cost
  gamma <- 2^gamma
  
  folds <- createFolds(y_train, k = 5)
  
  accs <- sapply(folds, function(val_idx){
    
    model <- svm(
      x = X_train[-val_idx, ],
      y = y_train[-val_idx],
      cost = cost,
      gamma = gamma,
      kernel = "radial"
    )
    
    preds <- predict(model, X_train[val_idx, ])
    
    mean(preds == y_train[val_idx])
  })
  
  list(Score = mean(accs), Pred = 0)
}

set.seed(123)

bayes_opt_result <- BayesianOptimization(
  FUN = svm_cv_bayes,
  bounds = list(
    cost = c(-5,15),
    gamma = c(-15,3)
  ),
  init_points = 10,
  n_iter = 20,
  acq = "ucb",
  kappa = 2.576
)

best_cost <- 2^bayes_opt_result$Best_Par["cost"]
best_gamma <- 2^bayes_opt_result$Best_Par["gamma"]

cat("Best Hyperparameters:\n")
cat("cost =", best_cost,"\n")
cat("gamma =", best_gamma,"\n")

# Train Final SVM Model
svm_model <- svm(
  x = X_train,
  y = y_train,
  cost = best_cost,
  gamma = best_gamma,
  kernel = "radial"
)

# Predictions

preds <- predict(svm_model, X_test)

y_pred <- factor(preds, levels = levels(y_test))

# Evaluation Metrics

conf <- caret::confusionMatrix(y_pred, y_test)

acc <- as.numeric(conf$overall["Accuracy"])
kappa <- as.numeric(conf$overall["Kappa"])

precision <- MLmetrics::Precision(y_pred = y_pred, y_true = y_test, positive = NULL)
recall <- MLmetrics::Recall(y_pred = y_pred, y_true = y_test, positive = NULL)
f1 <- MLmetrics::F1_Score(y_pred = y_pred, y_true = y_test, positive = NULL)

specificity <- mean(conf$byClass[, "Specificity"], na.rm = TRUE)

compute_multiclass_mcc <- function(conf_matrix) {
  n <- sum(conf_matrix)
  rowsums <- rowSums(conf_matrix)
  colsums <- colSums(conf_matrix)
  diag_vals <- diag(conf_matrix)
  s <- sum(diag_vals)
  p0 <- sum(rowsums * colsums)
  numerator <- (n * s) - p0
  denominator <- sqrt((n^2 - sum(colsums^2)) * (n^2 - sum(rowsums^2)))
  if (denominator == 0) return(NA)
  numerator / denominator
}

conf_mat_table <- table(y_test, y_pred)
mcc <- compute_multiclass_mcc(as.matrix(conf_mat_table))

cat("Evaluation Metrics:\n")
cat(sprintf("Accuracy       : %.4f\n", acc))
cat(sprintf("Precision      : %.4f\n", precision))
cat(sprintf("Recall         : %.4f\n", recall))
cat(sprintf("F1 Score       : %.4f\n", f1))
cat(sprintf("Cohen's Kappa  : %.4f\n", kappa))
cat(sprintf("Specificity    : %.4f\n", specificity))
cat(sprintf("MCC            : %.4f\n", mcc))

# Confusion Matrix

conf_mat_table <- table(y_test, y_pred)

cm_melted <- as.data.frame.table(conf_mat_table)
colnames(cm_melted) <- c("True","Predicted","Count")

ggplot(cm_melted,aes(Predicted,True,fill=Count))+
  geom_tile(color="black")+
  geom_text(aes(label=Count),size=3)+
  scale_fill_gradient(low="white",high="steelblue")+
  labs(title="Confusion Matrix – SVM")+
  theme_minimal(base_size=9)+
  theme(
    axis.title=element_blank(),
    axis.text=element_blank(),
    axis.ticks=element_blank(),
    panel.border=element_rect(color="black",fill=NA),
    panel.grid=element_blank(),
    legend.position="none"
  )

############ E: Random Forest

# Bayesian Optimization for RF

rf_cv_bayes <- function(mtry, nodesize, maxnodes){
  
  mtry <- floor(mtry)
  nodesize <- floor(nodesize)
  maxnodes <- floor(maxnodes)
  
  folds <- createFolds(y_train, k = 5, returnTrain = TRUE)
  
  accs <- sapply(folds, function(train_idx_fold){
    
    model <- randomForest(
      x = X_train[train_idx_fold, ],
      y = y_train[train_idx_fold],
      mtry = mtry,
      ntree = 500,
      nodesize = nodesize,
      maxnodes = maxnodes,
      replace = TRUE
    )
    
    preds <- predict(model, X_train[-train_idx_fold, ])
    
    mean(preds == y_train[-train_idx_fold])
  })
  
  list(Score = mean(accs), Pred = 0)
}

set.seed(123)

bayes_opt_result <- BayesianOptimization(
  FUN = rf_cv_bayes,
  bounds = list(
    mtry = c(3L,8L),
    nodesize = c(1L,5L),
    maxnodes = c(100L,500L)
  ),
  init_points = 10,
  n_iter = 20,
  acq = "ucb",
  kappa = 2.576
)

best_mtry <- floor(bayes_opt_result$Best_Par["mtry"])
best_nodesize <- floor(bayes_opt_result$Best_Par["nodesize"])
best_maxnodes <- floor(bayes_opt_result$Best_Par["maxnodes"])

cat("Best RF Hyperparameters:\n")
cat("mtry =", best_mtry,"\n")
cat("nodesize =", best_nodesize,"\n")
cat("maxnodes =", best_maxnodes,"\n")

# Train Final RF Model

rf_model <- randomForest(
  x = X_train,
  y = y_train,
  mtry = best_mtry,
  ntree = 500,
  nodesize = best_nodesize,
  maxnodes = best_maxnodes,
  replace = TRUE
)

# Predictions

preds <- predict(rf_model, X_test)

y_pred <- factor(preds, levels = levels(y_test))

# Evaluation Metrics

conf <- confusionMatrix(y_pred, y_test)

acc <- as.numeric(conf$overall["Accuracy"])
kappa <- as.numeric(conf$overall["Kappa"])

precision <- Precision(y_pred = y_pred, y_true = y_test)
recall <- Recall(y_pred = y_pred, y_true = y_test)
f1 <- F1_Score(y_pred = y_pred, y_true = y_test)

specificity <- mean(conf$byClass[,"Specificity"], na.rm = TRUE)

conf_mat_table <- table(y_test, y_pred)
mcc <- compute_multiclass_mcc(as.matrix(conf_mat_table))

cat("\nEvaluation Metrics – Random Forest\n")
cat(sprintf("Accuracy       : %.4f\n", acc))
cat(sprintf("Precision      : %.4f\n", precision))
cat(sprintf("Recall         : %.4f\n", recall))
cat(sprintf("F1 Score       : %.4f\n", f1))
cat(sprintf("Cohen's Kappa  : %.4f\n", kappa))
cat(sprintf("Specificity    : %.4f\n", specificity))
cat(sprintf("MCC            : %.4f\n", mcc))

# Confusion Matrix Plot

cm_melted <- as.data.frame.table(conf_mat_table)
colnames(cm_melted) <- c("True","Predicted","Count")

ggplot(cm_melted,aes(Predicted,True,fill=Count))+
  geom_tile(color="black")+
  geom_text(aes(label=Count),size=3)+
  scale_fill_gradient(low="white",high="steelblue")+
  labs(title="Confusion Matrix – Random Forest")+
  theme_minimal(base_size=9)+
  theme(
    axis.title=element_blank(),
    axis.text=element_blank(),
    axis.ticks=element_blank(),
    panel.border=element_rect(color="black",fill=NA),
    panel.grid=element_blank(),
    legend.position="none"
  )

############ F: MLP

# Bayesian Optimization for MLP

mlp_cv_bayes <- function(size, decay) {
  
  size <- round(size)
  decay <- 10^decay
  
  folds <- createFolds(y_train, k = 5)
  
  accs <- sapply(folds, function(val_idx){
    
    model <- nnet(
      X_train[-val_idx, ],
      class.ind(y_train[-val_idx]),
      size = size,
      decay = decay,
      maxit = 500,
      trace = FALSE,
      linout = FALSE,
      softmax = TRUE
    )
    
    preds <- predict(model, X_train[val_idx, ], type = "class")
    
    mean(preds == y_train[val_idx])
  })
  
  list(Score = mean(accs), Pred = 0)
}

set.seed(123)

bayes_opt_result <- BayesianOptimization(
  FUN = mlp_cv_bayes,
  bounds = list(
    size = c(1,20),
    decay = c(-6,-1)
  ),
  init_points = 10,
  n_iter = 20,
  acq = "ucb",
  kappa = 2.576
)

best_size <- round(bayes_opt_result$Best_Par["size"])
best_decay <- 10^bayes_opt_result$Best_Par["decay"]

cat("Best MLP Hyperparameters:\n")
cat("size =", best_size,"\n")
cat("decay =", best_decay,"\n")


# Train Final MLP Model

mlp_model <- nnet(
  X_train,
  class.ind(y_train),
  size = best_size,
  decay = best_decay,
  maxit = 500,
  trace = FALSE,
  linout = FALSE,
  softmax = TRUE
)

# Predictions
pred_probs <- predict(mlp_model, X_test, type = "raw")

preds <- factor(
  colnames(pred_probs)[apply(pred_probs,1,which.max)],
  levels = levels(y_train)
)

# Evaluation Metrics

conf <- confusionMatrix(preds, y_test)

acc <- as.numeric(conf$overall["Accuracy"])
kappa <- as.numeric(conf$overall["Kappa"])

precision <- Precision(y_pred = preds, y_true = y_test)
recall <- Recall(y_pred = preds, y_true = y_test)
f1 <- F1_Score(y_pred = preds, y_true = y_test)

specificity <- mean(conf$byClass[,"Specificity"], na.rm = TRUE)

conf_mat_table <- table(y_test, preds)
mcc <- compute_multiclass_mcc(as.matrix(conf_mat_table))

cat("\nEvaluation Metrics – MLP\n")

cat(sprintf("Accuracy       : %.4f\n", acc))
cat(sprintf("Precision      : %.4f\n", precision))
cat(sprintf("Recall         : %.4f\n", recall))
cat(sprintf("F1 Score       : %.4f\n", f1))
cat(sprintf("Cohen's Kappa  : %.4f\n", kappa))
cat(sprintf("Specificity    : %.4f\n", specificity))
cat(sprintf("MCC            : %.4f\n", mcc))


# Confusion Matrix Plot

cm_melted <- as.data.frame.table(conf_mat_table)
colnames(cm_melted) <- c("True","Predicted","Count")

ggplot(cm_melted,aes(Predicted,True,fill=Count))+
  geom_tile(color="black")+
  geom_text(aes(label=Count),size=3)+
  scale_fill_gradient(low="white",high="steelblue")+
  labs(title="Confusion Matrix – MLP")+
  theme_minimal(base_size=9)+
  theme(
    axis.title=element_blank(),
    axis.text=element_blank(),
    axis.ticks=element_blank(),
    panel.border=element_rect(color="black",fill=NA),
    panel.grid=element_blank(),
    legend.position="none"
  )

############ G: Naive Bayes

# Bayesian Optimization for Naive Bayes

nb_cv_bayes <- function(laplace) {
  
  folds <- createFolds(y_train, k = 5, returnTrain = TRUE)
  
  accs <- sapply(folds, function(train_idx_fold){
    
    model <- naiveBayes(
      x = X_train[train_idx_fold, ],
      y = y_train[train_idx_fold],
      laplace = laplace
    )
    
    preds <- predict(model, X_train[-train_idx_fold, ])
    
    mean(preds == y_train[-train_idx_fold])
  })
  
  list(Score = mean(accs), Pred = 0)
}

set.seed(123)

bayes_opt_result <- BayesianOptimization(
  FUN = nb_cv_bayes,
  bounds = list(
    laplace = c(0,5)
  ),
  init_points = 10,
  n_iter = 20,
  acq = "ucb",
  kappa = 2.576
)

best_laplace <- bayes_opt_result$Best_Par["laplace"]

cat("Best Naive Bayes Hyperparameter:\n")
cat("laplace =", best_laplace,"\n")


# Train Final Naive Bayes Model

nb_model <- naiveBayes(
  x = X_train,
  y = y_train,
  laplace = best_laplace
)


# Predictions

preds <- predict(nb_model, X_test)

y_pred <- factor(preds, levels = levels(y_test))


# Evaluation Metrics

conf <- confusionMatrix(y_pred, y_test)

acc <- as.numeric(conf$overall["Accuracy"])
kappa <- as.numeric(conf$overall["Kappa"])

precision <- Precision(y_pred = y_pred, y_true = y_test)
recall <- Recall(y_pred = y_pred, y_true = y_test)
f1 <- F1_Score(y_pred = y_pred, y_true = y_test)

specificity <- mean(conf$byClass[,"Specificity"], na.rm = TRUE)


conf_mat_table <- table(y_test, y_pred)
mcc <- compute_multiclass_mcc(as.matrix(conf_mat_table))


cat("\nEvaluation Metrics – Naive Bayes\n")

cat(sprintf("Accuracy       : %.4f\n", acc))
cat(sprintf("Precision      : %.4f\n", precision))
cat(sprintf("Recall         : %.4f\n", recall))
cat(sprintf("F1 Score       : %.4f\n", f1))
cat(sprintf("Cohen's Kappa  : %.4f\n", kappa))
cat(sprintf("Specificity    : %.4f\n", specificity))
cat(sprintf("MCC            : %.4f\n", mcc))


# Confusion Matrix Plot

cm_melted <- as.data.frame.table(conf_mat_table)
colnames(cm_melted) <- c("True","Predicted","Count")

ggplot(cm_melted,aes(Predicted,True,fill=Count))+
  geom_tile(color="black")+
  geom_text(aes(label=Count),size=3)+
  scale_fill_gradient(low="white",high="steelblue")+
  labs(title="Confusion Matrix – Naive Bayes")+
  theme_minimal(base_size=9)+
  theme(
    axis.title=element_blank(),
    axis.text=element_blank(),
    axis.ticks=element_blank(),
    panel.border=element_rect(color="black",fill=NA),
    panel.grid=element_blank(),
    legend.position="none"
  )

############ H: Stacking

# Hyperparameter tuning results already obtained earlier

best_rf_mtry <- best_mtry
best_rf_nodesize <- best_nodesize
best_rf_maxnodes <- best_maxnodes

best_eta <- best["eta"]
best_max_depth <- best["max_depth"]
best_subsample <- best["subsample"]
best_colsample <- best["colsample_bytree"]

folds <- createFolds(y_train, k = 5)

rf_meta <- matrix(NA, nrow = length(y_train), ncol = length(levels(y_train)))
xgb_meta <- matrix(NA, nrow = length(y_train), ncol = length(levels(y_train)))

colnames(rf_meta) <- paste0("RF_", levels(y_train))
colnames(xgb_meta) <- paste0("XGB_", levels(y_train))

for(i in seq_along(folds)){
  
  val_idx <- folds[[i]]
  tr_idx <- setdiff(1:length(y_train), val_idx)
  
  rf_fold <- randomForest(
    x = X_train[tr_idx,],
    y = y_train[tr_idx],
    mtry = best_rf_mtry,
    ntree = 500,
    nodesize = best_rf_nodesize,
    maxnodes = best_rf_maxnodes
  )
  
  rf_meta[val_idx,] <- predict(rf_fold, X_train[val_idx,], type="prob")
  
  xgb_fold <- xgboost(
    data = xgb.DMatrix(X_train[tr_idx,], label = as.integer(y_train[tr_idx]) - 1),
    objective = "multi:softprob",
    num_class = num_class,
    eta = best_eta,
    max_depth = as.integer(best_max_depth),
    subsample = best_subsample,
    colsample_bytree = best_colsample,
    nrounds = 100,
    verbose = 0
  )
  
  preds <- predict(xgb_fold, xgb.DMatrix(X_train[val_idx,]))
  xgb_meta[val_idx,] <- matrix(preds, ncol = num_class, byrow = TRUE)
}

meta_train <- data.frame(rf_meta, xgb_meta)
meta_train$Deposit.type <- y_train

# Train meta-model

meta_model <- multinom(Deposit.type ~ ., data = meta_train, trace = FALSE)

# Train final base models

rf_final <- randomForest(
  x = X_train,
  y = y_train,
  mtry = best_rf_mtry,
  ntree = 500,
  nodesize = best_rf_nodesize,
  maxnodes = best_rf_maxnodes
)

xgb_final <- xgboost(
  data = xgb.DMatrix(X_train, label = as.integer(y_train)-1),
  objective = "multi:softprob",
  num_class = num_class,
  eta = best_eta,
  max_depth = as.integer(best_max_depth),
  subsample = best_subsample,
  colsample_bytree = best_colsample,
  nrounds = 100,
  verbose = 0
)

# Generate meta test features

rf_test <- predict(rf_final, X_test, type="prob")

xgb_test <- predict(xgb_final, xgb.DMatrix(X_test))
xgb_test <- matrix(xgb_test, ncol = num_class, byrow = TRUE)

colnames(rf_test) <- paste0("RF_", levels(y_train))
colnames(xgb_test) <- paste0("XGB_", levels(y_train))

meta_test <- data.frame(rf_test, xgb_test)

final_preds <- predict(meta_model, meta_test)

# Evaluate

conf <- confusionMatrix(final_preds, y_test)

acc <- as.numeric(conf$overall["Accuracy"])
kappa <- as.numeric(conf$overall["Kappa"])

f1 <- F1_Score(final_preds, y_test, positive = NULL)
precision <- Precision(final_preds, y_test, positive = NULL)
recall <- Recall(final_preds, y_test, positive = NULL)

specificity <- mean(conf$byClass[, "Specificity"], na.rm = TRUE)

compute_multiclass_mcc <- function(conf_matrix) {
  
  n <- sum(conf_matrix)
  rowsums <- rowSums(conf_matrix)
  colsums <- colSums(conf_matrix)
  diag_vals <- diag(conf_matrix)
  
  s <- sum(diag_vals)
  
  p0 <- sum(rowsums * colsums)
  
  numerator <- (n * s) - p0
  denominator <- sqrt((n^2 - sum(colsums^2)) * (n^2 - sum(rowsums^2)))
  
  if (denominator == 0) return(NA)
  
  return(numerator / denominator)
}

mcc <- compute_multiclass_mcc(as.matrix(table(y_test, final_preds)))

cat("Evaluation Metrics:\n")
cat(sprintf("Accuracy       : %.4f\n", acc))
cat(sprintf("Precision      : %.4f\n", precision))
cat(sprintf("Recall         : %.4f\n", recall))
cat(sprintf("F1 Score       : %.4f\n", f1))
cat(sprintf("Cohen's Kappa  : %.4f\n", kappa))
cat(sprintf("Specificity    : %.4f\n", specificity))
cat(sprintf("MCC            : %.4f\n", mcc))

# Confusion Matrix Plot

cm <- table(True = y_test, Predicted = final_preds)

cm_melted <- as.data.frame(cm)
colnames(cm_melted) <- c("True", "Predicted", "Count")

ggplot(cm_melted, aes(x = Predicted, y = True, fill = Count)) +
  geom_tile(color = "black", linewidth = 0.5) +
  geom_text(aes(label = Count), color = "black", size = 3) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title="Confusion Matrix – Stacking")+
  theme_minimal(base_size = 9) +
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
    panel.grid = element_blank(),
    legend.position = "none"
  )

############ Part 5: Prediction of Blind Data ############

#Reload original dataset (NA rows were removed earlier)
data_original <- read.csv("SPHALERITE_CHEMISTRY2025.csv")

#Extract blind deposits
blind_data <- data_original[is.na(data_original$Deposit.type), ]


############ Color palette

deposit_colors <- c(
  "Epithermal" = "#F8766D",
  "MVT"        = "#A5A500",
  "SEDEX"      = "#00BE7D",
  "Skarn"      = "#00BFFF",
  "VMS"        = "#F876DF"
)


############ Train final models

### XGBoost
xgb_model <- xgboost(
  data = xgb.DMatrix(X_train, label = as.integer(y_train)-1),
  objective = "multi:softprob",
  num_class = length(levels(y_train)),
  nrounds = 30,
  eta = 0.182,
  max_depth = 5,
  subsample = 0.549,
  colsample_bytree = 0.937,
  verbose = 0
)

### AdaBoost
ada_model <- boosting(
  Deposit.type ~ .,
  data = data.frame(X_train, Deposit.type = y_train),
  mfinal = 150,
  control = rpart.control(cp = 0.001, maxdepth = 11)
)

### SVM
svm_model <- svm(
  x = X_train,
  y = y_train,
  kernel = "radial",
  cost = 13.06,
  gamma = 2^(-1.628),
  probability = TRUE
)

### MLP
mlp_model <- nnet(
  X_train,
  class.ind(y_train),
  size = 20,
  decay = 10^(-1.01),
  maxit = 500,
  trace = FALSE,
  softmax = TRUE
)

### Random Forest (for stacking)
rf_model <- randomForest(
  x = X_train,
  y = y_train,
  ntree = 500,
  mtry = 4,
  nodesize = 1,
  maxnodes = 441
)


############ Train stacking meta-model

rf_prob <- predict(rf_model, X_train, type="prob")

xgb_prob <- predict(xgb_model, xgb.DMatrix(X_train))
xgb_prob <- matrix(xgb_prob, ncol=num_class, byrow=TRUE)

colnames(rf_prob) <- paste0("RF_", levels(y_train))
colnames(xgb_prob) <- paste0("XGB_", levels(y_train))

meta_train <- data.frame(rf_prob, xgb_prob)
meta_train$Deposit.type <- y_train

meta_model <- nnet::multinom(Deposit.type ~ ., data=meta_train, trace=FALSE)


############ Preprocess blind samples

prepare_blind <- function(deposit_name){
  
  deposit_data <- blind_data[blind_data$Deposit == deposit_name, ]
  
  features_raw <- deposit_data[,4:15]
  
  min_positive <- min(features_raw[features_raw > 0], na.rm = TRUE)
  features_raw[features_raw <= 0] <- min_positive / 2
  
  features_clr <- clr_transform(features_raw)
  
  features_scaled <- scale(
    features_clr,
    center = attr(features_scaled_ml,"scaled:center"),
    scale  = attr(features_scaled_ml,"scaled:scale")
  )
  
  return(features_scaled)
}


############ Pie chart function

plot_pie <- function(preds){
  
  pie_data <- data.frame(table(preds))
  colnames(pie_data) <- c("Type","Count")
  
  ggplot(pie_data, aes(x="", y=Count, fill=Type)) +
    geom_bar(stat="identity", width=1) +
    coord_polar("y") +
    scale_fill_manual(values = deposit_colors) +
    theme_void() +
    theme(legend.position="none")
}


############ Prediction functions

predict_stack <- function(dep){
  
  X <- prepare_blind(dep)
  
  rf_pred <- predict(rf_model, X, type="prob")
  
  xgb_pred <- predict(xgb_model, xgb.DMatrix(X))
  xgb_pred <- matrix(xgb_pred, ncol=num_class, byrow=TRUE)
  
  colnames(rf_pred) <- paste0("RF_", levels(y_train))
  colnames(xgb_pred) <- paste0("XGB_", levels(y_train))
  
  meta_test <- data.frame(rf_pred, xgb_pred)
  
  preds <- predict(meta_model, meta_test)
  
  plot_pie(preds)
}


predict_xgb <- function(dep){
  
  X <- prepare_blind(dep)
  
  probs <- predict(xgb_model, xgb.DMatrix(X))
  probs <- matrix(probs, ncol=num_class, byrow=TRUE)
  
  preds <- factor(max.col(probs),
                  levels=1:num_class,
                  labels=levels(y_train))
  
  plot_pie(preds)
}


predict_adaboost <- function(dep){
  
  X <- prepare_blind(dep)
  
  preds <- predict.boosting(ada_model, newdata=data.frame(X))
  
  pred_labels <- factor(preds$class, levels=levels(y_train))
  
  plot_pie(pred_labels)
}


predict_svm <- function(dep){
  
  X <- prepare_blind(dep)
  
  preds <- predict(svm_model, X)
  
  plot_pie(preds)
}


predict_mlp <- function(dep){
  
  X <- prepare_blind(dep)
  
  probs <- predict(mlp_model, X, type="raw")
  
  preds <- factor(
    colnames(probs)[apply(probs,1,which.max)],
    levels=levels(y_train)
  )
  
  plot_pie(preds)
}


############ Generate panel figure

library(patchwork)
library(grid)
library(cowplot)

deposits <- c("Beishan","Zawarmala","Xulaojiugou")

stack_plots <- lapply(deposits,predict_stack)
xgb_plots   <- lapply(deposits,predict_xgb)
ada_plots   <- lapply(deposits,predict_adaboost)
svm_plots   <- lapply(deposits,predict_svm)
mlp_plots   <- lapply(deposits,predict_mlp)


############ Arrange rows

row1 <- stack_plots[[1]] + stack_plots[[2]] + stack_plots[[3]]
row2 <- xgb_plots[[1]]   + xgb_plots[[2]]   + xgb_plots[[3]]
row3 <- ada_plots[[1]]   + ada_plots[[2]]   + ada_plots[[3]]
row4 <- svm_plots[[1]]   + svm_plots[[2]]   + svm_plots[[3]]
row5 <- mlp_plots[[1]]   + mlp_plots[[2]]   + mlp_plots[[3]]

panel <- row1 / row2 / row3 / row4 / row5


############ Create legend

legend_plot <- ggplot(data.frame(Type=names(deposit_colors)),
                      aes(x=Type,y=1,fill=Type))+
  geom_bar(stat="identity")+
  scale_fill_manual(values=deposit_colors)+
  theme_void()+
  theme(legend.position="right")

legend <- cowplot::get_legend(legend_plot)


############ Combine panel and legend

final_plot <- cowplot::plot_grid(
  panel,
  legend,
  ncol=2,
  rel_widths=c(4,1)
)

print(final_plot)


############ Add row labels

grid.text("a. Stacking", x=0.02, y=0.92, just="left", gp=gpar(fontsize=14))
grid.text("b. XGBoost",  x=0.02, y=0.73, just="left", gp=gpar(fontsize=14))
grid.text("c. AdaBoost", x=0.02, y=0.54, just="left", gp=gpar(fontsize=14))
grid.text("d. SVM",      x=0.02, y=0.35, just="left", gp=gpar(fontsize=14))
grid.text("e. MLP",      x=0.02, y=0.16, just="left", gp=gpar(fontsize=14))


############ Add deposit labels

grid.text("Beishan",      x=0.27, y=0.03, gp=gpar(fontsize=12))
grid.text("Zawarmala",    x=0.50, y=0.03, gp=gpar(fontsize=12))
grid.text("Xulaojiugou",  x=0.73, y=0.03, gp=gpar(fontsize=12))

