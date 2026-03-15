############ Load Required Libraries

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
library(pROC)
library(shapviz)
library(fmsb)
library(fastshap)
library(patchwork)
library(themis)

############ Import Dataset

dataset <- read.csv("SPHALERITE_CHEMISTRY2025_3.csv")

dataset <- na.omit(dataset)

############ Part 1: Box Plots

deposit_colors <- c(
  Epithermal = "#E41A1C",
  MVT        = "#377EB8",
  SEDEX      = "#4DAF4A",
  Skarn      = "#984EA3",
  VMS        = "#FF7F00"
)

dataset_new <- dataset[,4:16]

unique_categories <- dataset_new %>% distinct(Deposit.type)
print(unique_categories)

dataset_long <- dataset_new %>%
  pivot_longer(
    cols = -Deposit.type,
    names_to = "Element",
    values_to = "Concentration"
  )

dataset_long <- dataset_long %>%
  filter(is.finite(Concentration) & Concentration > 0)

elements <- c("Fe","Mn","Co","Cu","Ga","Ge","Ag","Cd","In","Sn","Sb","Pb")

dataset_long$Element <- factor(
  dataset_long$Element,
  levels = elements,
  labels = paste0(elements," (ppm)")
)

ggplot(dataset_long,
       aes(x = Deposit.type,
           y = Concentration,
           fill = Deposit.type)) +
  
  stat_boxplot(geom="errorbar",width=0.3,linewidth=0.2) +
  
  geom_boxplot(
    outlier.shape = NA,
    color="black",
    width=0.6,
    linewidth=0.3
  ) +
  
  stat_summary(
    fun = mean,
    geom = "point",
    shape = 21,
    size = 2,
    fill = "gray"
  ) +
  
  facet_wrap(~Element,
             scales="free_y",
             ncol=3,
             strip.position="left") +
  
  scale_fill_manual(values = deposit_colors) + 
  
  scale_y_log10(
    labels = trans_format("log10",math_format(10^.x))
  ) +
  
  scale_x_discrete(expand=expansion(mult=0.2)) +
  
  theme_minimal() +
  
  theme(
    axis.text.x = element_text(size=9,color="black",angle=45,hjust=1),
    axis.text.y = element_text(size=9,color="black"),
    axis.title = element_blank(),
    strip.text.y.left = element_text(face="bold",size=9,angle=90),
    strip.placement="outside",
    panel.spacing=unit(0.4,"lines"),
    legend.position="none",
    panel.border=element_rect(color="black",fill=NA,linewidth=0.3),
    panel.grid.major=element_blank(),
    panel.grid.minor=element_blank(),
    axis.ticks.y=element_line(color="black"),
    aspect.ratio=1
  )

############ Part 2: Dimensionality Reduction

dataset_ml <- dataset[,4:16]

dataset_ml$Deposit.type <- as.factor(dataset_ml$Deposit.type)

features <- dataset_ml[,1:12]
labels <- dataset_ml$Deposit.type

############ Uniform Outlier Removal

remove_outliers <- function(df, max_outlier_cols = 3){
  
  Q1 <- apply(df,2,quantile,0.25)
  Q3 <- apply(df,2,quantile,0.75)
  
  IQR <- Q3 - Q1
  
  lower <- Q1 - 1.5 * IQR
  upper <- Q3 + 1.5 * IQR
  
  keep <- apply(df,1,function(row){
    sum(row < lower | row > upper) <= max_outlier_cols
  })
  
  list(
    data = df[keep,],
    indices = which(keep)
  )
}

outlier_result <- remove_outliers(features, max_outlier_cols = 3)

features_clean <- outlier_result$data
labels_clean <- labels[outlier_result$indices]

############ CLR Transformation

clr_transform <- function(x){
  
  x <- as.matrix(x)
  
  if(any(x <= 0))
    stop("CLR requires positive values")
  
  log_x <- log(x)
  
  gm <- rowMeans(log_x)
  
  sweep(log_x,1,gm,"-")
}

features_clr <- clr_transform(features_clean + 1e-6)

############ Standardization

features_scaled <- scale(features_clr)

############ PCA

set.seed(123)

pca_result <- PCA(
  features_scaled,
  scale.unit = TRUE,
  ncp = 3,
  graph = FALSE
)

pca_df <- data.frame(
  PC1 = pca_result$ind$coord[,1],
  PC2 = pca_result$ind$coord[,2],
  Deposit.type = labels_clean
)

ggplot(pca_df,aes(PC1,PC2,color=Deposit.type)) +
  geom_point(size=1) +
  scale_color_manual(values = deposit_colors) +
  
  labs(
    title="PCA: CLR-transformed geochemical data",
    x=paste0("PC1 (",round(pca_result$eig[1,2],1),"%)"),
    y=paste0("PC2 (",round(pca_result$eig[2,2],1),"%)")
  ) +
  
  xlim(-4,4) +
  ylim(-4,4) +
  
  theme_minimal(base_size=9) +
  
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
  dims = 2,
  perplexity = 50,
  theta = 0.5,
  eta = 200,
  max_iter = 1000
)

tsne_df <- data.frame(
  X = tsne_result$Y[,1],
  Y = tsne_result$Y[,2],
  Deposit.type = labels_tsne
)

ggplot(tsne_df, aes(X, Y, color = Deposit.type)) +
  geom_point(size = 1) +
  scale_color_manual(values = deposit_colors) +
  labs(x = "t-SNE1", y = "t-SNE2") +
  theme_minimal(base_size = 9) +
  theme(
    legend.title = element_blank(),
    panel.border = element_rect(color = "black", fill = NA),
    axis.ticks = element_line(color = "black"),
    axis.ticks.length = unit(0.15,"cm"),
    panel.grid = element_blank(),
    legend.position = "none"
  )

############ C: UMAP

set.seed(123)

umap_result <- umap(features_scaled)

umap_df <- data.frame(
  UMAP1 = umap_result[, 1],
  UMAP2 = umap_result[, 2],
  Deposit.type = labels_clean
)

ggplot(umap_df, aes(UMAP1, UMAP2, color = Deposit.type)) +
  geom_point(size = 1) +
  scale_color_manual(values = deposit_colors) +
  labs(
    title = "UMAP: Outlier-filtered CLR-transformed data",
    x = "UMAP Dimension 1",
    y = "UMAP Dimension 2"
  ) +
  theme_minimal(base_size = 9) +
  theme(
    legend.title = element_blank(),
    panel.border = element_rect(color = "black", fill = NA),
    axis.ticks.length = unit(0.15,"cm"),
    axis.ticks = element_line(color = "black"),
    panel.grid = element_blank(),
    legend.position = "none"
  )

############ Machine Learning

# Use the same processed data

features_ml <- features_scaled
labels_ml <- labels_clean

############ Train/Test Split

set.seed(123)

train_idx <- createDataPartition(labels_ml, p=0.8, list=FALSE)

X_train <- features_ml[train_idx,]
y_train <- labels_ml[train_idx]

X_test <- features_ml[-train_idx,]
y_test <- labels_ml[-train_idx]

num_class <- length(levels(y_train))

############ Apply SMOTE to Training Data Only

library(themis)

train_smote <- data.frame(X_train)
train_smote$Deposit.type <- as.factor(y_train)

# Apply SMOTE
smote_data <- themis::smote(
  train_smote,
  var = "Deposit.type"
)

# Extract balanced dataset
X_train <- smote_data[,1:12]
y_train <- smote_data$Deposit.type

cat("Class distribution after SMOTE:\n")
print(table(y_train))

############ Performance Storage for Radar Plot

performance <- data.frame(
  Model = character(),
  Accuracy = numeric(),
  Precision = numeric(),
  Recall = numeric(),
  Specificity = numeric(),
  F1_Score = numeric(),
  Cohens_Kappa = numeric(),
  MCC = numeric(),
  stringsAsFactors = FALSE
)

############ XGBoost ############

#Bayesian Optimization

xgb_cv_bayes <- function(eta,max_depth,subsample,colsample_bytree){
  
  param <- list(
    objective="multi:softprob",
    num_class=num_class,
    eval_metric="merror",
    learning_rate=eta,
    max_depth=as.integer(max_depth),
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    tree_method="hist"
  )
  
  dtrain <- xgb.DMatrix(data=X_train,label=as.integer(y_train)-1)
  
  cv <- xgb.cv(
    params=param,
    data=dtrain,
    nrounds=100,
    nfold=5,
    verbose=0,
    early_stopping_rounds=10
  )
  
  list(Score=-min(cv$evaluation_log$test_merror_mean),Pred=0)
}

set.seed(123)

bayes_opt_result <- BayesianOptimization(
  FUN=xgb_cv_bayes,
  bounds=list(
    eta=c(0.01,0.3),
    max_depth=c(3L,10L),
    subsample=c(0.5,1),
    colsample_bytree=c(0.5,1)
  ),
  init_points = 10,
  n_iter = 20
)

best <- bayes_opt_result$Best_Par

############ Final XGBoost Model

dtrain <- xgb.DMatrix(data=X_train,label=as.integer(y_train)-1)

params <- list(
  objective="multi:softprob",
  num_class=num_class,
  eval_metric="merror",
  learning_rate=best["eta"],
  max_depth=as.integer(best["max_depth"]),
  subsample=best["subsample"],
  colsample_bytree=best["colsample_bytree"],
  tree_method="hist"
)

final_model <- xgb.train(
  params=params,
  data=dtrain,
  nrounds=100,
  verbose=0
)

############ Prediction

dtest <- xgb.DMatrix(data = X_test)

pred_prob <- predict(final_model,dtest)

pred_matrix <- matrix(pred_prob,nrow=nrow(X_test),ncol=num_class)

preds <- max.col(pred_matrix)-1

preds_factor <- factor(preds,
                       levels=0:(num_class-1),
                       labels=levels(y_train))

############ Evaluation Metrics

conf <- caret::confusionMatrix(preds_factor,y_test)

acc <- as.numeric(conf$overall["Accuracy"])
kappa <- as.numeric(conf$overall["Kappa"])

precision <- MLmetrics::Precision(preds_factor,y_test)
recall <- MLmetrics::Recall(preds_factor,y_test)
f1 <- MLmetrics::F1_Score(preds_factor,y_test)

specificity <- mean(conf$byClass[,"Specificity"],na.rm=TRUE)

compute_multiclass_mcc <- function(conf_matrix){
  n <- sum(conf_matrix)
  rowsums <- rowSums(conf_matrix)
  colsums <- colSums(conf_matrix)
  diag_vals <- diag(conf_matrix)
  s <- sum(diag_vals)
  p0 <- sum(rowsums*colsums)
  numerator <- (n*s)-p0
  denominator <- sqrt((n^2-sum(colsums^2))*(n^2-sum(rowsums^2)))
  if(denominator==0) return(NA)
  numerator/denominator
}

conf_mat_table <- table(y_test,preds_factor)
mcc <- compute_multiclass_mcc(as.matrix(conf_mat_table))

cat("Evaluation Metrics:\n")
cat(sprintf("Accuracy       : %.4f\n",acc))
cat(sprintf("Precision      : %.4f\n",precision))
cat(sprintf("Recall         : %.4f\n",recall))
cat(sprintf("F1 Score       : %.4f\n",f1))
cat(sprintf("Cohen's Kappa  : %.4f\n",kappa))
cat(sprintf("Specificity    : %.4f\n",specificity))
cat(sprintf("MCC            : %.4f\n",mcc))

performance <- rbind(
  performance,
  data.frame(
    Model = "XGBoost",
    Accuracy = acc * 100,
    Precision = precision * 100,
    Recall = recall * 100,
    Specificity = specificity * 100,
    F1_Score = f1 * 100,
    Cohens_Kappa = kappa * 100,
    MCC = mcc * 100
  )
)

############ Confusion Matrix Plot

cm <- table(y_test,preds_factor)

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
    legend.position="right"
  )

############ ROC Curve – XGBoost (Multiclass)

library(pROC)

# Predict class probabilities
prob_raw <- predict(final_model, dtest)

prob_mat <- matrix(prob_raw,
                   nrow = nrow(X_test),
                   ncol = num_class)

colnames(prob_mat) <- levels(y_train)

# Compute one-vs-rest ROC curves
roc_list <- list()
auc_vals <- numeric(num_class)

for (i in seq_len(num_class)) {
  
  class_label <- levels(y_train)[i]
  
  binary_truth <- ifelse(y_test == class_label, 1, 0)
  
  # Skip if only one class present
  if(length(unique(binary_truth)) < 2) next
  
  roc_obj <- pROC::roc(binary_truth, prob_mat[, i])
  
  roc_list[[i]] <- roc_obj
  auc_vals[i] <- pROC::auc(roc_obj)
}

# Combine ROC curves for plotting
roc_df_list <- lapply(seq_along(roc_list), function(i) {
  
  data.frame(
    Sensitivity = roc_list[[i]]$sensitivities,
    Specificity = 1 - roc_list[[i]]$specificities,
    Class = paste0(
      levels(y_train)[i],
      " (AUC = ",
      sprintf("%.3f", auc_vals[i]),
      ")"
    )
  )
})

roc_df <- do.call(rbind, roc_df_list)

macro_auc <- mean(auc_vals)
# average AUC
macro_auc <- mean(auc_vals)

# ---- Rename classes to deposit names if needed ----
# roc_df$Class <- gsub("Class_1", "Skarn", roc_df$Class)
# roc_df$Class <- gsub("Class_2", "Porphyry", roc_df$Class)
# ... etc.

# Color palette (matches the stacked bar chart)
class_colors <- c("#E41A1C",
                  "#377EB8",
                  "#4DAF4A",
                  "#984EA3",
                  "#FF7F00")
names(class_colors) <- unique(roc_df$Class)

p_roc <- ggplot(roc_df, aes(x = Specificity, y = Sensitivity, color = Class)) +
  geom_line(linewidth = 0.7) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed",
              color = "grey50", linewidth = 0.4) +
  scale_color_manual(values = class_colors) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2),
                     expand = c(0.01, 0.01)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2),
                     expand = c(0.01, 0.01)) +
  labs(
    x     = "1 \u2013 Specificity (False Positive Rate)",
    y     = "Sensitivity (True Positive Rate)",
    title = "ROC Curves \u2013 XGBoost",
    color = NULL
  ) +
  annotate("text", x = 0.55, y = 0.08, size = 2.8, color = "black",
           label = paste0("average AUC = ", sprintf("%.3f", macro_auc))) +
  coord_equal() +
  theme_bw(base_size = 9, base_family = "sans") +
  theme(
    plot.title         = element_text(hjust = 0.5, face = "bold", size = 10),
    axis.title.x       = element_text(size = 8, color = "black",
                                      margin = ggplot2::margin(t = 6)),
    axis.title.y       = element_text(size = 8, color = "black",
                                      margin = ggplot2::margin(r = 6)),
    axis.text          = element_text(size = 7, color = "black"),
    axis.ticks         = element_line(linewidth = 0.3, color = "black"),
    axis.ticks.length  = unit(1.5, "pt"),
    panel.grid.major   = element_line(linewidth = 0.15, color = "grey90"),
    panel.grid.minor   = element_blank(),
    panel.border       = element_rect(linewidth = 0.5, color = "black", fill = NA),
    panel.background   = element_rect(fill = "white"),
    plot.background    = element_rect(fill = "white", color = NA),
    legend.position    = c(0.72, 0.28),
    legend.background  = element_rect(fill = "white", color = "grey60", linewidth = 0.3),
    legend.text        = element_text(size = 6.5),
    legend.key.size    = unit(0.4, "cm"),
    legend.key.width   = unit(0.6, "cm"),
    legend.spacing.y   = unit(1, "pt"),
    plot.margin        = ggplot2::margin(t = 8, r = 8, b = 8, l = 8, unit = "pt")
  )

print(p_roc)

############ B: Adaboost ############

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

# Bayesian Optimization

set.seed(123)

bayes_opt_result <- BayesianOptimization(
  FUN = adaboost_cv_bayes,
  bounds = list(
    mfinal = c(50L,150L),
    maxdepth = c(3L,15L),
    cp = c(0.001,0.01)
  ),
  init_points = 10,
  n_iter = 10,
  acq = "ucb",
  kappa = 2.576,
  eps = 0.01,
  verbose = TRUE
)

# Best Parameters

best_mfinal <- floor(bayes_opt_result$Best_Par[["mfinal"]])
best_maxdepth <- floor(bayes_opt_result$Best_Par[["maxdepth"]])
best_cp <- bayes_opt_result$Best_Par[["cp"]]

# Train Final Model

final_model <- boosting(
  Deposit.type ~ .,
  data = train_df,
  mfinal = best_mfinal,
  control = rpart.control(maxdepth = best_maxdepth, cp = best_cp)
)

# Predictions

test_df <- data.frame(X_test)

preds <- predict.boosting(final_model, newdata = test_df)

y_pred <- factor(preds$class, levels = levels(y_test))

# Evaluation Metrics

conf <- confusionMatrix(y_pred, y_test)

acc <- as.numeric(conf$overall["Accuracy"])
kappa <- as.numeric(conf$overall["Kappa"])

f1 <- F1_Score(y_pred = y_pred, y_true = y_test)
precision <- Precision(y_pred = y_pred, y_true = y_test)
recall <- Recall(y_pred = y_pred, y_true = y_test)

specificity <- mean(conf$byClass[,"Specificity"], na.rm = TRUE)

# MCC

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

performance <- rbind(
  performance,
  data.frame(
    Model = "AdaBoost",
    Accuracy = acc * 100,
    Precision = precision * 100,
    Recall = recall * 100,
    Specificity = specificity * 100,
    F1_Score = f1 * 100,
    Cohens_Kappa = kappa * 100,
    MCC = mcc * 100
  )
)

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

############ ROC Curve – AdaBoost (Multiclass)

library(pROC)

# Predict class probabilities
prob_mat <- preds$prob

prob_mat <- as.matrix(prob_mat)

colnames(prob_mat) <- levels(y_train)

roc_list <- list()
auc_vals <- c()

for(i in 1:num_class){
  
  class_label <- levels(y_train)[i]
  
  binary_truth <- ifelse(y_test == class_label,1,0)
  
  if(length(unique(binary_truth)) < 2) next
  
  roc_obj <- pROC::roc(
    response = binary_truth,
    predictor = prob_mat[,i],
    quiet = TRUE
  )
  
  roc_list[[class_label]] <- roc_obj
  auc_vals[class_label] <- pROC::auc(roc_obj)
}

############ Convert ROC data

roc_df_list <- lapply(names(roc_list), function(cls){
  
  roc_obj <- roc_list[[cls]]
  
  data.frame(
    Sensitivity = roc_obj$sensitivities,
    Specificity = 1 - roc_obj$specificities,
    Class = paste0(cls,
                   " (AUC=",
                   sprintf("%.3f", auc_vals[cls]),
                   ")")
  )
})

roc_df <- do.call(rbind, roc_df_list)

macro_auc <- mean(auc_vals)

############ Plot ROC

p_roc_ada <- ggplot(roc_df,
                    aes(Specificity,Sensitivity,color=Class))+
  
  geom_line(linewidth=0.8)+
  
  geom_abline(intercept=0,
              slope=1,
              linetype="dashed",
              color="grey50")+
  
  scale_color_manual(values=c(
    "#E41A1C",
    "#377EB8",
    "#4DAF4A",
    "#984EA3",
    "#FF7F00"
  ))+
  
  coord_equal()+
  
  labs(
    title="ROC Curves – AdaBoost",
    x="1 − Specificity (False Positive Rate)",
    y="Sensitivity (True Positive Rate)"
  )+
  
  annotate("text",
           x=0.55,
           y=0.08,
           label=paste0("Average AUC = ",
                        sprintf("%.3f",macro_auc)),
           size=3)+
  
  theme_bw(base_size=9)

print(p_roc_ada)

############ C: Random Forest ############

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

############ MCC calculation

compute_multiclass_mcc <- function(conf_matrix){
  
  n <- sum(conf_matrix)
  
  rowsums <- rowSums(conf_matrix)
  colsums <- colSums(conf_matrix)
  
  diag_vals <- diag(conf_matrix)
  
  s <- sum(diag_vals)
  
  p0 <- sum(rowsums * colsums)
  
  numerator <- (n*s) - p0
  
  denominator <- sqrt((n^2 - sum(colsums^2)) *
                        (n^2 - sum(rowsums^2)))
  
  if(denominator == 0) return(NA)
  
  numerator / denominator
}

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

performance <- rbind(
  performance,
  data.frame(
    Model = "Random Forest",
    Accuracy = acc * 100,
    Precision = precision * 100,
    Recall = recall * 100,
    Specificity = specificity * 100,
    F1_Score = f1 * 100,
    Cohens_Kappa = kappa * 100,
    MCC = mcc * 100
  )
)

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

############ ROC Curve – Random Forest (Multiclass)

# Predict probabilities
rf_prob <- predict(rf_model, X_test, type = "prob")

# Ensure matrix format
prob_mat <- as.matrix(rf_prob)

# Ensure columns match class labels
colnames(prob_mat) <- levels(y_train)

roc_list <- list()
auc_vals <- c()

for (i in 1:num_class) {
  
  class_label <- levels(y_train)[i]
  
  binary_truth <- ifelse(y_test == class_label, 1, 0)
  
  # Skip if only one class present
  if(length(unique(binary_truth)) < 2) next
  
  roc_obj <- pROC::roc(
    response = binary_truth,
    predictor = prob_mat[,i],
    quiet = TRUE
  )
  
  roc_list[[class_label]] <- roc_obj
  auc_vals[class_label] <- pROC::auc(roc_obj)
}

############ Convert ROC data for ggplot

roc_df_list <- lapply(names(roc_list), function(cls){
  
  roc_obj <- roc_list[[cls]]
  
  data.frame(
    Sensitivity = roc_obj$sensitivities,
    Specificity = 1 - roc_obj$specificities,
    Class = paste0(cls,
                   " (AUC=",
                   sprintf("%.3f", auc_vals[cls]),
                   ")")
  )
})

roc_df <- do.call(rbind, roc_df_list)

macro_auc <- mean(auc_vals)

############ Plot

p_roc_rf <- ggplot(roc_df,
                   aes(Specificity,Sensitivity,color=Class))+
  
  geom_line(linewidth=0.8)+
  
  geom_abline(intercept=0,
              slope=1,
              linetype="dashed",
              color="grey50")+
  
  scale_color_manual(values=c(
    "#E41A1C",
    "#377EB8",
    "#4DAF4A",
    "#984EA3",
    "#FF7F00"
  ))+
  
  coord_equal()+
  
  labs(
    title="ROC Curves – Random Forest",
    x="1 − Specificity (False Positive Rate)",
    y="Sensitivity (True Positive Rate)"
  )+
  
  annotate("text",
           x=0.55,
           y=0.08,
           label=paste0("Average AUC = ",
                        sprintf("%.3f",macro_auc)),
           size=3)+
  
  theme_bw(base_size=9)

print(p_roc_rf)

############ SHAP Analysis – Random Forest (All Classes)

classes <- levels(y_train)

importance_list <- list()

for(i in seq_along(classes)){
  
  cat("Computing SHAP for:", classes[i], "\n")
  
  pred_fun <- function(object, newdata){
    
    pred <- predict(object,
                    newdata,
                    type = "prob")
    
    pred[,i]
  }
  
  shap_vals <- fastshap::explain(
    object = rf_model,
    X = as.data.frame(X_train),
    pred_wrapper = pred_fun,
    nsim = 100
  )
  
  ############ Beeswarm Plot
  
  shp_rf <- shapviz(
    shap_vals,
    X = as.data.frame(X_train)
  )
  
  p_bee_rf <- sv_importance(
    shp_rf,
    kind = "beeswarm",
    max_display = 12,
    size = 0.8,
    alpha = 0.4
  ) +
    theme_classic(base_size = 9) +
    labs(
      title = paste0("SHAP Beeswarm – Random Forest (",classes[i],")")
    ) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.text = element_text(color = "black")
    )
  
  print(p_bee_rf)
  
  ############ Feature Importance
  
  importance_list[[i]] <- data.frame(
    Feature = colnames(shap_vals),
    Mean_SHAP = colMeans(abs(shap_vals)),
    Class = classes[i]
  )
}

############ Combine SHAP values

importance_df <- do.call(rbind, importance_list)

############ Order features

feature_totals <- tapply(
  importance_df$Mean_SHAP,
  importance_df$Feature,
  sum
)

feature_order <- names(sort(feature_totals))

importance_df$Feature <- factor(
  importance_df$Feature,
  levels = feature_order
)

############ Top 12 features

top_features <- tail(feature_order,12)

importance_df <- importance_df[
  importance_df$Feature %in% top_features, ]

############ SHAP Stacked Plot

p_bar_rf <- ggplot(
  importance_df,
  aes(x = Mean_SHAP,
      y = Feature,
      fill = Class)
) +
  
  geom_bar(stat="identity",width = 0.65) +
  
  scale_fill_manual(
    values = c(
      "#E41A1C",
      "#377EB8",
      "#4DAF4A",
      "#984EA3",
      "#FF7F00"
    )
  ) +
  
  scale_x_continuous(expand = expansion(mult = c(0,0.05))) +
  
  labs(
    x = "mean(|SHAP value|)",
    y = NULL,
    fill = NULL
  ) +
  
  theme_bw(base_size = 8) +
  
  theme(
    axis.text.y = element_text(size = 8,face = "bold"),
    axis.text.x = element_text(size = 7),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(linewidth = 0.2,color = "grey85"),
    panel.border = element_rect(color = "black",fill = NA),
    legend.position = c(0.82,0.22)
  )

print(p_bar_rf)

############ SHAP FORCE PLOT – ZAWARMALA (Random Forest)

############ Reload Original Dataset

data_original <- read.csv("SPHALERITE_CHEMISTRY2025_3.csv")

############ Extract Zawarmala Deposit

zawar_data <- data_original %>%
  filter(Deposit == "Zawarmala")

############ Extract Element Columns

X_zawar <- zawar_data %>%
  select(Fe, Mn, Co, Cu, Ga, Ge, Ag, Cd, In, Sn, Sb, Pb)

############ Replace NA values

for(i in 1:ncol(X_zawar)){
  
  X_zawar[[i]][is.na(X_zawar[[i]])] <-
    median(X_train[,i], na.rm = TRUE)
}

############ Apply Same Preprocessing

X_zawar_clr <- clr_transform(X_zawar + 1e-6)

X_zawar_scaled <- scale(
  X_zawar_clr,
  center = attr(features_scaled,"scaled:center"),
  scale  = attr(features_scaled,"scaled:scale")
)

############ Compute Mean Composition

zawar_mean <- as.data.frame(
  matrix(colMeans(X_zawar_scaled), nrow = 1)
)

colnames(zawar_mean) <- colnames(X_train)

############ Predict Deposit Type

prob <- predict(rf_model, zawar_mean, type="prob")

print(prob)

pred_class <- colnames(prob)[which.max(prob)]

cat("Predicted Deposit Type for Zawarmala (RF):", pred_class,"\n")

############ SHAP FORCE PLOTS – ZAWARMALA (Random Forest)

deposit_types <- levels(y_train)

for(i in seq_along(deposit_types)){
  
  pred_fun <- function(object,newdata){
    
    pred <- predict(object,newdata,type="prob")
    
    pred[,i]
  }
  
  shap_force <- fastshap::explain(
    object = rf_model,
    X = as.data.frame(X_train),
    pred_wrapper = pred_fun,
    newdata = zawar_mean,
    nsim = 50
  )
  
  shp_force <- shapviz(
    shap_force,
    X = zawar_mean
  )
  
  p_force_rf <- sv_force(
    shp_force,
    row_id = 1,
    max_display = 12
  ) +
    
    ggtitle(
      paste0(
        "SHAP Force Plot – Random Forest (",
        deposit_types[i],
        ")  f(x) = ",
        round(prob[i],3)
      )
    ) +
    
    theme(
      plot.title = element_text(hjust = 0.5, face="bold")
    )
  
  print(p_force_rf)
}

############ Radar Plots for Model Performance  ############

print(performance)

max_min <- data.frame(
  Accuracy = c(100,60),
  Precision = c(100,60),
  Recall = c(100,60),
  Specificity = c(100,60),
  F1_Score = c(100,60),
  Cohens_Kappa = c(100,60),
  MCC = c(100,60)
)

# Custom layout (2 top plots, 1 bottom centered)
layout(matrix(c(1,2,3,3), nrow = 2, byrow = TRUE))
par(mar = c(2,2,3,2))

for(i in 1:nrow(performance)){
  
  model_data <- rbind(max_min, performance[i,-1])
  
  rownames(model_data) <- c("Max","Min",performance$Model[i])
  
  radarchart(
    model_data,
    axistype = 1,
    pcol = rgb(0.2,0.5,0.9,0.9),
    pfcol = rgb(0.2,0.5,0.9,0.25),
    plwd = 2,
    plty = 1,
    cglcol = "grey70",
    cglty = 1,
    axislabcol = "grey30",
    caxislabels = seq(60,100,10),
    vlcex = 0.8,
    title = performance$Model[i]
  )
}

