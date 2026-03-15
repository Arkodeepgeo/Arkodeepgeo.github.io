# ---------------------------
# Load Libraries
# ---------------------------

library(caret)
library(dplyr)
library(xgboost)
library(adabag)
library(rpart)
library(randomForest)
library(themis)

# ---------------------------
# Helper Functions
# ---------------------------

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

clr_transform <- function(x){
  
  x <- as.matrix(x)
  
  if(any(x <= 0)){
    x[x <= 0] <- min(x[x > 0]) / 2
  }
  
  log_x <- log(x)
  
  gm <- rowMeans(log_x)
  
  sweep(log_x,1,gm,"-")
}

# ---------------------------
# Load Dataset
# ---------------------------

dataset <- read.csv("SPHALERITE_CHEMISTRY2025.csv")

dataset <- na.omit(dataset)

dataset_ml <- dataset[,4:16]

features <- dataset_ml[,1:12]
labels <- as.factor(dataset_ml$Deposit.type)

# ---------------------------
# Outlier Removal
# ---------------------------

outlier_result <- remove_outliers(features, max_outlier_cols = 3)

features_clean <- outlier_result$data
labels_clean <- labels[outlier_result$indices]

# ---------------------------
# CLR Transformation
# ---------------------------

features_clr <- clr_transform(features_clean + 1e-6)

# ---------------------------
# Standardization
# ---------------------------

features_scaled <- scale(features_clr)

scale_center <- attr(features_scaled,"scaled:center")
scale_scale  <- attr(features_scaled,"scaled:scale")

# ---------------------------
# Train/Test Split
# ---------------------------

set.seed(123)

train_idx <- createDataPartition(labels_clean,p=0.8,list=FALSE)

X_train <- features_scaled[train_idx,]
y_train <- labels_clean[train_idx]

# ---------------------------
# SMOTE (training data only)
# ---------------------------

train_df <- data.frame(X_train)
train_df$Deposit.type <- y_train

smote_data <- themis::smote(train_df,var="Deposit.type")

X_train <- smote_data[,1:12]
y_train <- smote_data$Deposit.type

cat("Class distribution after SMOTE:\n")
print(table(y_train))

label_levels <- levels(y_train)

# ---------------------------
# Train XGBoost Model
# ---------------------------

dtrain <- xgb.DMatrix(
  data = as.matrix(X_train),
  label = as.numeric(y_train) - 1
)

xgb_model <- xgb.train(
  params = list(
    objective = "multi:softmax",
    num_class = length(label_levels),
    eta = 0.266,
    max_depth = 7,
    subsample = 0.997,
    colsample_bytree = 0.897,
    eval_metric = "merror"
  ),
  data = dtrain,
  nrounds = 30,
  verbose = 0
)

# ---------------------------
# Train AdaBoost Model
# ---------------------------

ada_model <- boosting(
  Deposit.type ~ .,
  data = data.frame(X_train, Deposit.type = y_train),
  mfinal = 112,
  control = rpart.control(
    cp = 0.001,
    maxdepth = 13
  )
)

# ---------------------------
# Train Random Forest Model
# ---------------------------

rf_model <- randomForest(
  x = X_train,
  y = y_train,
  ntree = 500,
  mtry = 3,
  nodesize = 1,
  maxnodes = 500
)

# ---------------------------
# Save Models and Metadata
# ---------------------------

saveRDS(xgb_model,"xgb_model.rds")
saveRDS(rf_model,"rf_model.rds")
saveRDS(ada_model,"ada_model.rds")

saveRDS(label_levels,"label_levels.rds")
saveRDS(scale_center,"scale_center.rds")
saveRDS(scale_scale,"scale_scale.rds")

cat("\n✅ Models and preprocessing parameters saved successfully.\n")
