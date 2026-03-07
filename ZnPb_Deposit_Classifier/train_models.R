# ---------------------------
# Load libraries
# ---------------------------
library(caret)
library(dplyr)
library(e1071)
library(xgboost)
library(adabag)
library(rpart)
library(randomForest)
library(nnet)

# ---------------------------
# Helper Functions
# ---------------------------
remove_outliers <- function(df, max_outlier_cols = 2) {
  Q1 <- apply(df, 2, quantile, 0.25)
  Q3 <- apply(df, 2, quantile, 0.75)
  IQR <- Q3 - Q1
  lower <- Q1 - 1.5 * IQR
  upper <- Q3 + 1.5 * IQR
  keep <- apply(df, 1, function(row) {
    sum(row < lower | row > upper) <= max_outlier_cols
  })
  list(data = df[keep, ], indices = which(keep))
}

clr_transform <- function(x) {
  x <- as.matrix(x)
  if (any(x <= 0)) {
    x[x <= 0] <- min(x[x > 0], na.rm = TRUE) / 2
  }
  log_x <- log(x)
  gm <- rowMeans(log_x)
  sweep(log_x, 1, gm, "-")
}

# ---------------------------
# Load and Preprocess Data
# ---------------------------
data_all <- read.csv("SPHALERITE_CHEMISTRY2025.csv")
data_all <- data_all[!is.na(data_all$Deposit.type), ]

known_data <- data_all[, 4:16]
labels <- as.factor(known_data$Deposit.type)
features <- known_data[, 1:12]

outlier_result <- remove_outliers(features)
features_clean <- outlier_result$data
labels_clean <- labels[outlier_result$indices]

min_positive <- min(features_clean[features_clean > 0], na.rm = TRUE)
features_clean[features_clean <= 0] <- min_positive / 2

features_clr <- clr_transform(features_clean)
features_transformed <- scale(features_clr)
colnames(features_transformed) <- colnames(features)

scale_center <- attr(features_transformed, "scaled:center")
scale_scale <- attr(features_transformed, "scaled:scale")

train_df <- data.frame(features_transformed)
train_df$Deposit.type <- labels_clean

label_levels <- levels(labels_clean)

# ---------------------------
# Train Models
# ---------------------------

set.seed(123)
svm_model <- svm(
  Deposit.type ~ .,
  data = train_df,
  type = "C-classification",
  kernel = "radial",
  cost = 1,
  gamma = 0.373,
  probability = TRUE
)

y_numeric <- as.numeric(labels_clean) - 1
dtrain <- xgb.DMatrix(data = features_transformed, label = y_numeric)

xgb_params <- list(
  objective = "multi:softmax",
  num_class = length(label_levels),
  eta = 0.104,
  max_depth = 8,
  subsample = 0.703,
  colsample_bytree = 0.593,
  eval_metric = "merror"
)

set.seed(123)
xgb_model <- xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = 100,
  verbose = 0
)

set.seed(123)
ada_model <- boosting(
  Deposit.type ~ .,
  data = train_df,
  boos = TRUE,
  mfinal = 150,
  control = rpart.control(cp = 0.001, maxdepth = 13)
)

set.seed(123)
rf_model <- randomForest(
  Deposit.type ~ .,
  data = train_df,
  ntree = 500,
  mtry = 6,
  nodesize = 1,
  maxnodes = 323
)

set.seed(123)
mlp_model <- nnet(
  Deposit.type ~ .,
  data = train_df,
  size = 20,
  decay = 0.1,
  maxit = 500,
  trace = FALSE
)

# ---------------------------
# Train Stacking Model
# ---------------------------
rf_preds_train <- predict(rf_model, newdata = train_df, type = "response")
xgb_preds_train <- predict(xgb_model, newdata = dtrain)
xgb_preds_train <- factor(label_levels[xgb_preds_train + 1], levels = label_levels)

meta_train <- data.frame(RF = rf_preds_train, XGB = xgb_preds_train)

stacking_model <- multinom(
  Deposit.type ~ .,
  data = cbind(meta_train, Deposit.type = labels_clean),
  trace = FALSE
)

# ---------------------------
# Save All Models and Objects
# ---------------------------
saveRDS(svm_model, "svm_model.rds")
saveRDS(xgb_model, "xgb_model.rds")
saveRDS(rf_model, "rf_model.rds")
saveRDS(ada_model, "ada_model.rds")
saveRDS(mlp_model, "mlp_model.rds")
saveRDS(stacking_model, "stacking_model.rds")
saveRDS(label_levels, "label_levels.rds")
saveRDS(scale_center, "scale_center.rds")
saveRDS(scale_scale, "scale_scale.rds")

cat("✅ All models and transformation parameters saved successfully.\n")

