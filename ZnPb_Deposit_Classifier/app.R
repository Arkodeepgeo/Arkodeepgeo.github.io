# ---------------------------
# Load libraries
# ---------------------------
library(shiny)
library(caret)
library(dplyr)
library(e1071)
library(xgboost)
library(adabag)
library(rpart)
library(ggplot2)
library(reshape2)
library(scales)
library(randomForest)
library(nnet)

# ---------------------------
# Helper functions
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
# Load pre-trained models and metadata
# ---------------------------
svm_model <- readRDS("svm_model.rds")
xgb_model <- readRDS("xgb_model.rds")
rf_model <- readRDS("rf_model.rds")
ada_model <- readRDS("ada_model.rds")
mlp_model <- readRDS("mlp_model.rds")
stacking_model <- readRDS("stacking_model.rds")

label_levels <- readRDS("label_levels.rds")
scale_center <- readRDS("scale_center.rds")
scale_scale <- readRDS("scale_scale.rds")

deposit_colors <- c(
  "Epithermal" = "#F8766D",
  "MVT" = "#A5A500",
  "SEDEX" = "#00BE7D",
  "Skarn" = "#00BFFF",
  "VMS" = "#F876DF"
)

# ---------------------------
# UI
# ---------------------------
ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      body { background-color: #001f3f; color: white; margin: 0; }
      .title-box {
        background-color: #003366;
        padding: 15px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
      }
      .custom-sidebar {
        background-color: #ADD8E6;
        padding: 15px 20px;
        border-radius: 10px;
      }
      .plot-box {
        background-color: #003366;
        padding: 20px;
        border-radius: 10px;
      }
      .subtitle-box {
        background-color: #336699;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        text-align: center;
        font-weight: 600;
        font-size: 18px;
        color: white;
      }
      table {
        background-color: white;
        color: black;
        font-size: 12px;
        margin-top: 10px;
        width: 100%;
      }

      /* Make tab titles bold and white */
      .nav-tabs > li > a {
        font-weight: bold !important;
        color: white !important;
      }
      .nav-tabs > li.active > a, 
      .nav-tabs > li.active > a:hover, 
      .nav-tabs > li.active > a:focus {
        font-weight: bold !important;
        color: white !important;
        background-color: #003366 !important;
      }

      /* Make images smaller and fit without scroll */
      .tab-panel img {
        max-width: 100% !important;
        max-height: 400px !important;
        height: auto !important;
        display: block;
        margin-left: auto;
        margin-right: auto;
        object-fit: contain;
        overflow: hidden;
      }
    "))
  ),
  
  tabsetPanel(id = "mainTabs",
              tabPanel("Prediction",
                       fluidRow(
                         column(12, div(class = "title-box", "Categorization of Zn-Pb Deposits Based on Sphalerite Geochemistry"))
                       ),
                       column(4,
                              div(class = "custom-sidebar",
                                  tags$label("Upload CSV File", style = "color:black"),
                                  fileInput("file", NULL, accept = ".csv", width = "100%"),
                                  
                                  # Buttons section above model select, with black lines
                                  tags$hr(style = "border-top: 2px solid black; margin-top: 10px; margin-bottom: 10px;"),
                                  tags$p("Choose the Model best suited for your Analysis", 
                                         style = "color:black; font-weight: bold; margin-bottom: 10px; text-align: center;"),
                                  div(style = "display: flex; gap: 10px; justify-content: center; flex-wrap: nowrap;",
                                      actionButton("toConfusion", "Confusion Matrix", 
                                                   class = "btn btn-info", style = "flex: 1 1 48%; white-space: nowrap;"),
                                      actionButton("toRadar", "Radar Plot", 
                                                   class = "btn btn-secondary", style = "flex: 1 1 48%; white-space: nowrap;")
                                  ),
                                  tags$hr(style = "border-top: 2px solid black; margin-top: 10px; margin-bottom: 10px;"),
                                  
                                  # Select model for prediction dropdown
                                  selectInput("model_select", 
                                              label = tags$span(style = "color:black; font-weight:bold;", "Select Model for Prediction"), 
                                              choices = c("XGBoost", "SVM", "AdaBoost", "Random Forest", "MLP", "Stacking"), selected = "XGBoost"),
                                  
                                  div(class = "row",
                                      div(class = "col-md-6 col-12 mb-2", 
                                          actionButton("predict_btn", "Predict", class = "btn btn-primary", style = "width: 100%;")
                                      ),
                                      div(class = "col-md-6 col-12", 
                                          downloadButton("downloadSheet", "Reference Sheet", class = "btn btn-warning", style = "width: 100%;")
                                      )
                                  ),
                                  br(),
                                  tags$hr(style = "border-top: 1px solid black;"),
                                  tags$p(tags$b("Input Format"), style = "color:black;"),
                                  div(style = "max-width: 100%; overflow-x: auto; margin-top: 10px; background-color: white; padding: 5px; border-radius: 5px;",
                                      tableOutput("sampleTable")
                                  ),
                                  tags$p("Note: Please ensure all missing BDL values in the input data are imputed before uploading.", 
                                         style = "color: black; font-style: italic; font-size: 12px; margin-top: 5px;")
                              )
                       ),
                       column(8,
                              div(class = "plot-box",
                                  div(class = "subtitle-box", "Predicted Deposit Types by Selected Model"),
                                  plotOutput("piePlotPredicted", height = "400px"),
                                  downloadButton("downloadPlot", "Download Plot", class = "btn btn-success", style = "margin-top: 10px;")
                              )
                       )
              ),
              tabPanel("Confusion Matrix",
                       fluidRow(
                         column(12, div(class = "title-box", "Confusion Matrix")),
                         column(12, align = "center",
                                div(style = "max-width: 600px; margin: auto;",
                                    tags$img(src = "Confusion Matrices.jpg", style = "width: 100%; height: auto;")
                                )
                         ),
                         column(12, align = "center",
                                actionButton("backToMain1", "⬅ Back", class = "btn btn-light")
                         )
                       )
              ),
              tabPanel("Radar Plot",
                       fluidRow(
                         column(12, div(class = "title-box", "Radar Plot")),
                         column(12, align = "center",
                                div(style = "max-width: 600px; margin: auto;",
                                    tags$img(src = "Radar Plot.jpg", style = "width: 100%; height: auto;")
                                )
                         ),
                         column(12, align = "center",
                                actionButton("backToMain2", "⬅ Back", class = "btn btn-light")
                         )
                       )
              )
  ),
  
  div(
    tags$footer(
      tags$p("\u00a9 2025 Arkodeep Sengupta. All rights reserved. This application is part of a research that is currently awaiting publication.")
    ),
    style = "position: fixed; bottom: 5px; right: 10px; color: white; font-size: 12px; z-index: 999;"
  )
)

# ---------------------------
# Server
# ---------------------------
server <- function(input, output, session) {
  
  output$sampleTable <- renderTable({
    data.frame(
      Fe = "(ppm)", Mn = "(ppm)", Co = "(ppm)", Cu = "(ppm)",
      Ga = "(ppm)", Ge = "(ppm)", Ag = "(ppm)", Cd = "(ppm)",
      In = "(ppm)", Sn = "(ppm)", Sb = "(ppm)", Pb = "(ppm)",
      check.names = FALSE
    )
  }, bordered = TRUE, spacing = "xs", align = "c")
  
  output$downloadSheet <- downloadHandler(
    filename = function() { "REFERENCE SHEET.csv" },
    content = function(file) { file.copy("REFERENCE SHEET.csv", file) }
  )
  
  processed_data <- eventReactive(input$predict_btn, {
    req(input$file)
    df <- read.csv(input$file$datapath)
    features <- df[, 1:12]
    
    outlier_result <- remove_outliers(features)
    clean_features <- outlier_result$data
    
    min_positive_new <- min(clean_features[clean_features > 0], na.rm = TRUE)
    clean_features[clean_features <= 0] <- min_positive_new / 2
    
    clr_data <- clr_transform(clean_features)
    scaled_data <- scale(clr_data, center = scale_center, scale = scale_scale)
    
    as.data.frame(scaled_data)
  })
  
  predictions <- eventReactive(input$predict_btn, {
    df <- processed_data()
    req(nrow(df) > 0)
    
    switch(input$model_select,
           "SVM" = predict(svm_model, newdata = df),
           "XGBoost" = {
             dtest <- xgb.DMatrix(data = as.matrix(df))
             preds_numeric <- predict(xgb_model, dtest)
             factor(label_levels[preds_numeric + 1], levels = label_levels)
           },
           "AdaBoost" = {
             pred <- predict.boosting(ada_model, newdata = df)
             factor(pred$class, levels = names(deposit_colors))
           },
           "Random Forest" = predict(rf_model, newdata = df),
           "MLP" = {
             preds <- predict(mlp_model, newdata = df, type = "class")
             factor(preds, levels = label_levels)
           },
           "Stacking" = {
             rf_preds <- predict(rf_model, newdata = df)
             xgb_preds <- predict(xgb_model, newdata = xgb.DMatrix(as.matrix(df)))
             xgb_preds <- factor(label_levels[xgb_preds + 1], levels = label_levels)
             meta_features <- data.frame(RF = rf_preds, XGB = xgb_preds)
             predict(stacking_model, newdata = meta_features)
           },
           rep(NA, nrow(df))
    )
  })
  
  plot_object <- reactive({
    preds <- predictions()
    pred_summary <- as.data.frame(table(preds))
    names(pred_summary) <- c("Deposit.type", "Count")
    pred_summary <- pred_summary[pred_summary$Count > 0, ]
    
    ggplot(pred_summary, aes(x = "", y = Count, fill = Deposit.type)) +
      geom_col(width = 1, color = "white") +
      coord_polar(theta = "y") +
      theme_void() +
      scale_fill_manual(values = deposit_colors) +
      ggtitle(paste(input$model_select)) +
      theme(
        legend.position = "right",
        legend.title = element_text(size = 12, face = "bold"),
        legend.text = element_text(size = 10, face = "bold"),
        plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
      ) +
      guides(fill = guide_legend(title = "Deposit Types"))
  })
  
  output$piePlotPredicted <- renderPlot({ plot_object() })
  
  output$downloadPlot <- downloadHandler(
    filename = function() {
      paste0("Predicted_Deposits_", input$model_select, ".jpg")
    },
    content = function(file) {
      ggsave(file, plot = plot_object(), width = 8, height = 6, dpi = 300, device = "jpeg")
    }
  )
  
  observeEvent(input$toConfusion, {
    updateTabsetPanel(session, "mainTabs", selected = "Confusion Matrix")
  })
  
  observeEvent(input$toRadar, {
    updateTabsetPanel(session, "mainTabs", selected = "Radar Plot")
  })
  
  observeEvent(input$backToMain1, {
    updateTabsetPanel(session, "mainTabs", selected = "Prediction")
  })
  
  observeEvent(input$backToMain2, {
    updateTabsetPanel(session, "mainTabs", selected = "Prediction")
  })
}

# ---------------------------
# Run the app
# ---------------------------
shinyApp(ui, server)
