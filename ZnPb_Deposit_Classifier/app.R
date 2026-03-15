# ---------------------------
# Load libraries
# ---------------------------
library(shiny)
library(caret)
library(dplyr)
library(xgboost)
library(adabag)
library(rpart)
library(ggplot2)
library(reshape2)
library(scales)
library(randomForest)

# ---------------------------
# Helper functions
# ---------------------------

remove_outliers <- function(df, max_outlier_cols = 2) {
  
  Q1 <- apply(df,2,quantile,0.25)
  Q3 <- apply(df,2,quantile,0.75)
  
  IQR <- Q3 - Q1
  
  lower <- Q1 - 1.5 * IQR
  upper <- Q3 + 1.5 * IQR
  
  keep <- apply(df,1,function(row){
    sum(row < lower | row > upper) <= max_outlier_cols
  })
  
  list(data=df[keep,],indices=which(keep))
}

clr_transform <- function(x){
  
  x <- as.matrix(x)
  
  if(any(x<=0)){
    x[x<=0] <- min(x[x>0],na.rm=TRUE)/2
  }
  
  log_x <- log(x)
  gm <- rowMeans(log_x)
  
  sweep(log_x,1,gm,"-")
}

# ---------------------------
# Load trained models
# ---------------------------

xgb_model <- readRDS("xgb_model.rds")
rf_model  <- readRDS("rf_model.rds")
ada_model <- readRDS("ada_model.rds")

label_levels <- readRDS("label_levels.rds")
scale_center <- readRDS("scale_center.rds")
scale_scale  <- readRDS("scale_scale.rds")

deposit_colors <- c(
  "Epithermal"="#F8766D",
  "MVT"="#A5A500",
  "SEDEX"="#00BE7D",
  "Skarn"="#00BFFF",
  "VMS"="#F876DF"
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

"))
  ),
  
  tabsetPanel(id="mainTabs",
              
              # ---------------------------
              # Prediction Tab
              # ---------------------------
              
              tabPanel("Prediction",
                       
                       fluidRow(
                         column(12,
                                div(class="title-box",
                                    "Categorization of Zn-Pb Deposits Based on Sphalerite Geochemistry"))
                       ),
                       
                       column(4,
                              
                              div(class="custom-sidebar",
                                  
                                  tags$label("Upload CSV File",style="color:black"),
                                  fileInput("file",NULL,accept=".csv",width="100%"),
                                  
                                  tags$hr(style="border-top:2px solid black"),
                                  
                                  tags$p("Choose the Model best suited for your Analysis",
                                         style="color:black;font-weight:bold;text-align:center"),
                                  
                                  div(style="display:flex;gap:10px",
                                      
                                      actionButton("toConfusion","Confusion Matrix",
                                                   class="btn btn-info"),
                                      
                                      actionButton("toRadar","SHAP Feature Importance",
                                                   class="btn btn-secondary")
                                  ),
                                  
                                  tags$hr(style="border-top:2px solid black"),
                                  
                                  selectInput(
                                    "model_select",
                                    label=tags$span(style="color:black;font-weight:bold;",
                                                    "Select Model for Prediction"),
                                    choices=c("XGBoost","AdaBoost","Random Forest"),
                                    selected="XGBoost"
                                  ),
                                  
                                  actionButton("predict_btn","Predict",
                                               class="btn btn-primary"),
                                  
                                  br(),
                                  br(),
                                  
                                  tags$p(tags$b("Input Format"), style = "color:black;"),
                                  
                                  div(style = "max-width: 100%; overflow-x: auto; margin-top: 10px; 
             background-color: white; padding: 5px; border-radius: 5px;",
                                      tableOutput("sampleTable")
                                  )
                                  
                              )
                       ),
                       
                       column(8,
                              
                              div(class="plot-box",
                                  
                                  div(class="subtitle-box",
                                      "Predicted Deposit Types by Selected Model"),
                                  
                                  plotOutput("piePlotPredicted",height="400px")
                              )
                       )
              ),
              
              # ---------------------------
              # Confusion Matrix
              # ---------------------------
              
              tabPanel("Confusion Matrix",
                       
                       fluidRow(
                         
                         column(12,
                                div(class="title-box","Confusion Matrix")),
                         
                         column(12,align="center",
                                tags$img(src="Confusion Matrices.jpg",
                                         style="width:80%")),
                         
                         column(12,align="center",
                                actionButton("backToMain1","⬅ Back"))
                       )
              ),
              
              # ---------------------------
              # SHAP Importance
              # ---------------------------
              
              tabPanel("Importance Interpretability Diagram",
                       
                       fluidRow(
                         
                         column(12,
                                div(class="title-box",
                                    "Feature Importance and Interpretability")),
                         
                         column(12,align="center",
                                tags$img(src="SHAP feature importance.jpg",
                                         style="width:80%")),
                         
                         column(12,align="center",
                                actionButton("backToMain2","⬅ Back"))
                       )
              )
  )
)

# ---------------------------
# Server
# ---------------------------

server <- function(input,output,session){
  
  output$sampleTable <- renderTable({
    
    data.frame(
      Fe="(ppm)",Mn="(ppm)",Co="(ppm)",Cu="(ppm)",
      Ga="(ppm)",Ge="(ppm)",Ag="(ppm)",Cd="(ppm)",
      In="(ppm)",Sn="(ppm)",Sb="(ppm)",Pb="(ppm)",
      check.names=FALSE
    )
    
  },bordered=TRUE,spacing="xs",align="c")
  
  
  processed_data <- eventReactive(input$predict_btn,{
    
    req(input$file)
    
    df <- read.csv(input$file$datapath)
    
    features <- df[,1:12]
    
    outlier_result <- remove_outliers(features)
    
    clean_features <- outlier_result$data
    
    min_positive_new <- min(clean_features[clean_features>0],na.rm=TRUE)
    
    clean_features[clean_features<=0] <- min_positive_new/2
    
    clr_data <- clr_transform(clean_features)
    
    scaled_data <- scale(clr_data,
                         center=scale_center,
                         scale=scale_scale)
    
    as.data.frame(scaled_data)
    
  })
  
  
  predictions <- eventReactive(input$predict_btn,{
    
    df <- processed_data()
    
    switch(input$model_select,
           
           "XGBoost"={
             
             dtest <- xgb.DMatrix(data=as.matrix(df))
             
             preds_numeric <- predict(xgb_model,dtest)
             
             factor(label_levels[preds_numeric+1],
                    levels=label_levels)
           },
           
           "AdaBoost"={
             
             pred <- predict.boosting(ada_model,newdata=df)
             
             factor(pred$class,levels=label_levels)
           },
           
           "Random Forest"={
             
             predict(rf_model,newdata=df)
           }
    )
    
  })
  
  
  plot_object <- reactive({
    
    preds <- predictions()
    
    pred_summary <- as.data.frame(table(preds))
    
    names(pred_summary) <- c("Deposit.type","Count")
    
    ggplot(pred_summary,
           aes(x="",y=Count,fill=Deposit.type))+
      
      geom_col(width=1,color="white")+
      
      coord_polar(theta="y")+
      
      theme_void()+
      
      scale_fill_manual(values=deposit_colors)+
      
      ggtitle(input$model_select)
    
  })
  
  
  output$piePlotPredicted <- renderPlot({
    
    plot_object()
    
  })
  
  
  observeEvent(input$toConfusion,{
    
    updateTabsetPanel(session,"mainTabs",
                      selected="Confusion Matrix")
    
  })
  
  
  observeEvent(input$toRadar,{
    
    updateTabsetPanel(session,"mainTabs",
                      selected="Importance Interpretability Diagram")
    
  })
  
  
  observeEvent(input$backToMain1,{
    
    updateTabsetPanel(session,"mainTabs",
                      selected="Prediction")
    
  })
  
  
  observeEvent(input$backToMain2,{
    
    updateTabsetPanel(session,"mainTabs",
                      selected="Prediction")
    
  })
}

# ---------------------------
# Run App
# ---------------------------

shinyApp(ui,server)