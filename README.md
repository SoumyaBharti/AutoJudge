AutoJudge — Problem Difficulty Prediction System

Author

    Soumya Bharti
    Mathematics and Computing, IIT Roorkee

Project Overview

    AutoJudge is a machine learning–based system that predicts the difficulty of programming problems using only textual information.
    The system outputs:
    
    A difficulty score (regression)
    
    A difficulty class (Easy / Medium / Hard)
    
    The project includes data preprocessing, feature engineering, ML models, and a Streamlit web interface.

Dataset

    Source: Curated competitive programming problems
    
    Fields used:
    
    Title
    
    Description
    
    Input description
    
    Output description
    
    Difficulty score
    
    Difficulty class

Approach
  1️ Data Preprocessing
  
    Combined multiple text fields into a single full_text
    
    Removed noise and standardized text
    
    Stratified train-test split based on difficulty class
  
  2️ Feature Engineering
    Textual Features
    
    TF-IDF vectorization (unigrams & bigrams)
    
    Handcrafted Features
    
    Text length
    
    Word count
    
    Line count
    
    Symbol count
    
    Keyword indicators (DP, graph, constraints, etc.)
    
    Big-O notation detection
  
  3️ Models Used
  
    Classification (Easy / Medium / Hard)
    
          Logistic Regression
      
          Linear SVM
          
          Random Forest (best performing)
          
          Histogram Gradient Boosting
          
          Metric: Accuracy, Precision, Recall, Confusion Matrix
        
    Regression (Difficulty Score)
    
          Histogram Gradient Boosting Regressor (best performing)
          
          Random Forest Regressor
          
          Gradient Boosting Regressor
          
          Ridge Regression
          
          Metric: MAE, RMSE, R² score

Evaluation Results (Best Models)
      
      Classification	    Random Forest	~52% Accuracy
      Regression	        HistGradientBoosting	MAE ~1.65, RMSE ~2.0
