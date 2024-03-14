# ecs171_ml_project

## experiment_results/


> ### stroke_prediction.ipynb
This notebook is where we did our exploratory data analysis and created the models and did hyperparameter tuning with Grid Search. 

> ### cross_validation and final model.ipynb
This notebook contains the code for final preprocessing and creating the models, along with cross validation metrics displayed. It exports the models as well

> ### data.csv, test.csv
data contains the kaggle dataset and test is 20% stored for testing

## {encoder, models, scaler}/
These directories contain exported onehot encoder, min max scaler, and all the models

## final_report
Contains the LaTeX final report.

## frontend.py
This is where the code to run the streamlit app is.

### To run the app
```streamlit run frontend.py```