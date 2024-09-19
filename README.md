# Churn Prediction Model for Direct Debit

## Task
This task is to build a churn prediction model in direct debit.

## Provided Submission Files
* Jupyter notebook html report (`churn_creditor_model_eda.html`) covering 
the end to end data analysis and models evaluation & selection
* Model file: `churn_model.pkl` (`config` folder) 
* Python scripts (`run.py` and `config.py`) to run the end to end process
* Output probability file: `prediction.csv`

## Data
After joining `creditors`, `mandates` and `payments` tables together, 
I apply a rolling window on payments' `created_at` column and then transform to get training data
(see `rolling_tranform` function in jupyter notebook). 

As I think quarterly data would bear more information of business activity patterns, I compared 3, 6, 9 months for the rolling window size.
I select 6-month windows as ML models generally perform better on the 6-month data.

## Model Pipeline
### Features
According to feature analysis, I use the following features for each `creditor` in each 6-month time window:
* `has_logo`
* `merchant_type`
* `refunds_enabled`
* `pct_payments_require_approval`: the percentage of mandates that its `payments_require_approval`
* `pct_is_business_customer_type`: the percentage of mandates which `is_business_customer_type`
* `num_mandates`:  number of mandates
* `amount_sum`:  total amount
* `num_payments`: number of payments 
* `active_aging`:  the difference (in days) between the last `payment_created_at` and the first `payment_created_at` 
* `pct_has_ref`: the percentage of payments with reference
* `pct_source_api`: the percentage of payments via api
* `pct_source_app`: the percentage of payments via app

Function `process_creditor_data` in `run.py` and  generates these features and adds to the data


### Model Evaluation and Selection
I use `DummyClassifier` as a base model and `Logistic Regression`, `SVM`, `Decision Tree Classifier` and `Random Forest Classifier`
as candidate models.

I use cross validation and compare the model performance based on F1 score, precision, recall and ROC AUC. 
I also use SMOTE to over-sample the minority class and evaluate the performance.

The overall performance is: 
`RandomForestClassifier > DecisionTreeClassifier > Logistic Regression > SVM > DummyClassifier`

`RandomForestClassifier` is used for fine-tuning as final model

