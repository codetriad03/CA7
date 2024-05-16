# CA7

TASK 1:
The data was taken of News and for keeping the data update a task is scheduled in the Windows Task Scheduler to keep running the script for getting the news headlines every 5 minutes for indefinite time.

TASK 2:
In this step, the data was processed and evaluation was done on certain models to select the best fit model. The following steps were implemented:
1. Data Preparation:
   Ensured preprocessed data from DVC is split into training, validation, and testing sets.
2. Model Selection:
   Chose suitable ML algorithms like decision trees, random forests, SVMs, or neural networks.
3. Model Training with MLflow:
   Implemented training scripts using MLflow, logging parameters, metrics, and artifacts.
4. Hyperparameter Tuning (Optional):
   Optimized model performance with techniques like grid search or random search.
5. Model Evaluation:
   Evaluated models using predefined criteria such as accuracy, precision, recall, F1-score, or AUC.
6. Select Best Model:
   Chose the best-performing model based on evaluation results.
7. Model Deployment:
   Deployed the selected model for production, involving packaging, API creation, containerization, or integration.


TASK 3:
Testing Performance with Live Data:

1. Created an MLflow project-compatible repository containing the trained machine learning model and associated scripts.
2. Split the fetched live data into training and testing sets. Validated the model's performance using a validation set and evaluated its generalization ability on the testing set.
3. Executed the MLflow project script using live data fetched by DVC. Monitored the model's performance metrics and compared them with predefined thresholds or benchmarks.

Commands to Execute
To run the MLflow project script:

mlflow run https://github.com/account_name/CA7.git






