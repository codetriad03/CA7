import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
import mlflow
import joblib

# Get the experiment ID if it exists, otherwise create a new experiment
experiment_name = "Model_Selection"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment:
    experiment_id = experiment.experiment_id
else:
    experiment_id = mlflow.create_experiment(experiment_name)

# Load preprocessed data from CSV files
train_df = pd.read_csv('data/train.csv')
val_df = pd.read_csv('data/validation.csv')

# Extract features and labels
X_train = train_df['headline']
y_train = train_df['source']

X_val = val_df['headline']
y_val = val_df['source']

# Convert text data into numerical features
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)

# Train selected machine learning models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Support Vector Machine': SVC(),
    'Neural Network': MLPClassifier(max_iter=500, random_state=42)
}

best_model = None
best_score = 0.0

for name, model in models.items():
    with mlflow.start_run(experiment_id=experiment_id, run_name=name):
        if name == 'Neural Network':
            # Define parameter grid for MLPClassifier
            param_dist = {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1]
            }
            random_search = RandomizedSearchCV(model, param_dist, n_iter=5, cv=3, scoring='accuracy', random_state=42)
            random_search.fit(X_train_tfidf, y_train)

            # Train with best parameters
            best_model = random_search.best_estimator_
            best_model.fit(X_train_tfidf, y_train)
            y_pred = best_model.predict(X_val_tfidf)
        elif name in ['Gradient Boosting', 'Support Vector Machine']:
            # Define parameter distributions for RandomizedSearchCV
            if name == 'Gradient Boosting':
                param_dist = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.5],
                    'max_depth': [3, 5, 10],
                    'min_samples_split': [2, 5, 10]
                }
            else:  # Support Vector Machine
                param_dist = {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                    'gamma': ['scale', 'auto']
                }
            random_search = RandomizedSearchCV(model, param_dist, n_iter=5, cv=3, scoring='accuracy', random_state=42)
            random_search.fit(X_train_tfidf, y_train)

            # Train with best parameters
            best_model = random_search.best_estimator_
            best_model.fit(X_train_tfidf, y_train)
            y_pred = best_model.predict(X_val_tfidf)
        else:
            # Train the model
            model.fit(X_train_tfidf, y_train)

            # Evaluate the model
            y_pred = model.predict(X_val_tfidf)

        # Model evaluation
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='macro')
        recall = recall_score(y_val, y_pred, average='macro')
        f1 = f1_score(y_val, y_pred, average='macro')

        # Log parameters and metrics to MLflow
        mlflow.log_param('model', name)
        mlflow.log_param('max_features', 1000)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1_score', f1)

        if accuracy > best_score:
            best_score = accuracy
            best_model = model

        # Print evaluation metrics
        print(f'{name} Metrics:')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-score: {f1:.4f}')
        print('-----------------------------------------')

# Print the best-performing model
print(f"Best-performing model: {best_model}")

# Save the trained MLPClassifier model
joblib.dump(best_model, 'best_model.pkl')

# Save the TF-IDF vectorizer
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
