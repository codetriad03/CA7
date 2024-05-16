import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def split_data(data_dir):
    train_path = data_dir + "/train.csv"
    test_path = data_dir + "/test.csv"
    validation_path = data_dir + "/validation.csv"

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    validation_data = pd.read_csv(validation_path)

    return train_data, test_data, validation_data

# Assuming your data is in the 'data' folder
data_dir = 'data'
train_data, test_data, validation_data = split_data(data_dir)

# Assuming 'headline' and 'source' are column names
X_train = train_data[['headline', 'source']]
y_train = train_data['headline']

X_test = test_data[['headline', 'source']]
y_test = test_data['headline']

X_val = validation_data[['headline', 'source']]
y_val = validation_data['headline']

# Preprocess the data (e.g., vectorize the text data)
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['headline'] + " " + X_train['source'])
X_test_tfidf = tfidf_vectorizer.transform(X_test['headline'] + " " + X_test['source'])
X_val_tfidf = tfidf_vectorizer.transform(X_val['headline'] + " " + X_val['source'])

# Train a model (e.g., logistic regression) on the training set
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Validate the model
y_val_pred = model.predict(X_val_tfidf)
validation_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", validation_accuracy)

# Evaluate generalization ability on the testing set
y_test_pred = model.predict(X_test_tfidf)
testing_accuracy = accuracy_score(y_test, y_test_pred)
print("Testing Accuracy:", testing_accuracy)

# Add any other metrics or evaluation you need
# For example, classification report
print("Classification Report:")
print(classification_report(y_test, y_test_pred))
