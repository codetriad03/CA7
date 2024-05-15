from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the best-performing model and preprocessing steps
model = joblib.load('best_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Preprocess input data
    input_text = data['headline']  # Assuming the JSON key is 'headline'
    input_vector = tfidf_vectorizer.transform([input_text])
    # Make prediction
    prediction = model.predict(input_vector)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
