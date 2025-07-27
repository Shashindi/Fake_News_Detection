from flask import Flask, render_template, request
import pickle
import os

# Define paths
model_path = os.path.join('model', 'fake_news_model.pkl')
vectorizer_path = os.path.join('model', 'vectorizer.pkl')

# Load model and vectorizer
with open(model_path, 'rb') as f_model, open(vectorizer_path, 'rb') as f_vector:
    model = pickle.load(f_model)
    vectorizer = pickle.load(f_vector)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    news_text = ""

    if request.method == 'POST':
        news_text = request.form.get('news', '').strip().lower()
        print(f"Received text: {news_text}")  # Debug input

        if not news_text:
            prediction = "No text provided!"
            return render_template('index.html', prediction=prediction)

        try:
            transformed_text = vectorizer.transform([news_text])
            prediction_raw = model.predict(transformed_text)[0]
            print(f"Model raw prediction: {prediction_raw}")  # Debug model output

            # ✅ Proper indentation
            if prediction_raw == 0:
                prediction = "FAKE"
            else:
                prediction = "REAL"

        except Exception as e:
            print(f"Error during prediction: {e}")
            prediction = "Error in prediction."

        return render_template('result.html', prediction=prediction, text=news_text)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5010)
