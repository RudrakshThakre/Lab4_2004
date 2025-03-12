from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load model, scaler, and label encoder
with open("fish_model.pkl", "rb") as f:
    model, scaler, le = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from form
        data = [float(request.form[key]) for key in ["Length1", "Length2", "Length3", "Height", "Width"]]

        # Preprocess input
        data = scaler.transform([data])

        # Make prediction
        prediction = model.predict(data)
        species = le.inverse_transform(prediction)[0]

        return jsonify({"Predicted Species": species})
    
    except Exception as e:
        return jsonify({"Error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
