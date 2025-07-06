import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Try to load the model at startup
try:
    model = pickle.load(open("ufo-model.pkl", "rb"))
except:
    model = None  # If not trained yet


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get form values and convert to integer array
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    # Make prediction
    prediction = model.predict(final_features)
    output = prediction[0]

    # Convert output code to country name
    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template(
        "index.html",
        prediction_text="Likely country: {}".format(countries[output]),
    )


@app.route("/train")
def train():
    global model  # Needed to reuse the model in /predict

    # Step 1: Load CSV (relative to this file's location)
    df = pd.read_csv("../data/ufos.csv")

    # Step 2: Drop rows with missing values
    df = df.dropna()

    # Step 3: Keep only selected countries
    df = df[df["country"].isin(["au", "ca", "de", "gb", "us"])]

    # Step 4: Create numerical labels for countries
    df["country_code"] = df["country"].map({"au": 0, "ca": 1, "de": 2, "gb": 3, "us": 4})

    # Step 5: Select the correct column name for duration
    # NOTE: The real column is 'duration (seconds)', not 'seconds'
    X = df[["duration (seconds)", "latitude", "longitude"]]
    y = df["country_code"]

    # Step 6: Train the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Step 7 (Optional): Save model to .pkl file
    pickle.dump(model, open("ufo-model.pkl", "wb"))

    return "✅ Model trained successfully!"


if __name__ == "__main__":
    app.run(debug=True)
