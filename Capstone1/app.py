
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    artifacts = pickle.load(f)

model = artifacts["model"]
num_imputer = artifacts["num_imputer"]
num_cols = artifacts["num_cols"]

@app.route("/")
def home():
    return render_template("index.html", columns=num_cols)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values safely
        values = []
        for col in num_cols:
            val = request.form.get(col)
            print(f"{col} = {val}")   # Debug print
            values.append(float(val))

        input_df = pd.DataFrame([values], columns=num_cols)

        # Apply preprocessing
        input_df[num_cols] = num_imputer.transform(input_df[num_cols])

        prediction = model.predict(input_df[num_cols])[0]
        print("Prediction:", prediction)

        return render_template("index.html",
                               columns=num_cols,
                               prediction_text=f"Predicted Parts Per Hour: {prediction:.2f}")

    except Exception as e:
        print("ERROR:", e)
        return render_template("index.html",
                               columns=num_cols,
                               prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
