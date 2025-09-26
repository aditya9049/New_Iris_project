from flask import Flask, render_template, request
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the tuned Random Forest model
rf_model = joblib.load("tuned_random_forest_iris.pkl")

# Map target labels to flower names
target_names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        try:
            # Get input values from form
            sepal_length = float(request.form["sepal_length"])
            sepal_width = float(request.form["sepal_width"])
            petal_length = float(request.form["petal_length"])
            petal_width = float(request.form["petal_width"])

            # Create DataFrame from inputs
            input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                      columns=['sepal length (cm)', 'sepal width (cm)',
                                               'petal length (cm)', 'petal width (cm)'])

            # Predict using tuned Random Forest
            pred_class = rf_model.predict(input_data)[0]
            prediction = target_names[pred_class]

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
