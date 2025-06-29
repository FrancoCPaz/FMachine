from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd  # NUEVO

app = Flask(__name__)

# === Cargar modelo y scaler ===
model = joblib.load("random_forest_model.joblib")
scaler = joblib.load("scaler.joblib")

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    raw_input = {}
    transformed = []

    if request.method == "POST":
        try:
            # === Entrada cruda
            kills = float(request.form["kills"])
            flank_kills = float(request.form["flank_kills"])
            headshots = float(request.form["headshots"])

            raw_input = {
                "kills": kills,
                "flank_kills": flank_kills,
                "headshots": headshots
            }

            # === Transformación log1p
            transformed = np.log1p([kills, flank_kills, headshots])

            # Usar DataFrame con nombres correctos de columnas
            data = pd.DataFrame(
                [transformed],
                columns=["MatchKills_log", "MatchFlankKills_log", "MatchHeadshots_log"]
            )

            # === Escalado antes de la predicción
            data_scaled = scaler.transform(data)

            # === Predicción
            prediction = int(model.predict(data_scaled)[0])

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        raw_input=raw_input,
        transformed=transformed
    )

if __name__ == "__main__":
    app.run(debug=True)
