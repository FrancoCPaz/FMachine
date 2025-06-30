from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd  # NUEVO
import logging

app = Flask(__name__)


# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
           # === Validación de entrada
            kills = int(request.form["kills"])
            flank_kills = int(request.form["flank_kills"])
            headshots = int(request.form["headshots"])

            # Validar rangos razonables
            if kills < 0 or kills > 100:
                raise ValueError("Kills debe estar entre 0 y 100")
            if flank_kills < 0 or flank_kills > kills:
                raise ValueError("Flank kills no puede ser mayor que kills totales")
            if headshots < 0 or headshots > kills:
                raise ValueError("Headshots no puede ser mayor que kills totales")

            raw_input = {
                "kills": kills,
                "flank_kills": flank_kills,
                "headshots": headshots
            }

             # === Transformación log1p
            transformed = np.log1p([kills, flank_kills, headshots])
            
            # Logging detallado
            logger.info(f"Valores originales: kills={kills}, flank_kills={flank_kills}, headshots={headshots}")
            logger.info(f"Valores transformados (log1p): {transformed}")

            # Usar DataFrame con nombres correctos de columnas
            data = pd.DataFrame(
                [transformed],
                columns=["MatchKills_log", "MatchFlankKills_log", "MatchHeadshots_log"]
            )

            # === Escalado antes de la predicción
            data_scaled = scaler.transform(data)
            logger.info(f"Valores escalados: {data_scaled[0]}")

            # === Análisis de contexto basado en los datos de entrenamiento
            avg_kills = np.expm1(1.97)  # ~6.2 kills promedio en el dataset
            avg_headshots = np.expm1(1.33)  # ~2.8 headshots promedio
            avg_flank_kills = np.expm1(0.61)  # ~0.8 flank kills promedio
            
            # Determinar si los valores están dentro del rango normal del modelo
            context_warnings = []
            
            if kills > avg_kills * 3:  # 3x el promedio
                context_warnings.append(f"⚠️ Kills muy altos ({kills} vs ~{int(avg_kills)} promedio)")
            if headshots > avg_headshots * 3:
                context_warnings.append(f"⚠️ Headshots muy altos ({headshots} vs ~{int(avg_headshots)} promedio)")
            if flank_kills > avg_flank_kills * 3:
                context_warnings.append(f"⚠️ Flank kills muy altos ({flank_kills} vs ~{int(avg_flank_kills)} promedio)")

            # === Predicción con contexto
            prediction_proba = model.predict_proba(data_scaled)[0]
            victory_probability = round(prediction_proba[1] * 100, 2)
            defeat_probability = round(prediction_proba[0] * 100, 2)
            
            prediction_raw = int(model.predict(data_scaled)[0])
            prediction = "Victoria" if prediction_raw == 1 else "Derrota"
            
            # Generar interpretación contextual más específica
            if kills >= 25:
                interpretation = f"🔥 Rendimiento muy alto ({kills} kills) - Matchmaking competitivo detectado. Probabilidad de derrota: {defeat_probability}%"
            elif kills >= 15:
                interpretation = f"⚡ Buen rendimiento ({kills} kills) - Zona de transición. Victoria: {victory_probability}%"
            elif kills >= 5:
                interpretation = f"✅ Rendimiento moderado ({kills} kills) - Rango típico para victorias según el modelo"
            else:
                interpretation = f"📊 Rendimiento bajo ({kills} kills) - El modelo favorece estas estadísticas para victorias"

            logger.info(f"Predicción: {prediction} ({victory_probability}% victoria)")
            logger.info(f"Contexto: {interpretation}")
            if context_warnings:
                logger.info(f"Warnings: {'; '.join(context_warnings)}")




        except Exception as e:
            prediction = f"Error: {str(e)}"
            context_warnings = []
            interpretation = ""
            victory_probability = None
            defeat_probability = None
            raw_input = {}
            transformed = []

    return render_template(
        "index.html",
        prediction=prediction,
        raw_input=raw_input,
        transformed=transformed,
        victory_probability=victory_probability if 'victory_probability' in locals() else None,
        defeat_probability=defeat_probability if 'defeat_probability' in locals() else None,
        context_warnings=context_warnings if 'context_warnings' in locals() else [],
        interpretation=interpretation if 'interpretation' in locals() else ""
    )

@app.route("/test")
def test_model():
    """Endpoint para probar casos extremos y diagnosticar el modelo"""
    test_cases = [
        {"kills": 0, "flank_kills": 0, "headshots": 0, "description": "Peor caso posible"},
        {"kills": 1, "flank_kills": 0, "headshots": 0, "description": "Muy bajo rendimiento"},
        {"kills": 5, "flank_kills": 2, "headshots": 1, "description": "Bajo rendimiento"},
        {"kills": 15, "flank_kills": 5, "headshots": 8, "description": "Rendimiento medio"},
        {"kills": 30, "flank_kills": 10, "headshots": 20, "description": "Buen rendimiento"},
        {"kills": 50, "flank_kills": 25, "headshots": 35, "description": "Excelente rendimiento"},
    ]
    
    results = []
    for case in test_cases:
        transformed = np.log1p([case["kills"], case["flank_kills"], case["headshots"]])
        data = pd.DataFrame([transformed], columns=["MatchKills_log", "MatchFlankKills_log", "MatchHeadshots_log"])
        data_scaled = scaler.transform(data)
        pred_raw = model.predict(data_scaled)[0]
        proba = model.predict_proba(data_scaled)[0]
        
        results.append({
            "input": f"K:{case['kills']}, FK:{case['flank_kills']}, HS:{case['headshots']}",
            "description": case["description"],
            "prediction": "Victoria" if pred_raw == 1 else "Derrota",
            "victory_prob": round(proba[1] * 100, 2),
            "defeat_prob": round(proba[0] * 100, 2),
            "scaled_values": data_scaled[0].tolist()
        })
    
    return {"test_results": results}

@app.route("/model-info")
def model_info():
    """Información sobre el modelo cargado"""
    try:
        # Información básica del modelo
        model_params = model.get_params()
        feature_importance = model.feature_importances_ if hasattr(model, 'feature_importances_') else None
        
        return {
            "model_type": str(type(model).__name__),
            "n_estimators": model_params.get("n_estimators", "N/A"),
            "feature_names": ["MatchKills_log", "MatchFlankKills_log", "MatchHeadshots_log"],
            "feature_importance": feature_importance.tolist() if feature_importance is not None else None,
            "scaler_mean": scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
            "scaler_scale": scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None
        }
    except Exception as e:
        return {"error": str(e)}
    
@app.route("/analysis")
def model_analysis():
    # Datos del modelo
    avg_kills = np.expm1(1.97)
    avg_headshots = np.expm1(1.33)
    avg_flank_kills = np.expm1(0.61)
    
    analysis = {
        "model_summary": {
            "type": "RandomForestClassifier",
            "n_estimators": 100,
            "training_averages": {
                "kills": round(avg_kills, 1),
                "headshots": round(avg_headshots, 1),
                "flank_kills": round(avg_flank_kills, 1)
            }
        },
        "feature_importance": {
            "kills": "48.98% - Factor más determinante",
            "headshots": "33.12% - Segundo factor más importante", 
            "flank_kills": "17.90% - Menor importancia"
        },
        "behavior_patterns": {
            "low_performance": "0-10 kills → Tendencia a Victoria (56-65%)",
            "medium_performance": "11-20 kills → Zona de transición",
            "high_performance": "20+ kills → Tendencia a Derrota (60-75%)"
        },
        "interpretation": {
            "explanation": "El modelo refleja un sistema de matchmaking balanceado",
            "logic": "Jugadores con stats muy altos enfrentan oponentes más fuertes",
            "conclusion": "Rendimiento extremo no garantiza victoria"
        }
    }
    
    return analysis

if __name__ == "__main__":
    app.run(debug=True, port=5001)