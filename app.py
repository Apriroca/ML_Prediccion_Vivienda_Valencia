from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)

MODEL_PATH = 'modelo_random_forest.pkl'

# Carga el modelo al iniciar la API
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

@app.route("/", methods=["GET"])
def hello():
    return "Bienvenido a la API del Modelo Inmobiliario"

@app.route("/api/v1/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "El modelo no está cargado. Reentrena primero."}), 400

    data = request.get_json()
    if not data:
        return jsonify({"error": "No se enviaron datos."}), 400

    input_data = pd.DataFrame([data])
    
    try:
        if hasattr(model, 'feature_names_in_'):
            expected_cols = model.feature_names_in_
            input_data = input_data.reindex(columns=expected_cols, fill_value=0)

        prediction = model.predict(input_data)
        return jsonify({'prediccion_precio_unitario': float(prediction[0])})
    
    except Exception as e:
        return jsonify({"error": f"Error al predecir: {str(e)}"}), 500


@app.route("/api/v1/retrain", methods=["POST"])
def retrain():
    global model
    target_col = 'Precio unitario'
    
    # 1. Recibimos el payload JSON que contiene los datos
    payload = request.get_json()
    
    if not payload or 'datos_nuevos' not in payload:
        return jsonify({"error": "Debes enviar un JSON con la clave 'datos_nuevos' que contenga una lista de registros."}), 400

    # 2. Convertimos la lista de diccionarios en un DataFrame
    data = pd.DataFrame(payload['datos_nuevos'])

    if target_col not in data.columns:
        return jsonify({"error": f"La columna '{target_col}' no existe en los datos enviados."}), 400

    # 3. Separar X e y
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Validar que haya suficientes datos para dividir
    if len(data) < 5:
        return jsonify({"error": "No hay suficientes datos para reentrenar. Envía al menos 5 registros."}), 400

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    try:
        # 4. Entrenar y evaluar
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mape = mean_absolute_percentage_error(y_test, predictions)
        
        # 5. Entrenar con el dataset completo para producción
        model.fit(X, y)

        # 6. Guardar el modelo actualizado
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)

        return jsonify({
            "message": "Modelo reentrenado exitosamente con los datos proporcionados.",
            "filas_procesadas": len(data),
            "metrics": {
                "RMSE": rmse,
                "MAPE": mape
            }
        })
    except Exception as e:
        return jsonify({"error": f"Error durante el reentrenamiento: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)