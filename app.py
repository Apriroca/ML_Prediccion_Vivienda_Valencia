from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
# Asegúrate de tener lgbm instalado si es el modelo que usaste en el notebook
# from lightgbm import LGBMRegressor 

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)

MODEL_PATH = 'modelo_random_forest.pkl'

# --- CONFIGURACIÓN BASADA EN TU MAIN.IPYNB ---
# Definimos las columnas numéricas que usas
features_num = [
    'Metros cuadrados', 'Habitaciones', 'Baños', 'Planta', 
    'Ascensor', 'Antigüedad', 'Estado_conservacion' # Ajusta estos nombres a tus features_num reales
]

# Definimos los prefijos de tus columnas One-Hot Encoding
prefixes_ohe = [
    'Fuente_', 'Distrito_', 'Barrio_', 'Anunciante_',
    'Aire acondicionado_', 'Ascensor_', 'Garaje_', 'Trastero_',
    'Terraza_', 'Piscina_', 'Zonas verdes_', 'Zona deportiva_', 'Demanda_'
]

# Función para cargar el modelo
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model()

@app.route("/", methods=["GET"])
def hello():
    return "API de Predicción Inmobiliaria (Adaptada de main.ipynb)"

@app.route("/api/v1/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Modelo no cargado."}), 400

    data = request.get_json()
    if not data:
        return jsonify({"error": "No se proporcionaron datos."}), 400

    # Convertimos la entrada en DataFrame
    input_df = pd.DataFrame([data])
    
    try:
        # Recuperamos las features que el modelo espera (las que usaste en el entrenamiento)
        # Si usaste model.fit(X, y), scikit-learn guarda los nombres en feature_names_in_
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
            # Reindexamos: añade las columnas que faltan con 0 y elimina las que sobran
            input_df = input_df.reindex(columns=expected_features, fill_value=0)
        
        prediction = model.predict(input_df)
        
        return jsonify({
            'precio_unitario_predicho': float(prediction[0]),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({"error": f"Error en la predicción: {str(e)}"}), 500

@app.route("/api/v1/retrain", methods=["POST"])
def retrain():
    global model
    target = 'Precio unitario'
    
    payload = request.get_json()
    if not payload or 'datos' not in payload:
        return jsonify({"error": "Envía los datos en la clave 'datos'"}), 400

    # Creamos el DataFrame con los nuevos datos enviados por el usuario
    new_data = pd.DataFrame(payload['datos'])

    if target not in new_data.columns:
        return jsonify({"error": f"Falta la columna objetivo: {target}"}), 400

    try:
        # Aplicamos la misma lógica de selección de columnas de tu notebook
        cols_ohe = [c for c in new_data.columns if any(
            c.startswith(p) for p in prefixes_ohe
        )]
        
        # 'features_num' debe estar presente en los datos enviados
        features = [f for f in features_num if f in new_data.columns] + cols_ohe
        
        X = new_data[features].copy()
        y = new_data[target]

        # Splitting (mismo random_state que tu notebook)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        # Reentrenamos (el modelo mantiene su configuración, ej. hiperparámetros de LightGBM)
        model.fit(X_train, y_train)
        
        # Métricas
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = mean_absolute_percentage_error(y_test, preds)

        # Guardar modelo actualizado
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)

        return jsonify({
            "message": "Modelo actualizado con éxito",
            "features_usadas": len(features),
            "metricas": {
                "RMSE": float(rmse),
                "MAPE": float(mape)
            }
        })

    except Exception as e:
        return jsonify({"error": f"Error reentrenando: {str(e)}"}), 500

if __name__ == '__main__':
    # El modelo debe estar en la misma carpeta que este script
    app.run(debug=True, port=5000)