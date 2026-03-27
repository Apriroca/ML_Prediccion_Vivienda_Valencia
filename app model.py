from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np    

os.chdir(os.path.dirname(__file__))


app = Flask(__name__)

# Definir las rutas a los modelos
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'src', 'models', 'modelo_viviendas_valencia.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'src', 'models', 'scaler.pkl')

# Cargar el modelo y el escalador
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return "<h1>API Predictiva - Viviendas en Valencia</h1>"

@app.route('/api/v1/predict', methods=['GET'])
def predict():
    try:
        # Obtener los datos de la URL enviados por el usuario
        # Se asumen 'Dormitorios', 'Baños' y 'Superficie' como características base (ajusta según tus variables exactas)
        dormitorios = float(request.args.get('dormitorios', 1))
        banos = float(request.args.get('banos', 1))
        superficie = float(request.args.get('superficie', 50))
        
        # Crear un DataFrame con las variables en el mismo orden que el entrenamiento
        nuevo_dato = pd.DataFrame({
            'Dormitorios': [dormitorios],
            'Baños': [banos],
            'Superficie': [superficie]
            # Añade aquí más variables si tu modelo XGBoost las requiere (ej. Latitud, Longitud)
        })
        
        # Escalar los datos usando el MinMaxScaler cargado
        dato_escalado = scaler.transform(nuevo_dato)
        
        # Predecir con el modelo XGBoost
        prediccion = model.predict(dato_escalado)
        
        # Devolver el resultado en JSON
        return jsonify({'precio_estimado': float(prediccion[0])})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)