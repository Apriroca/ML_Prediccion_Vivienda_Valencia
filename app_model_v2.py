from flask import Flask, jsonify, request
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Carga del modelo, scaler y features al arrancar la app
# ---------------------------------------------------------------------------
with open('src/models/modelo_viviendas_valencia.pkl', 'rb') as f:
    model = pickle.load(f)

with open('src/models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('src/models/features.pkl', 'rb') as f:
    MODEL_FEATURES = pickle.load(f)

FEATURES_NUM = ['Latitud', 'Longitud', 'Dormitorios', 'Baños', 'Superficie']

CAT_COLS = [
    'Fuente', 'Anunciante', 'Aire acondicionado',
    'Ascensor', 'Garaje', 'Trastero', 'Terraza', 'Piscina',
    'Zonas verdes', 'Zona deportiva', 'Demanda'
]

# Mapeo nombre query param → nombre columna original (para evitar espacios en la URL)
PARAM_MAP = {
    'fuente':             'Fuente',
    'anunciante':         'Anunciante',
    'aire_acondicionado': 'Aire acondicionado',
    'ascensor':           'Ascensor',
    'garaje':             'Garaje',
    'trastero':           'Trastero',
    'terraza':            'Terraza',
    'piscina':            'Piscina',
    'zonas_verdes':       'Zonas verdes',
    'zona_deportiva':     'Zona deportiva',
    'demanda':            'Demanda',
}


def preprocess(data: dict) -> pd.DataFrame:
    """Aplica el mismo preprocesado que en el entrenamiento."""
    df = pd.DataFrame([data])
    cols_presentes = [c for c in CAT_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cols_presentes, dtype=int)
    df = df.reindex(columns=MODEL_FEATURES, fill_value=0)
    df[FEATURES_NUM] = scaler.transform(df[FEATURES_NUM])
    return df


# ---------------------------------------------------------------------------
# ENDPOINT 1 — Landing page  /
# ---------------------------------------------------------------------------
@app.route('/', methods=['GET'])
def hello():
    return """
    <h1>🏠 API de Predicción de Precios de Viviendas en Valencia</h1>
    <p>Modelo: <b>XGBoost Tuned</b> — predice el precio unitario (€/m²) a partir
    de las características de la vivienda.</p>

    <hr>
    <h2>Endpoints</h2>

    <h3>📌 GET /api/v1/predict</h3>
    <p>Devuelve el precio unitario estimado (€/m²) y el precio total.</p>

    <b>Parámetros obligatorios:</b>
    <ul>
        <li><code>latitud</code> — Ej: 39.47</li>
        <li><code>longitud</code> — Ej: -0.37</li>
        <li><code>dormitorios</code> — Ej: 3</li>
        <li><code>banos</code> — Ej: 2</li>
        <li><code>superficie</code> — m². Ej: 90</li>
    </ul>

    <b>Parámetros opcionales:</b>
    <ul>
        <li><code>fuente</code> — Fotocasa / Idealista</li>
        <li><code>anunciante</code> — Particular / Agencia</li>
        <li><code>aire_acondicionado</code> — Sí / No</li>
        <li><code>ascensor</code> — Sí / No</li>
        <li><code>garaje</code> — Sí / No</li>
        <li><code>trastero</code> — Sí / No</li>
        <li><code>terraza</code> — Sí / No</li>
        <li><code>piscina</code> — Sí / No</li>
        <li><code>zonas_verdes</code> — Sí / No</li>
        <li><code>zona_deportiva</code> — Sí / No</li>
        <li><code>demanda</code> — Alta / Media / Baja / Muy baja</li>
    </ul>

    <b>Ejemplo:</b><br>
    <code>/api/v1/predict?latitud=39.47&longitud=-0.37&dormitorios=3&banos=2&superficie=90&demanda=Alta&ascensor=Sí&aire_acondicionado=Sí</code>

    <b>Respuesta:</b>
    <pre>{"precio_total_estimado": 270000.0, "precio_unitario_eur_m2": 3000.0}</pre>

    <hr>
    <h3>📌 GET /api/v1/retrain</h3>
    <p>Reentrena el modelo con nuevos datos si existe la carpeta
    <code>src/data_sample/ventas_new/</code> con ficheros CSV.</p>
    """


# ---------------------------------------------------------------------------
# ENDPOINT 2 — Predicción  /api/v1/predict
# ---------------------------------------------------------------------------
@app.route('/api/v1/predict', methods=['GET'])
def predict():
    try:
        latitud     = request.args.get('latitud',     type=float)
        longitud    = request.args.get('longitud',    type=float)
        dormitorios = request.args.get('dormitorios', type=float)
        banos       = request.args.get('banos',       type=float)
        superficie  = request.args.get('superficie',  type=float)

        missing = [name for name, val in [
            ('latitud', latitud), ('longitud', longitud),
            ('dormitorios', dormitorios), ('banos', banos),
            ('superficie', superficie)
        ] if val is None]

        if missing:
            return jsonify({'error': f"Faltan parámetros obligatorios: {', '.join(missing)}"}), 400

        input_data = {
            'Latitud':     latitud,
            'Longitud':    longitud,
            'Dormitorios': dormitorios,
            'Baños':       banos,
            'Superficie':  superficie,
        }
        for param, col in PARAM_MAP.items():
            val = request.args.get(param)
            if val is not None:
                input_data[col] = val

        X = preprocess(input_data)
        precio_unitario = float(model.predict(X)[0])

        return jsonify({
            'precio_unitario_eur_m2': round(precio_unitario, 2),
            'precio_total_estimado':  round(precio_unitario * superficie, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# ENDPOINT 3 — Info del modelo  /api/v1/model-info
# PARA LA DEMO EN CLASE: descomentar este bloque, hacer git push y redesplegar
# ---------------------------------------------------------------------------
# @app.route('/api/v1/model-info', methods=['GET'])
# def model_info():
#     """Devuelve los hiperparámetros del modelo actualmente en producción."""
#     return jsonify({
#         'modelo':           'XGBoost Regressor (Tuned)',
#         'target':           'Precio unitario (euros/m2)',
#         'n_features':       len(MODEL_FEATURES),
#         'top_features':     MODEL_FEATURES[:10],
#         'hiperparametros':  model.get_params()
#     })


# ---------------------------------------------------------------------------
# ENDPOINT EXTRA — Reentrenamiento  /api/v1/retrain
# ---------------------------------------------------------------------------
@app.route('/api/v1/retrain', methods=['GET'])
def retrain():
    global model
    nueva_ruta = 'src/data_sample/ventas_new/'

    if not os.path.exists(nueva_ruta):
        return "<h2>No se encontraron datos nuevos para reentrenar. Nothing done!</h2>"

    try:
        archivos = [f for f in os.listdir(nueva_ruta) if f.endswith('.csv')]
        if not archivos:
            return "<h2>La carpeta existe pero no contiene ficheros CSV. Nothing done!</h2>"

        df = pd.concat(
            [pd.read_csv(nueva_ruta + f) for f in archivos],
            ignore_index=True
        )

        # Mismo preprocesado que en el entrenamiento
        df['Referencia_interna'] = df['Latitud'].astype(str) + df['Longitud'].astype(str)
        df = df.sort_values('Precio').drop_duplicates(subset=['Referencia_interna'])
        df = df.dropna(subset=['Barrio', 'Baños', 'Dormitorios'])
        df = df[(df['Precio unitario'] >= 500) & (df['Precio unitario'] <= 15000)]

        columnas_a_eliminar = [
            'Tipologia', 'Titulo', 'Provincia', 'Municipio', 'C.P.',
            'Planta', 'Empresa', 'Imagen', 'URL', 'Telefono', 'Email',
            'Posible agencia', 'Referencia', 'Operacion', 'Fecha de creacion',
            'Comentarios', 'Descartado', 'Estado', 'Ranking', 'Dias',
            'Conservacion', 'Referencia_interna'
        ]
        df = df.drop(columns=columnas_a_eliminar, errors='ignore')
        df = pd.get_dummies(df, columns=CAT_COLS, dtype=int)

        X = df.drop(columns=['Precio unitario']).reindex(columns=MODEL_FEATURES, fill_value=0)
        y = df['Precio unitario']
        X[FEATURES_NUM] = scaler.transform(X[FEATURES_NUM])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)

        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        mae  = mean_absolute_error(y_test, model.predict(X_test))
        r2   = r2_score(y_test, model.predict(X_test))

        # Reentrenamos con todos los datos y persistimos
        model.fit(X, y)
        with open('src/models/modelo_viviendas_valencia.pkl', 'wb') as f:
            pickle.dump(model, f)

        return (f"Modelo reentrenado con {len(df)} registros nuevos. "
                f"RMSE: {round(rmse, 2)} €/m² | MAE: {round(mae, 2)} €/m² | R²: {round(r2, 3)}")

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
