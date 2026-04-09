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
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predictor de Viviendas Valencia</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', sans-serif; background: #f0f4f8; color: #333; }

        header {
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            color: white; padding: 30px 40px;
        }
        header h1 { font-size: 1.8rem; }
        header p  { margin-top: 6px; opacity: 0.75; font-size: 0.95rem; }

        .container { max-width: 860px; margin: 40px auto; padding: 0 20px; }

        .card {
            background: white; border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08); padding: 30px; margin-bottom: 24px;
        }
        .card h2 { font-size: 1.1rem; color: #1a1a2e; margin-bottom: 20px;
                   border-bottom: 2px solid #e8edf2; padding-bottom: 10px; }

        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }

        .field label { display: block; font-size: 0.82rem; font-weight: 600;
                       color: #555; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.4px; }
        .field input, .field select {
            width: 100%; padding: 10px 12px; border: 1.5px solid #dde3ea;
            border-radius: 8px; font-size: 0.95rem; transition: border 0.2s;
        }
        .field input:focus, .field select:focus {
            outline: none; border-color: #4f8ef7;
        }

        .btn {
            width: 100%; padding: 14px; background: linear-gradient(135deg, #4f8ef7, #1a5fd4);
            color: white; border: none; border-radius: 8px; font-size: 1rem;
            font-weight: 600; cursor: pointer; margin-top: 10px; transition: opacity 0.2s;
        }
        .btn:hover { opacity: 0.9; }

        #resultado { display: none; }
        .resultado-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 4px; }
        .metrica {
            background: #f0f4f8; border-radius: 10px; padding: 20px; text-align: center;
        }
        .metrica .valor { font-size: 1.9rem; font-weight: 700; color: #1a5fd4; }
        .metrica .etiqueta { font-size: 0.8rem; color: #777; margin-top: 4px; text-transform: uppercase; }

        .error-box {
            background: #fff0f0; border: 1px solid #f5c6c6; border-radius: 8px;
            padding: 14px; color: #c0392b; font-size: 0.9rem; margin-top: 12px;
        }

        .tag {
            display: inline-block; background: #eef3ff; color: #1a5fd4;
            border-radius: 20px; padding: 3px 10px; font-size: 0.78rem; font-weight: 600;
            margin-right: 6px; margin-bottom: 4px;
        }
    </style>
</head>
<body>

<header>
    <h1>🏠 Predictor de Precios de Viviendas — Valencia</h1>
    <p>Predice el precio (€/m²) y el precio total estimado</p>
</header>

<div class="container">

    <!-- FORMULARIO -->
    <div class="card">
        <h2>Características de la vivienda</h2>
        <div class="grid">
            <div class="field" style="grid-column: span 2;">
                <label>Barrio *</label>
                <select id="barrio" onchange="actualizarCoordenadas()">
                    <option value="">— Selecciona un barrio —</option>
                    <option value="39.4831,-0.3769">Benimaclet</option>
                    <option value="39.4780,-0.3530">Albors</option>
                    <option value="39.4750,-0.3640">Amistat</option>
                    <option value="39.4920,-0.3580">Benifaraig</option>
                    <option value="39.4900,-0.3990">Beniferri</option>
                    <option value="39.4990,-0.3770">Borbotó</option>
                    <option value="39.4850,-0.3690">Camí de Vera</option>
                    <option value="39.4720,-0.3760">Campanar</option>
                    <option value="39.4680,-0.3740">Carme</option>
                    <option value="39.4660,-0.3720">Ciutat Vella</option>
                    <option value="39.4640,-0.3550">El Cabanyal</option>
                    <option value="39.4580,-0.3530">El Canyamelar</option>
                    <option value="39.4700,-0.3530">El Grao</option>
                    <option value="39.4550,-0.3760">El Pla del Real</option>
                    <option value="39.4590,-0.3870">En Corts</option>
                    <option value="39.4530,-0.3820">Exposició</option>
                    <option value="39.4480,-0.3790">Favara</option>
                    <option value="39.4820,-0.3820">Fuensanta</option>
                    <option value="39.4700,-0.3900">Jesús</option>
                    <option value="39.4750,-0.3800">La Creu Coberta</option>
                    <option value="39.4770,-0.3850">La Creu del Grau</option>
                    <option value="39.4680,-0.3810">La Petxina</option>
                    <option value="39.4650,-0.3790">La Roqueta</option>
                    <option value="39.4630,-0.3760">La Seu</option>
                    <option value="39.4610,-0.3780">La Xerea</option>
                    <option value="39.4560,-0.3840">L'Hort de Senabre</option>
                    <option value="39.4740,-0.3710">Marxalenes</option>
                    <option value="39.4800,-0.3750">Mestalla</option>
                    <option value="39.4510,-0.3760">Monteolivete</option>
                    <option value="39.4870,-0.3730">Morvedre</option>
                    <option value="39.4760,-0.3780">Nou Moles</option>
                    <option value="39.4690,-0.3870">Patraix</option>
                    <option value="39.4840,-0.3610">Penya-roja</option>
                    <option value="39.4760,-0.3660">Poble Nou</option>
                    <option value="39.4730,-0.3870">Russafa</option>
                    <option value="39.4670,-0.3850">Sant Francesc</option>
                    <option value="39.4860,-0.3860">Sant Pau</option>
                    <option value="39.4790,-0.3910">Soternes</option>
                    <option value="39.4810,-0.3780">Tormos</option>
                    <option value="39.4880,-0.3810">Trinitat</option>
                    <option value="39.4940,-0.3840">Torrefiel</option>
                    <option value="39.4660,-0.3690">Velluters</option>
                    <option value="39.4630,-0.3820">Vara de Quart</option>
                    <option value="39.4530,-0.3700">Viveros</option>
                </select>
            </div>
            <input type="hidden" id="latitud">
            <input type="hidden" id="longitud">
            <div class="field">
                <label>Dormitorios *</label>
                <input type="number" id="dormitorios" min="0" max="20" placeholder="3" value="3">
            </div>
            <div class="field">
                <label>Baños *</label>
                <input type="number" id="banos" min="0" max="10" placeholder="2" value="2">
            </div>
            <div class="field">
                <label>Superficie (m²) *</label>
                <input type="number" id="superficie" min="10" placeholder="90" value="90">
            </div>
            <div class="field">
                <label>Demanda</label>
                <select id="demanda">
                    <option value="">— Sin especificar —</option>
                    <option value="Alta">Alta</option>
                    <option value="Media">Media</option>
                    <option value="Baja">Baja</option>
                    <option value="Muy baja">Muy baja</option>
                </select>
            </div>
            <div class="field">
                <label>Fuente</label>
                <select id="fuente">
                    <option value="">— Sin especificar —</option>
                    <option value="Fotocasa">Fotocasa</option>
                    <option value="Idealista">Idealista</option>
                </select>
            </div>
            <div class="field">
                <label>Anunciante</label>
                <select id="anunciante">
                    <option value="">— Sin especificar —</option>
                    <option value="Particular">Particular</option>
                    <option value="Agencia">Agencia</option>
                </select>
            </div>
            <div class="field">
                <label>Aire acondicionado</label>
                <select id="aire_acondicionado">
                    <option value="">— Sin especificar —</option>
                    <option value="Sí">Sí</option>
                    <option value="No">No</option>
                </select>
            </div>
            <div class="field">
                <label>Ascensor</label>
                <select id="ascensor">
                    <option value="">— Sin especificar —</option>
                    <option value="Sí">Sí</option>
                    <option value="No">No</option>
                </select>
            </div>
            <div class="field">
                <label>Garaje</label>
                <select id="garaje">
                    <option value="">— Sin especificar —</option>
                    <option value="Sí">Sí</option>
                    <option value="No">No</option>
                </select>
            </div>
            <div class="field">
                <label>Trastero</label>
                <select id="trastero">
                    <option value="">— Sin especificar —</option>
                    <option value="Sí">Sí</option>
                    <option value="No">No</option>
                </select>
            </div>
            <div class="field">
                <label>Terraza</label>
                <select id="terraza">
                    <option value="">— Sin especificar —</option>
                    <option value="Sí">Sí</option>
                    <option value="No">No</option>
                </select>
            </div>
            <div class="field">
                <label>Piscina</label>
                <select id="piscina">
                    <option value="">— Sin especificar —</option>
                    <option value="Sí">Sí</option>
                    <option value="No">No</option>
                </select>
            </div>
            <div class="field">
                <label>Zonas verdes</label>
                <select id="zonas_verdes">
                    <option value="">— Sin especificar —</option>
                    <option value="Sí">Sí</option>
                    <option value="No">No</option>
                </select>
            </div>
            <div class="field">
                <label>Zona deportiva</label>
                <select id="zona_deportiva">
                    <option value="">— Sin especificar —</option>
                    <option value="Sí">Sí</option>
                    <option value="No">No</option>
                </select>
            </div>
        </div>
        <button class="btn" onclick="predecir()">🔍 Predecir precio</button>
    </div>

    <!-- RESULTADO -->
    <div class="card" id="resultado">
        <h2>Resultado de la predicción</h2>
        <div class="resultado-grid">
            <div class="metrica">
                <div class="valor" id="precio-unitario">—</div>
                <div class="etiqueta">Precio unitario (€/m²)</div>
            </div>
            <div class="metrica">
                <div class="valor" id="precio-total">—</div>
                <div class="etiqueta">Precio total estimado</div>
            </div>
        </div>
    </div>

</div>

<script>
function actualizarCoordenadas() {
    const barrio = document.getElementById('barrio');
    const coords = barrio.value.split(',');
    document.getElementById('latitud').value  = coords[0] || '';
    document.getElementById('longitud').value = coords[1] || '';
}

async function predecir() {
    const params = new URLSearchParams();

    if (!document.getElementById('barrio').value) {
        alert('Por favor selecciona un barrio');
        return;
    }

    const obligatorios = ['latitud', 'longitud', 'dormitorios', 'banos', 'superficie'];
    for (const campo of obligatorios) {
        const val = document.getElementById(campo).value;
        if (!val) { alert('Por favor rellena todos los campos obligatorios (*)'); return; }
        params.append(campo, val);
    }

    const opcionales = ['demanda', 'fuente', 'anunciante', 'aire_acondicionado',
                        'ascensor', 'garaje', 'trastero', 'terraza',
                        'piscina', 'zonas_verdes', 'zona_deportiva'];
    for (const campo of opcionales) {
        const val = document.getElementById(campo).value;
        if (val) params.append(campo, val);
    }

    try {
        const res  = await fetch('/api/v1/predict?' + params.toString());
        const data = await res.json();

        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }

        const fmt = n => new Intl.NumberFormat('es-ES', {
            style: 'currency', currency: 'EUR', maximumFractionDigits: 0
        }).format(n);

        document.getElementById('precio-unitario').textContent = fmt(data.precio_unitario_eur_m2);
        document.getElementById('precio-total').textContent    = fmt(data.precio_total_estimado);
        document.getElementById('resultado').style.display     = 'block';
        document.getElementById('resultado').scrollIntoView({ behavior: 'smooth' });

    } catch(e) {
        alert('Error al conectar con la API');
    }
}
</script>

</body>
</html>
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
