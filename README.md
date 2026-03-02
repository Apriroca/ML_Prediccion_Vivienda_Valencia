🏠 Análisis del Mercado Inmobiliario en Valencia: EDA y Predicción de Precios

📋 Descripción del Proyecto
Este proyecto consiste en un análisis exploratorio de datos (EDA) profundo y el desarrollo de un modelo predictivo sobre el precio de venta de viviendas en la ciudad de Valencia, España.

El objetivo principal es identificar los factores que más influyen en el valor de los inmuebles y construir un modelo de regresión capaz de estimar el precio de venta con precisión, ayudando a inversores y compradores a tomar decisiones basadas en datos.

🎯 Objetivos
EDA (Exploratory Data Analysis): Identificar patrones, tendencias y valores atípicos (outliers) en el mercado inmobiliario valenciano.

Limpieza de Datos: Tratamiento de valores nulos y filtrado de precios unitarios extremos para mejorar la calidad del análisis.

Feature Engineering: Selección de variables clave como ubicación (Latitud/Longitud), superficie y número de estancias.

Modelado: Implementación y evaluación de un algoritmo de Random Forest Regressor para la predicción de precios.

🗂️ Estructura del Repositorio
├── src/                # El directorio source que contiene el resto de carpetas
│   ├── data_sample/    # Archivos de datos de muestra (máx. 5MB) que permitan ejecutar el código
│   ├── img/            # Imágenes utilizadas en el proyecto
│   ├── models/         # Modelos guardados en formato pickle o joblib
│   ├── notebooks/      # Notebooks de desarrollo y pruebas
│   ├── utils/          # Módulos, funciones auxiliares o clases creadas para el proyecto
├── main.ipynb          # Notebook final: claro, conciso y bien estructurado
├── Presentacion.pdf    # Documento soporte de la exposición en vídeo
├── README.md           # Fichero README resumen del proyecto

📊 Insights Principales (Resultados del EDA)
(Aquí puedes añadir los hallazgos reales de tu notebook, por ejemplo:)

Localización: Se observa una correlación significativa entre la proximidad al centro/costa y el incremento del precio por metro cuadrado.

Superficie: Es la variable con mayor peso predictivo, seguida del número de baños.

Outliers: Se detectaron y filtraron propiedades con precios unitarios incongruentes que sesgaban el promedio del mercado.

🛠️ Tecnologías Utilizadas
Lenguaje: Python

Librerías de Análisis: Pandas, NumPy

Visualización: Matplotlib, Seaborn

Machine Learning: Scikit-Learn (Random Forest, MinMaxScaler, Train/Test Split)

🚀 Instalación y Uso
Clona el repositorio:

Bash
git clone https://github.com/Apriroca/ML_Prediccion_Vivienda_Valencia

Instala las dependencias:

Bash
pip install -r requirements.txt
Ejecuta el notebook principal:

Bash
jupyter notebook notebooks/main.ipynb

📈 Conclusiones del Modelo
El modelo final basado en Random Forest permite explicar una gran parte de la variabilidad de los precios (R² score). El análisis de importancia de variables (Feature Importance) confirmó que la ubicación geográfica y el tamaño de la vivienda son los pilares fundamentales del valor inmobiliario en Valencia.