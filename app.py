import os
import logging
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get the base directory
BASE = os.path.dirname(os.path.dirname(__file__))
logger.debug(f"Base directory: {BASE}")

# Define paths
CSV_PATH = os.path.join(BASE, "data", "github.csv")
ENCODER_P = os.path.join(BASE, "models", "color_encoder.pkl")
METRICS_P = os.path.join(BASE, "models", "model_metrics.pkl")

# Load dataset to get categorical values
logger.debug("Loading dataset to get categorical values...")
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

# Define features
FEATURES = [
    'citric acid','urea','ethylenediamine','HCl','ammonium','NaOH',
    'boric acid','sodium thiosulfate','KOH','formic acid',
    'Reaction temperatrue( C )','reaction time (min)',
    'Reaction method','Solvent','pH','purification method'
]

# Get categorical options
CATEGORICAL_OPTIONS = {}
for col in ['Reaction method','Solvent','pH','purification method']:
    CATEGORICAL_OPTIONS[col] = sorted(df[col].unique().tolist())
    # Clean up the values
    CATEGORICAL_OPTIONS[col] = [str(x).strip() for x in CATEGORICAL_OPTIONS[col] if pd.notna(x)]
logger.debug(f"Categorical options: {CATEGORICAL_OPTIONS}")

# Define model names
MODEL_NAMES = ['LogReg', 'DecisionTree', 'RandomForest', 'SVM', 'ANN', 'XGBoost']
PIPE_PATHS = {name: os.path.join(BASE, "models", f"{name}_best.pkl") for name in MODEL_NAMES}

# Load models, encoder, and metrics
try:
    logger.debug("Loading models, encoder, and metrics...")
    pipelines = {}
    for name, path in PIPE_PATHS.items():
        if os.path.exists(path):
            logger.debug(f"Loading model: {name}")
            pipelines[name] = joblib.load(path)
        else:
            logger.error(f"Model file not found: {path}")
    
    logger.debug("Loading color encoder...")
    color_le = joblib.load(ENCODER_P)
    
    logger.debug("Loading model metrics...")
    model_metrics = joblib.load(METRICS_P)
    logger.debug("Models, encoder, and metrics loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

app = Flask(__name__, 
            template_folder=os.path.join(BASE, 'templates'),
            static_folder=os.path.join(BASE, 'static'))

@app.route('/')
def index():
    return render_template('index.html',
                         features=FEATURES,
                         categorical_options=CATEGORICAL_OPTIONS,
                         model_metrics=model_metrics)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logger.debug(f"Received prediction request with data: {data}")
        
        # Create input DataFrame
        input_data = {}
        for feature in FEATURES:
            if feature in CATEGORICAL_OPTIONS:
                input_data[feature] = [data.get(feature, CATEGORICAL_OPTIONS[feature][0])]
            else:
                input_data[feature] = [float(data.get(feature, 0))]
        
        df_in = pd.DataFrame(input_data)
        
        # Get predictions from all models
        predictions = {}
        for name, pipe in pipelines.items():
            try:
                code = pipe.predict(df_in)[0]
                color = color_le.inverse_transform([code])[0]
                predictions[name] = color
            except Exception as e:
                logger.error(f"Error predicting with {name}: {str(e)}")
                predictions[name] = "Error"
        
        return jsonify(predictions)
    except Exception as e:
        logger.error(f"Error processing prediction request: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 