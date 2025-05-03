# Color Prediction App

This application predicts the color outcome of chemical reactions using multiple machine learning models.

## Project Structure

```
ColorPredictionApp/
├── data/           # Contains the dataset (github.csv)
├── templates/      # HTML templates
├── Feature_correlations/      # The ipynb file for model training and saving
├── static/         # Static files (CSS, JS)
└── src/            # Source code
    ├── app.py      # Flask application
    
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows:
```bash
venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place your dataset:
- Copy `github.csv` to the `data/` directory

5. Train the models:
```bash
Feature_correlations.ipynb

\\Save the models in pkl format
```

## Running the Application

1. Start the Flask server:
```bash
cd src
set FLASK_APP=app.py  # Windows
export FLASK_APP=app.py  # Linux/Mac
flask run --host=0.0.0.0
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

## Features

- Multiple model predictions (Logistic Regression, Decision Tree, Random Forest, SVM, ANN, XGBoost)
- User-friendly web interface
- Real-time predictions
- Error handling and validation

## Notes

- Make sure all model files (*_best.pkl) and color_encoder.pkl are present in the models/ directory
- The application expects the dataset to be in the data/ directory
- All paths are relative to the project root 
