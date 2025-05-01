import pandas as pd
import joblib
import numpy as np
import os
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Get the base directory
BASE = os.path.dirname(os.path.dirname(__file__))
CSV_PATH = os.path.join(BASE, "data", "github.csv")

# 1. Load & clean data
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

# Merge red/orange/yellow → 'yor', drop 'multicolor'
df['Color'] = (
    df['Color']
      .str.strip().str.lower()
      .replace({'red':'yor','orange':'yor','yellow':'yor'})
)
df = df[df['Color'] != 'multicolor']

# 2. Define features & target
FEATURES = [
    'citric acid','urea','ethylenediamine','HCl','ammonium','NaOH',
    'boric acid','sodium thiosulfate','KOH','formic acid',
    'Reaction temperatrue( C )','reaction time (min)',
    'Reaction method','Solvent','pH','purification method'
]
TARGET = 'Color'

X = df[FEATURES]
y_raw = df[TARGET]

# 3. Encode target labels
color_le = LabelEncoder()
y = color_le.fit_transform(y_raw)
joblib.dump(color_le, os.path.join(BASE, 'models', 'color_encoder.pkl'))  # for inference

# 4. Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Build preprocessing transformer
numeric_features = [
    f for f in FEATURES
    if f not in ['Reaction method','Solvent','pH','purification method']
]
categorical_features = [
    'Reaction method','Solvent','pH','purification method'
]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# 6. Define your models and (small) hyperparameter grids
models = {
    'LogReg':       LogisticRegression(max_iter=1000, random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'SVM':          SVC(probability=True, random_state=42),
    'ANN':          MLPClassifier(hidden_layer_sizes=(100,50), max_iter=1000, random_state=42),
    'XGBoost':      XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42)
}

param_grids = {
    'LogReg':       {'clf__C': [0.1, 1, 10]},
    'RandomForest': {'clf__n_estimators': [50, 100]},
    'XGBoost':      {'clf__n_estimators': [50, 100]},
    'ANN':          {'clf__hidden_layer_sizes': [(50,), (100,50)]}
    # DecisionTree & SVM use defaults
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store per-color metrics for each model
model_metrics = {}

# 7. Train + GridSearchCV + save best pipelines
for name, clf in models.items():
    print(f"\n>> Training & tuning {name} pipeline...")
    pipe = ImbPipeline([
        ('pre',   preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf',   clf)
    ])
    grid = GridSearchCV(
        pipe,
        param_grids.get(name, {}),
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    best_pipe = grid.best_estimator_
    
    # Get predictions on test set
    y_pred = best_pipe.predict(X_test)
    y_pred_proba = best_pipe.predict_proba(X_test)
    
    # Calculate per-color metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Calculate AUC for each class
    n_classes = len(np.unique(y))
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    auc_scores = {}
    
    for i in range(n_classes):
        color = color_le.inverse_transform([i])[0]
        try:
            auc = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])
            auc_scores[color] = auc
        except:
            auc_scores[color] = 0.0
    
    # Find the color with highest AUC
    best_color = max(auc_scores, key=auc_scores.get)
    best_auc = auc_scores[best_color]
    
    # Store metrics
    model_metrics[name] = {
        'accuracy': f"{report['accuracy']*100:.2f}%",
        'best_metric': f"AUC: {best_auc:.4f}",
        'best_color': best_color,
        'per_color_metrics': report,
        'auc_scores': auc_scores
    }
    
    print(f" * Best CV F1_macro for {name}: {grid.best_score_:.4f}")
    print(f" * Best performing color by AUC: {best_color} (AUC: {best_auc:.4f})")
    print(f" * AUC scores for all colors: {auc_scores}")
    
    out_path = os.path.join(BASE, 'models', f"{name}_best.pkl")
    joblib.dump(best_pipe, out_path)
    print(f" * Saved pipeline to {out_path}")

# Save model metrics
joblib.dump(model_metrics, os.path.join(BASE, 'models', 'model_metrics.pkl'))

print("\n✅ Training complete. Pipelines, color_encoder.pkl, and model_metrics.pkl are ready for your app.") 