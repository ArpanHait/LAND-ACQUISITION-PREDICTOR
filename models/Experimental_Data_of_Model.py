"""Experimental Data of Model: Benchmark Evaluation Script (BhoomiAI - SIH)
Compares Logistic Regression, Random Forest, Neural Network (MLP), and XGBoost.
"""

import time
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "prototype_land_acquisition_cases.csv"

print("=" * 90)
print("     BHOOMI-AI: EMPIRICAL BENCHMARK — XGBOOST vs RANDOM FOREST vs MLP vs LOGISTIC REGRESSION     ")
print("=" * 90)

df = pd.read_csv(DATASET_PATH)

def engineer_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = df_in.copy()
    df_out['area_per_owner'] = df_out['land_area_hectares'] / (df_out['affected_owner_count'] + 1)
    df_out['cost_per_hectare'] = df_out['compensation_estimate_inr_lakh'] / (df_out['land_area_hectares'] + 0.01)
    df_out['dispute_score'] = df_out['has_title_dispute'] + df_out['has_court_case'] + df_out['has_objection']
    df_out['severe_legal_risk'] = ((df_out['has_court_case'] == 1) & (df_out['has_title_dispute'] == 1)).astype(int)
    df_out['milestone_avg'] = (
        df_out['land_notified_percent'] + df_out['award_completed_percent'] +
        df_out['compensation_disbursed_percent'] + df_out['possession_completed_percent']
    ) / 4.0
    df_out['possession_lag'] = df_out['award_completed_percent'] - df_out['possession_completed_percent']
    df_out['disbursal_ratio'] = df_out['compensation_disbursed_percent'] / (df_out['award_completed_percent'] + 1e-3)
    df_out['unfunded_amount'] = (1 - df_out['compensation_funding_available']) * df_out['compensation_estimate_inr_lakh']
    return df_out

df_eng = engineer_features(df)

CATEGORICAL_FEATURES = ['state', 'district_type', 'project_type', 'land_type', 'notification_stage']
BASE_NUMERIC_FEATURES = [
    'land_area_hectares', 'affected_owner_count', 'is_multi_village',
    'has_title_dispute', 'has_court_case', 'has_objection',
    'compensation_estimate_inr_lakh', 'compensation_funding_available',
    'days_since_case_opened', 'land_notified_percent', 'award_completed_percent',
    'compensation_disbursed_percent', 'possession_completed_percent',
]
ENG_NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + [
    'area_per_owner', 'cost_per_hectare', 'dispute_score',
    'severe_legal_risk', 'milestone_avg', 'possession_lag',
    'disbursal_ratio', 'unfunded_amount',
]
ALL_FEATURES = CATEGORICAL_FEATURES + ENG_NUMERIC_FEATURES

RISK_MAP = {'Low': 0, 'Medium': 1, 'High': 2}
y_class = df_eng['delay_risk'].map(RISK_MAP).values
y_reg = df_eng['delay_probability'].values
X_raw = df_eng[ALL_FEATURES]

X_train_raw, X_test_raw, y_train_cls, y_test_cls, y_train_reg, y_test_reg = train_test_split(
    X_raw, y_class, y_reg, test_size=0.20, random_state=42, stratify=y_class
)

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES),
    ('num', RobustScaler(), ENG_NUMERIC_FEATURES)
])

X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

models = {
    'Logistic Regression': {
        'cls': LogisticRegression(max_iter=1000, random_state=42),
        'reg': Ridge(alpha=1.0, random_state=42)
    },
    'Random Forest': {
        'cls': RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
        'reg': RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    },
    'Neural Network (MLP)': {
        'cls': MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42, early_stopping=True),
        'reg': MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42, early_stopping=True)
    },
    'XGBoost (BhoomiAI)': {
        'cls': xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42, eval_metric='mlogloss'),
        'reg': xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42)
    }
}

results = []
for name, model_pair in models.items():
    cls_model = model_pair['cls']
    reg_model = model_pair['reg']
    
    t0 = time.perf_counter()
    cls_model.fit(X_train, y_train_cls)
    reg_model.fit(X_train, y_train_reg)
    train_time_sec = time.perf_counter() - t0
    
    t_lat = time.perf_counter()
    for i in range(100):
        _ = cls_model.predict(X_test[i:i+1])
        _ = reg_model.predict(X_test[i:i+1])
    latency_ms = ((time.perf_counter() - t_lat) / 100.0) * 1000.0
    
    y_pred_cls = cls_model.predict(X_test)
    acc = accuracy_score(y_test_cls, y_pred_cls) * 100.0
    prec_high = precision_score(y_test_cls, y_pred_cls, labels=[2], average='macro', zero_division=0) * 100.0
    rec_high = recall_score(y_test_cls, y_pred_cls, labels=[2], average='macro', zero_division=0) * 100.0
    f1_macro = f1_score(y_test_cls, y_pred_cls, average='macro') * 100.0
    
    y_pred_reg = np.clip(reg_model.predict(X_test), 0.0, 1.0)
    r2 = r2_score(y_test_reg, y_pred_reg)
    mae = mean_absolute_error(y_test_reg, y_pred_reg) * 100.0
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)) * 100.0
    
    results.append({
        'Model Architecture': name,
        'Accuracy (%)': round(acc, 2),
        'High-Risk Precision (%)': round(prec_high, 2),
        'High-Risk Recall (%)': round(rec_high, 2),
        'Macro F1 (%)': round(f1_macro, 2),
        'R2 Score': round(r2, 4),
        'MAE (%)': round(mae, 2),
        'RMSE (%)': round(rmse, 2),
        'Train Time (s)': round(train_time_sec, 3),
        'Latency (ms)': round(latency_ms, 2)
    })

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
print("=" * 90)
