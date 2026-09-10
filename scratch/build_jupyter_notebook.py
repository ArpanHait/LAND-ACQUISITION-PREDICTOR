"""Script to generate, execute, and save Experimental_Data_of_Model.ipynb and Experimental_Data_of_Model.py

Provides rigorous empirical comparison of:
1. Logistic Regression & Ridge Regressor (Linear Baseline)
2. Random Forest Classifier & Regressor (Bagging Ensemble)
3. Multi-Layer Perceptron / Neural Network (Deep Learning Baseline)
4. XGBoost Dual-Engine Classifier & Regressor (BhoomiAI Core)
"""

import json
import time
import base64
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler

BASE_DIR = Path(r"w:\Project Work\SIH")
DATASET_PATH = BASE_DIR / "models" / "prototype_land_acquisition_cases.csv"
OUTPUT_IPYNB = BASE_DIR / "models" / "Experimental_Data_of_Model.ipynb"
OUTPUT_PY = BASE_DIR / "models" / "Experimental_Data_of_Model.py"
OUTPUT_CHART = BASE_DIR / "models" / "experimental_model_comparison_chart.png"

# 1. Load Dataset
df = pd.read_csv(DATASET_PATH)

# Feature Engineering
def engineer_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = df_in.copy()
    df_out["area_per_owner"] = df_out["land_area_hectares"] / (df_out["affected_owner_count"] + 1)
    df_out["cost_per_hectare"] = df_out["compensation_estimate_inr_lakh"] / (df_out["land_area_hectares"] + 0.01)
    df_out["dispute_score"] = df_out["has_title_dispute"] + df_out["has_court_case"] + df_out["has_objection"]
    df_out["severe_legal_risk"] = ((df_out["has_court_case"] == 1) & (df_out["has_title_dispute"] == 1)).astype(int)
    df_out["milestone_avg"] = (
        df_out["land_notified_percent"] + df_out["award_completed_percent"] +
        df_out["compensation_disbursed_percent"] + df_out["possession_completed_percent"]
    ) / 4.0
    df_out["possession_lag"] = df_out["award_completed_percent"] - df_out["possession_completed_percent"]
    df_out["disbursal_ratio"] = df_out["compensation_disbursed_percent"] / (df_out["award_completed_percent"] + 1e-3)
    df_out["unfunded_amount"] = (1 - df_out["compensation_funding_available"]) * df_out["compensation_estimate_inr_lakh"]
    return df_out

df_eng = engineer_features(df)

CATEGORICAL_FEATURES = ["state", "district_type", "project_type", "land_type", "notification_stage"]
BASE_NUMERIC_FEATURES = [
    "land_area_hectares", "affected_owner_count", "is_multi_village",
    "has_title_dispute", "has_court_case", "has_objection",
    "compensation_estimate_inr_lakh", "compensation_funding_available",
    "days_since_case_opened", "land_notified_percent", "award_completed_percent",
    "compensation_disbursed_percent", "possession_completed_percent",
]
ENG_NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + [
    "area_per_owner", "cost_per_hectare", "dispute_score",
    "severe_legal_risk", "milestone_avg", "possession_lag",
    "disbursal_ratio", "unfunded_amount",
]
ALL_FEATURES = CATEGORICAL_FEATURES + ENG_NUMERIC_FEATURES

RISK_MAP = {"Low": 0, "Medium": 1, "High": 2}
y_class = df_eng["delay_risk"].map(RISK_MAP).values
y_reg = df_eng["delay_probability"].values
X_raw = df_eng[ALL_FEATURES]

# Split 80/20
X_train_raw, X_test_raw, y_train_cls, y_test_cls, y_train_reg, y_test_reg = train_test_split(
    X_raw, y_class, y_reg, test_size=0.20, random_state=42, stratify=y_class
)

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ("num", RobustScaler(), ENG_NUMERIC_FEATURES)
])

X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

# Models to evaluate
models = {
    "Logistic Regression": {
        "cls": LogisticRegression(max_iter=1000, random_state=42),
        "reg": Ridge(alpha=1.0, random_state=42)
    },
    "Random Forest": {
        "cls": RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
        "reg": RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    },
    "Neural Network (MLP)": {
        "cls": MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42, early_stopping=True),
        "reg": MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42, early_stopping=True)
    },
    "XGBoost (BhoomiAI)": {
        "cls": xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42, eval_metric="mlogloss"),
        "reg": xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42)
    }
}

results = []
cm_dict = {}

for name, model_pair in models.items():
    cls_model = model_pair["cls"]
    reg_model = model_pair["reg"]
    
    # Measure Training Time
    t0 = time.perf_counter()
    cls_model.fit(X_train, y_train_cls)
    reg_model.fit(X_train, y_train_reg)
    train_time_sec = time.perf_counter() - t0
    
    # Measure Latency (100 single samples)
    t_lat_start = time.perf_counter()
    for i in range(100):
        _ = cls_model.predict(X_test[i:i+1])
        _ = reg_model.predict(X_test[i:i+1])
    latency_ms = ((time.perf_counter() - t_lat_start) / 100.0) * 1000.0
    
    # Classification Predictions
    y_pred_cls = cls_model.predict(X_test)
    acc = accuracy_score(y_test_cls, y_pred_cls) * 100.0
    prec_high = precision_score(y_test_cls, y_pred_cls, labels=[2], average="macro", zero_division=0) * 100.0
    rec_high = recall_score(y_test_cls, y_pred_cls, labels=[2], average="macro", zero_division=0) * 100.0
    f1_macro = f1_score(y_test_cls, y_pred_cls, average="macro") * 100.0
    
    cm = confusion_matrix(y_test_cls, y_pred_cls)
    cm_dict[name] = cm
    
    # Regression Predictions
    y_pred_reg = np.clip(reg_model.predict(X_test), 0.0, 1.0)
    r2 = r2_score(y_test_reg, y_pred_reg)
    mae = mean_absolute_error(y_test_reg, y_pred_reg) * 100.0
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)) * 100.0
    
    results.append({
        "Model Architecture": name,
        "Accuracy (%)": round(acc, 2),
        "High-Risk Precision (%)": round(prec_high, 2),
        "High-Risk Recall (%)": round(rec_high, 2),
        "Macro F1 (%)": round(f1_macro, 2),
        "R2 Score": round(r2, 4),
        "MAE (%)": round(mae, 2),
        "RMSE (%)": round(rmse, 2),
        "Train Time (s)": round(train_time_sec, 3),
        "Latency (ms/case)": round(latency_ms, 2)
    })

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))

# Generate high-contrast presentation chart
plt.style.use('dark_background')
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.patch.set_facecolor('#0B0F19')

for row in axes:
    for ax in row:
        ax.set_facecolor('#111827')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#374151')
        ax.spines['bottom'].set_color('#374151')

model_names = res_df["Model Architecture"].tolist()
colors = ['#64748B', '#38BDF8', '#A855F7', '#10B981']

# Subplot 1: Classification Accuracy & High-Risk Precision
ax1 = axes[0, 0]
x = np.arange(len(model_names))
w = 0.35
rects1 = ax1.bar(x - w/2, res_df["Accuracy (%)"], w, label='Overall Accuracy (%)', color='#38BDF8', alpha=0.9)
rects2 = ax1.bar(x + w/2, res_df["High-Risk Precision (%)"], w, label='High-Risk Precision (%)', color='#10B981', alpha=0.9)
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, fontsize=9.5, fontweight='600', color='#E5E7EB')
ax1.set_ylim(70, 102)
ax1.set_title('Risk Tier Classification & High-Risk Precision', fontsize=12, fontweight='bold', color='#FFFFFF', pad=10)
ax1.legend(loc='lower right', framealpha=0.3)
ax1.grid(axis='y', linestyle='--', alpha=0.2)
for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f'{h:.1f}%', (rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8.5, color='#E2E8F0')
for rect in rects2:
    h = rect.get_height()
    ax1.annotate(f'{h:.1f}%', (rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8.5, fontweight='bold', color='#4ADE80')

# Subplot 2: R2 Score & Mean Absolute Error (MAE)
ax2 = axes[0, 1]
rects3 = ax2.bar(x - w/2, res_df["R2 Score"] * 100, w, label='Delay R² Score (x100)', color='#818CF8', alpha=0.9)
rects4 = ax2.bar(x + w/2, res_df["MAE (%)"], w, label='Error MAE (%) [Lower is Better]', color='#F43F5E', alpha=0.9)
ax2.set_xticks(x)
ax2.set_xticklabels(model_names, fontsize=9.5, fontweight='600', color='#E5E7EB')
ax2.set_title('Continuous Probability Calibration (R² vs Error MAE)', fontsize=12, fontweight='bold', color='#FFFFFF', pad=10)
ax2.legend(loc='upper right', framealpha=0.3)
ax2.grid(axis='y', linestyle='--', alpha=0.2)
for rect in rects3:
    h = rect.get_height()
    ax2.annotate(f'{h/100:.3f}', (rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8.5, color='#E2E8F0')
for rect in rects4:
    h = rect.get_height()
    ax2.annotate(f'{h:.2f}%', (rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8.5, fontweight='bold', color='#FB7185')

# Subplot 3: Training Time vs Inference Latency
ax3 = axes[1, 0]
rects5 = ax3.bar(x, res_df["Latency (ms/case)"], 0.45, color=colors, alpha=0.85, edgecolor='#4B5563')
ax3.set_xticks(x)
ax3.set_xticklabels(model_names, fontsize=9.5, fontweight='600', color='#E5E7EB')
ax3.set_title('Inference Latency per Case (Milliseconds)', fontsize=12, fontweight='bold', color='#FFFFFF', pad=10)
ax3.set_ylabel('ms / case', color='#9CA3AF')
ax3.grid(axis='y', linestyle='--', alpha=0.2)
for rect in rects5:
    h = rect.get_height()
    ax3.annotate(f'{h:.2f} ms', (rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='600', color='#FFFFFF')

# Subplot 4: XGBoost Feature Importance
ax4 = axes[1, 1]
xgb_cls = models["XGBoost (BhoomiAI)"]["cls"]
feature_names = preprocessor.get_feature_names_out()
importances = xgb_cls.feature_importances_
top_idx = np.argsort(importances)[-10:]
top_feats = [feature_names[i].replace("num__", "").replace("cat__", "") for i in top_idx]
top_imps = importances[top_idx]

ax4.barh(range(len(top_idx)), top_imps, color='#10B981', alpha=0.9, edgecolor='#34D399')
ax4.set_yticks(range(len(top_idx)))
ax4.set_yticklabels(top_feats, fontsize=9, color='#E5E7EB')
ax4.set_title('Top 10 Statutory Drivers (XGBoost Feature Importance)', fontsize=12, fontweight='bold', color='#FFFFFF', pad=10)
ax4.set_xlabel('Relative Importance Score', color='#9CA3AF')
ax4.grid(axis='x', linestyle='--', alpha=0.2)

plt.tight_layout(pad=3.0)
fig.savefig(OUTPUT_CHART, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close(fig)
print(f"Chart saved to {OUTPUT_CHART}")

print("Benchmarking completed successfully.")
