# cvd_prediction.py
"""
Integrated CVD prediction pipeline (Drop-in replacement)
- Clinical relabeling (train-fit only)
- Feature engineering transformer
- Preprocessing (imputer/scaler/one-hot)
- SelectKBest inside pipeline
- SMOTE inside pipeline (per-fold)
- Base learners + Voting (soft) + Stacking ensembles
- Bayesian hyperparameter optimization for LightGBM (Optuna)
- Bootstrapped test CIs, ROC, calibration, confusion matrices, feature importance
- Compatible with varied LightGBM versions (uses callback API)
"""

import warnings
warnings.filterwarnings("ignore")

import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc, roc_auc_score,
    log_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.base import BaseEstimator, TransformerMixin

# Boosting libraries
from lightgbm import LGBMClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Imbalanced-learn pipeline & samplers
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Optuna for Bayesian optimization
import optuna

sns.set(style="whitegrid")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUTPUT_DIR = './cvd_results_' + datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------- Utilities ----------------------
def safe_divide(numerator: pd.Series, denominator: pd.Series, floor: float = 1.0) -> pd.Series:
    denom = denominator.copy().astype(float)
    denom = denom.fillna(floor)
    denom = np.where(np.abs(denom) < floor, np.sign(denom) * floor, denom)
    return numerator / denom


def bootstrap_metric(model, X, y, metric_func, n_iterations=1000, confidence=0.95):
    scores = []
    n = len(y)
    for _ in range(n_iterations):
        idx = np.random.choice(n, n, replace=True)
        Xb = X[idx] if isinstance(X, np.ndarray) else X.iloc[idx]
        yb = y[idx]
        ypred = model.predict(Xb)
        scores.append(metric_func(yb, ypred))
    alpha = 1 - confidence
    lower = np.percentile(scores, 100 * (alpha / 2.0))
    upper = np.percentile(scores, 100 * (1 - alpha / 2.0))
    return float(np.mean(scores)), float(lower), float(upper)


# ---------------- Clinical relabeler ----------------
class ClinicalRiskScorer:
    """
    Clinically-guided relabeler that reconstructs LOW/INTERMEDIATE/HIGH
    from available clinical fields. Fit on training set only.
    """
    def __init__(self):
        self.low_thr = None
        self.high_thr = None

    def compute_raw_score(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        score = np.zeros(n, dtype=float)

        # Age
        if 'Age' in df.columns:
            a = df['Age'].fillna(df['Age'].median()).clip(18, 100)
            score += np.where(a < 65,
                              ((a - 40) / 15).clip(0, 4),
                              ((a - 40) / 12).clip(0, 5))

        # Blood pressure (MAP-esque)
        if all(c in df.columns for c in ['Systolic BP', 'Diastolic BP']):
            sbp = df['Systolic BP'].fillna(df['Systolic BP'].median())
            dbp = df['Diastolic BP'].fillna(df['Diastolic BP'].median())
            map_ = dbp + (sbp - dbp) / 3.0
            score += ((map_ - 85) / 10).clip(0, 4)

        # LDL
        if 'Estimated LDL (mg/dL)' in df.columns:
            ldl = df['Estimated LDL (mg/dL)'].fillna(df['Estimated LDL (mg/dL)'].median())
            score += np.where(ldl >= 190, 4.0, ((ldl - 100) / 30).clip(0, 3.5))

        # HDL protective factor
        if 'HDL (mg/dL)' in df.columns:
            hdl = df['HDL (mg/dL)'].fillna(df['HDL (mg/dL)'].median())
            score += np.where(hdl < 40, 2.5, np.where(hdl < 50, 1.0, 0.0))
            score -= np.where(hdl >= 60, 1.5, 0.0)

        # Triglycerides
        if 'Triglycerides (mg/dL)' in df.columns:
            tg = df['Triglycerides (mg/dL)'].fillna(df['Triglycerides (mg/dL)'].median())
            score += np.where(tg >= 500, 3.5, np.where(tg >= 200, 2.0, np.where(tg >= 150, 1.0, 0.0)))

        # Fasting blood sugar
        if 'Fasting Blood Sugar (mg/dL)' in df.columns:
            fbs = df['Fasting Blood Sugar (mg/dL)'].fillna(df['Fasting Blood Sugar (mg/dL)'].median())
            score += np.where(fbs >= 126, 3.5, np.where(fbs >= 100, 1.5, 0.0))

        # BMI
        if 'BMI' in df.columns:
            bmi = df['BMI'].fillna(df['BMI'].median())
            score += np.where(bmi < 18.5, ((18.5 - bmi) / 2.0).clip(0, 2.0), 0.0)
            score += np.where(bmi >= 40, 3.5,
                              np.where(bmi >= 35, 3.0,
                                       np.where(bmi >= 30, 2.0,
                                                np.where(bmi >= 25, 1.0, 0.0))))

        # Smoking
        if 'Smoking Status' in df.columns:
            score += np.where(df['Smoking Status'].fillna('N') == 'Y', 3.0, 0.0)

        # Family history
        if 'Family History of CVD' in df.columns:
            score += np.where(df['Family History of CVD'].fillna('N') == 'Y', 2.0, 0.0)

        # Physical activity
        if 'Physical Activity Level' in df.columns:
            pal = df['Physical Activity Level'].fillna('Moderate')
            score -= np.where(pal == 'High', 1.5, 0.0)
            score += np.where(pal == 'Low', 1.2, 0.0)

        return score

    def fit(self, df: pd.DataFrame):
        raw = self.compute_raw_score(df)
        self.low_thr = float(np.percentile(raw, 33))
        self.high_thr = float(np.percentile(raw, 67))
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        raw = self.compute_raw_score(df)
        labels = np.where(raw <= self.low_thr, 'LOW',
                          np.where(raw <= self.high_thr, 'INTERMEDIATE', 'HIGH'))
        return df.copy(), labels


# ---------------- Feature engineering transformer ----------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.hdl_floor = 20.0

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if 'HDL (mg/dL)' in df.columns and 'Total Cholesterol (mg/dL)' in df.columns:
            df['TC_HDL_Ratio'] = safe_divide(df['Total Cholesterol (mg/dL)'],
                                             df['HDL (mg/dL)'], floor=self.hdl_floor).clip(1, 10)
        if 'Estimated LDL (mg/dL)' in df.columns and 'HDL (mg/dL)' in df.columns:
            df['LDL_HDL_Ratio'] = safe_divide(df['Estimated LDL (mg/dL)'],
                                              df['HDL (mg/dL)'], floor=self.hdl_floor).clip(0.5, 8)
        if 'Triglycerides (mg/dL)' in df.columns and 'HDL (mg/dL)' in df.columns:
            df['TG_HDL_Ratio'] = safe_divide(df['Triglycerides (mg/dL)'],
                                             df['HDL (mg/dL)'], floor=self.hdl_floor).clip(0.5, 10)
        if 'Total Cholesterol (mg/dL)' in df.columns and 'HDL (mg/dL)' in df.columns:
            df['Non_HDL_Chol'] = (df['Total Cholesterol (mg/dL)'] - df['HDL (mg/dL)']).clip(0, 300)
        if 'Systolic BP' in df.columns and 'Diastolic BP' in df.columns:
            df['Pulse_Pressure'] = (df['Systolic BP'] - df['Diastolic BP']).clip(20, 100)
            df['MAP'] = df['Diastolic BP'] + df['Pulse_Pressure'] / 3.0
            df['Wide_Pulse_Pressure'] = (df['Pulse_Pressure'] > 60).astype(float)
        if 'Age' in df.columns and 'BMI' in df.columns:
            age_norm = (df['Age'] / 100.0).clip(0, 1)
            bmi_norm = (df['BMI'] / 50.0).clip(0, 1)
            df['Age_BMI_Interaction'] = age_norm * bmi_norm
        if 'Age' in df.columns and 'Systolic BP' in df.columns:
            age_norm = (df['Age'] / 100.0).clip(0, 1)
            sbp_norm = (df['Systolic BP'] / 200.0).clip(0, 1)
            df['Age_SBP_Interaction'] = age_norm * sbp_norm
        if 'BMI' in df.columns:
            df['BMI_Obese'] = (df['BMI'] >= 30).astype(float)
            df['BMI_Overweight'] = ((df['BMI'] >= 25) & (df['BMI'] < 30)).astype(float)
            df['BMI_Underweight'] = (df['BMI'] < 18.5).astype(float)
        if 'Age' in df.columns:
            df['Age_65_plus'] = (df['Age'] >= 65).astype(float)
            df['Age_55_64'] = ((df['Age'] >= 55) & (df['Age'] < 65)).astype(float)
        if 'Estimated LDL (mg/dL)' in df.columns:
            df['LDL_Very_High'] = (df['Estimated LDL (mg/dL)'] >= 190).astype(float)
            df['LDL_High'] = ((df['Estimated LDL (mg/dL)'] >= 160) &
                              (df['Estimated LDL (mg/dL)'] < 190)).astype(float)
        if 'HDL (mg/dL)' in df.columns:
            df['HDL_Low'] = (df['HDL (mg/dL)'] < 40).astype(float)
            df['HDL_Optimal'] = (df['HDL (mg/dL)'] >= 60).astype(float)
        if 'Systolic BP' in df.columns and 'Diastolic BP' in df.columns:
            df['BP_Stage2_HTN'] = ((df['Systolic BP'] >= 140) |
                                   (df['Diastolic BP'] >= 90)).astype(float)
            df['BP_Stage1_HTN'] = (((df['Systolic BP'] >= 130) & (df['Systolic BP'] < 140)) |
                                   ((df['Diastolic BP'] >= 80) & (df['Diastolic BP'] < 90))).astype(float)
        if 'Fasting Blood Sugar (mg/dL)' in df.columns:
            df['Diabetes'] = (df['Fasting Blood Sugar (mg/dL)'] >= 126).astype(float)
            df['Prediabetes'] = ((df['Fasting Blood Sugar (mg/dL)'] >= 100) &
                                 (df['Fasting Blood Sugar (mg/dL)'] < 126)).astype(float)
        return df


# ---------------- Plotting helpers ----------------
def plot_roc_curves(models_dict: Dict, X_test, y_test, class_names: List[str], save_path: str):
    """
    Plot ROC curves for each class in vertical layout.
    One subplot per class, stacked vertically.
    """
    n_classes = len(class_names)
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
    
    # Create vertical subplots: n_classes rows, 1 column
    fig, axes = plt.subplots(n_classes, 1, figsize=(8, 5 * n_classes))
    
    # Generate colors for different models
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(models_dict))))
    
    # Ensure axes is always a list, even with single class
    if n_classes == 1:
        axes = [axes]
    
    # Plot ROC curve for each class
    for ci in range(n_classes):
        ax = axes[ci]
        
        # Plot each model's ROC curve for this class
        for (mname, m), color in zip(models_dict.items(), colors):
            try:
                # Get probability scores
                if hasattr(m, 'predict_proba'):
                    y_score = m.predict_proba(X_test)[:, ci]
                elif hasattr(m, 'decision_function'):
                    dfun = m.decision_function(X_test)
                    y_score = dfun[:, ci] if dfun.ndim > 1 else dfun
                else:
                    continue
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test_bin[:, ci], y_score)
                roc_auc = auc(fpr, tpr)
                
                # Plot
                ax.plot(fpr, tpr, color=color, lw=2, 
                       label=f'{mname} (AUC={roc_auc:.3f})')
            except Exception as e:
                print(f"Warning: Could not plot ROC for {mname}, class {ci}: {e}")
                continue
        
        # Plot diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.500)')
        
        # Labels and formatting
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(f'ROC Curve - {class_names[ci]}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curves (vertical layout) to: {save_path}")


def plot_calibration_curves(models_dict: Dict, X_test, y_test, class_names: List[str], save_path: str):
    from sklearn.calibration import calibration_curve
    n_classes = len(class_names)
    top_models = list(models_dict.items())[:min(4, len(models_dict))]
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5))
    if n_classes == 1:
        axes = [axes]
    for ci in range(n_classes):
        ax = axes[ci]
        for mname, m in top_models:
            try:
                if hasattr(m, 'predict_proba'):
                    probs = m.predict_proba(X_test)[:, ci]
                    y_true_bin = (y_test == ci).astype(int)
                    frac_pos, mean_pred = calibration_curve(y_true_bin, probs, n_bins=10, strategy='uniform')
                    ax.plot(mean_pred, frac_pos, marker='o', label=mname)
            except Exception:
                continue
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('Predicted prob')
        ax.set_ylabel('True prob')
        ax.set_title(f'Calibration - {class_names[ci]}')
        ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(models_dict: Dict, X_test, y_test, class_names: List[str], save_path: str):
    top_models = list(models_dict.items())[:4]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    for ax, (mname, m) in zip(axes, top_models):
        ypred = m.predict(X_test)
        cm = confusion_matrix(y_test, ypred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_title(mname)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(model, feature_names: List[str], save_path: str, top_n=20):
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'named_steps') and 'classifier' in model.named_steps and hasattr(model.named_steps['classifier'], 'feature_importances_'):
            importances = model.named_steps['classifier'].feature_importances_
        else:
            print("Model does not expose feature_importances_")
            return
        indices = np.argsort(importances)[::-1][:top_n]
        top_feats = [feature_names[i] for i in indices]
        top_imp = importances[indices]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(top_feats))[::-1], top_imp[::-1], color='#3498db')
        ax.set_yticks(range(len(top_feats)))
        ax.set_yticklabels(top_feats[::-1])
        ax.set_xlabel('Importance')
        ax.set_title('Top feature importances')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Could not plot feature importance: {e}")


# ---------------- Bayesian optimization (Optuna) for LightGBM ----------------
def bayesian_optimize_lgbm(X_train, y_train, n_trials=30, random_state=RANDOM_STATE):
    """
    Optuna-based Bayesian optimization for LightGBM.
    Uses StratifiedKFold on training set only. Uses early stopping via callback.
    Does NOT pass explicit 'objective' or 'num_class' to avoid version compatibility issues.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    def objective(trial):
        param = {
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 16, 128),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
            'n_estimators': 500,
            'random_state': random_state,
            'verbosity': -1
        }

        losses = []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            model = LGBMClassifier(**param)

            # Use callback-based early stopping for compatibility
            try:
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=0)]
                )
            except TypeError:
                # fallback if callback not supported in very old versions
                model.fit(X_tr, y_tr)

            proba = model.predict_proba(X_val)
            losses.append(log_loss(y_val, proba))

        return float(np.mean(losses))

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    final_params = {
        'learning_rate': best.get('learning_rate', 0.05),
        'num_leaves': best.get('num_leaves', 31),
        'feature_fraction': best.get('feature_fraction', 1.0),
        'bagging_fraction': best.get('bagging_fraction', 1.0),
        'bagging_freq': best.get('bagging_freq', 0),
        'min_child_samples': best.get('min_child_samples', 20),
        'reg_alpha': best.get('reg_alpha', 0.0),
        'reg_lambda': best.get('reg_lambda', 0.0),
        'n_estimators': 500,
        'random_state': random_state,
        'verbosity': -1
    }
    return final_params, study


# ---------------- Main pipeline ----------------
class CVDModelPipeline:
    def __init__(self, data_path: str, k_features: int = 150):
        self.data_path = data_path
        self.k_features = k_features
        self.df = None
        self.target_col = None
        self.feature_engineer = FeatureEngineer()
        self.risk_scorer = ClinicalRiskScorer()
        self.numeric_cols = []
        self.cat_cols = []
        self.feature_names = []
        self.models = {}
        self.results = []

    def load_and_inspect(self):
        df = pd.read_csv(self.data_path)
        df.columns = [c.strip() for c in df.columns]
        potential_targets = ['CVD_Outcome', 'CVD_Diagnosis', 'Cardiovascular_Disease', 'Heart_Disease', 'MI', 'Stroke', 'CVD_Event']
        self.target_col = next((t for t in potential_targets if t in df.columns), None)
        self.df = df
        return df

    def _choose_stratify_col(self, df: pd.DataFrame):
        if 'Age' in df.columns:
            return pd.cut(df['Age'].fillna(50), bins=[0, 40, 55, 65, 200], labels=[0, 1, 2, 3])
        for c in ['Smoking Status', 'Family History of CVD']:
            if c in df.columns:
                return df[c].fillna('Unknown')
        return None

    def initial_split(self, test_size=0.2):
        df = self.df.copy()
        strat = self._choose_stratify_col(df)
        if strat is None:
            tr, te = train_test_split(df, test_size=test_size, random_state=RANDOM_STATE)
        else:
            tr, te = train_test_split(df, test_size=test_size, random_state=RANDOM_STATE, stratify=strat)
        return tr.reset_index(drop=True), te.reset_index(drop=True)

    def build_preprocessor(self, X_train: pd.DataFrame):
        num_cols = [c for c in X_train.columns if X_train[c].dtype in ['int64', 'float64', 'float32', 'int32']]
        cat_cols = [c for c in X_train.columns if X_train[c].dtype == 'object']
        self.numeric_cols, self.cat_cols = num_cols, cat_cols

        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ], remainder='drop')
        return preprocessor

    def prepare_and_train(self, test_size=0.2, optimize_lgbm=True, lgbm_trials=30):
        print("=" * 80)
        print("CVD MODEL PIPELINE - Integrated (Ensembles + Bayesian Opt)")
        print("=" * 80)

        # Load and split
        self.load_and_inspect()
        train_df, test_df = self.initial_split(test_size=test_size)

        # Fit clinical relabeler on training set only
        self.risk_scorer.fit(train_df)
        train_df, y_train_labels = self.risk_scorer.transform(train_df)
        test_df, y_test_labels = self.risk_scorer.transform(test_df)
        label_map = {'LOW': 0, 'INTERMEDIATE': 1, 'HIGH': 2}
        y_train = np.array([label_map[l] for l in y_train_labels])
        y_test = np.array([label_map[l] for l in y_test_labels])

        print(f"Training distribution: {pd.Series(y_train_labels).value_counts().to_dict()}")
        print(f"Test distribution    : {pd.Series(y_test_labels).value_counts().to_dict()}")

        # Feature engineering
        X_train_eng = self.feature_engineer.fit_transform(train_df)
        X_test_eng = self.feature_engineer.transform(test_df)

        # Preprocessing
        preprocessor = self.build_preprocessor(X_train_eng)
        X_train_processed = preprocessor.fit_transform(X_train_eng)
        X_test_processed = preprocessor.transform(X_test_eng)

        # Feature names
        try:
            num_feats = self.numeric_cols
            cat_feats = []
            if self.cat_cols:
                cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
                cat_feats = cat_encoder.get_feature_names_out(self.cat_cols).tolist()
            self.feature_names = num_feats + cat_feats
        except Exception:
            self.feature_names = [f'feature_{i}' for i in range(X_train_processed.shape[1])]

        print(f"After preprocessing: {X_train_processed.shape[1]} features")

        # Bayesian optimization for LightGBM (training-set CV only)
        tuned_lgbm_params = None
        if optimize_lgbm:
            print("\n>> Running Bayesian optimization for LightGBM (training-set CV only, no test leakage)")
            tuned_lgbm_params, study = bayesian_optimize_lgbm(X_train_processed, y_train, n_trials=lgbm_trials)
            print("Best LightGBM params found (Optuna):", tuned_lgbm_params)
            try:
                study.trials_dataframe().to_csv(os.path.join(OUTPUT_DIR, 'optuna_lgbm_trials.csv'), index=False)
            except Exception:
                pass

        # SMOTE decision
        min_class_count = np.bincount(y_train).min()
        k_neighbors = max(1, min(5, min_class_count - 1)) if min_class_count > 1 else 1
        use_smote = (min_class_count > 1)
        if use_smote:
            print(f"SMOTE enabled (k_neighbors={k_neighbors})")
        else:
            print("SMOTE disabled - insufficient class counts")

        # Base models
        base_models = {
            'Logistic Regression': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE),
            'CatBoost': CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, l2_leaf_reg=3, verbose=False, random_state=RANDOM_STATE),
            'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, eval_metric='mlogloss', random_state=RANDOM_STATE),
        }

        # LightGBM (tuned if available)
        if tuned_lgbm_params:
            lgbm_model = LGBMClassifier(**tuned_lgbm_params)
        else:
            lgbm_model = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=RANDOM_STATE)

        models_to_train = {
            'Logistic Regression': base_models['Logistic Regression'],
            'CatBoost': base_models['CatBoost'],
            'XGBoost': base_models['XGBoost'],
            'LightGBM': lgbm_model
        }

        # Ensembles
        voting_ensemble = VotingClassifier(estimators=[
            ('cat', CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, random_state=RANDOM_STATE, verbose=False)),
            ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, eval_metric='mlogloss', random_state=RANDOM_STATE)),
            ('lgbm', LGBMClassifier(**(tuned_lgbm_params if tuned_lgbm_params else {'n_estimators': 200, 'learning_rate': 0.05, 'random_state': RANDOM_STATE})))
        ], voting='soft', n_jobs=-1)

        stacking_ensemble = StackingClassifier(estimators=[
            ('cat', CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, random_state=RANDOM_STATE, verbose=False)),
            ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, eval_metric='mlogloss', random_state=RANDOM_STATE)),
            ('lgbm', LGBMClassifier(**(tuned_lgbm_params if tuned_lgbm_params else {'n_estimators': 200, 'learning_rate': 0.05, 'random_state': RANDOM_STATE})))
        ], final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced'), n_jobs=-1, passthrough=False, stack_method='predict_proba')

        models_to_train['Voting Ensemble'] = voting_ensemble
        models_to_train['Stacking Ensemble'] = stacking_ensemble

        # Train & evaluate each model
        results = []
        trained_models = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        for mname, base_model in models_to_train.items():
            print(f"\n--- Training {mname} ---")
            steps = []
            steps.append(('feature_selection', SelectKBest(f_classif, k=min(self.k_features, X_train_processed.shape[1]))))
            if use_smote:
                steps.append(('smote', SMOTE(k_neighbors=k_neighbors, random_state=RANDOM_STATE)))
                steps.append(('classifier', base_model))
                pipeline = ImbPipeline(steps)
            else:
                steps.append(('classifier', base_model))
                pipeline = Pipeline(steps)

            # CV metrics on training set
            cv_accs, cv_bal_accs, cv_f1s, cv_rocs = [], [], [], []
            for train_idx, val_idx in cv.split(X_train_processed, y_train):
                Xtr, Xval = X_train_processed[train_idx], X_train_processed[val_idx]
                ytr, yval = y_train[train_idx], y_train[val_idx]
                try:
                    pipeline.fit(Xtr, ytr)
                    ypred = pipeline.predict(Xval)
                    yprob = pipeline.predict_proba(Xval) if hasattr(pipeline, 'predict_proba') else None
                    cv_accs.append(accuracy_score(yval, ypred))
                    cv_bal_accs.append(balanced_accuracy_score(yval, ypred))
                    cv_f1s.append(f1_score(yval, ypred, average='weighted'))
                    if yprob is not None:
                        try:
                            cv_rocs.append(roc_auc_score(yval, yprob, multi_class='ovr', average='macro'))
                        except Exception:
                            cv_rocs.append(0.0)
                    else:
                        cv_rocs.append(0.0)
                except Exception as e:
                    print(f"  CV fold failed for {mname}: {e}")
                    cv_accs.append(0.0); cv_bal_accs.append(0.0); cv_f1s.append(0.0); cv_rocs.append(0.0)

            # Fit on full training set
            pipeline.fit(X_train_processed, y_train)

            # Test evaluation
            y_test_pred = pipeline.predict(X_test_processed)
            test_acc = accuracy_score(y_test, y_test_pred)
            test_bal = balanced_accuracy_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred, average='weighted')
            try:
                y_test_proba = pipeline.predict_proba(X_test_processed)
                test_roc = roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='macro')
            except Exception:
                test_roc = 0.0

            # Bootstrap CI
            mean_acc_boot, ci_low, ci_high = bootstrap_metric(pipeline, X_test_processed, y_test, accuracy_score, n_iterations=1000)

            results.append({
                'Model': mname,
                'Train_Accuracy': float(np.mean(cv_accs)),
                'Train_Acc_Std': float(np.std(cv_accs)),
                'Test_Accuracy': float(test_acc),
                'Test_CI_Lower': float(ci_low),
                'Test_CI_Upper': float(ci_high),
                'Balanced_Accuracy': float(test_bal),
                'F1_Score': float(test_f1),
                'ROC_AUC': float(test_roc),
                'Overfitting': float(np.mean(cv_accs) - test_acc),
                'CV_Balanced_Acc': float(np.mean(cv_bal_accs)),
                'CV_F1': float(np.mean(cv_f1s)),
                'CV_ROC_AUC': float(np.mean(cv_rocs))
            })
            trained_models[mname] = pipeline
            print(f"  CV Acc: {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}")
            print(f"  Test Acc: {test_acc:.4f} [95% CI {ci_low:.4f}-{ci_high:.4f}]  Balanced: {test_bal:.4f}  ROC-AUC: {test_roc:.4f}")

        # Save results
        results_df = pd.DataFrame(results).sort_values('Balanced_Accuracy', ascending=False)
        results_df.to_csv(os.path.join(OUTPUT_DIR, 'model_results_detailed.csv'), index=False)
        self.models = trained_models
        self.results = results_df

        # Visualizations
        class_names = ['LOW', 'INTERMEDIATE', 'HIGH']
        try:
            plot_roc_curves(trained_models, X_test_processed, y_test, class_names, os.path.join(OUTPUT_DIR, 'roc_curves.png'))
            plot_calibration_curves(trained_models, X_test_processed, y_test, class_names, os.path.join(OUTPUT_DIR, 'calibration_curves.png'))
            plot_confusion_matrices(trained_models, X_test_processed, y_test, class_names, os.path.join(OUTPUT_DIR, 'confusion_matrices.png'))
            # Feature importance for top model (if exposed)
            top_model = trained_models[results_df.iloc[0]['Model']]
            plot_feature_importance(top_model, self.feature_names, os.path.join(OUTPUT_DIR, 'feature_importance.png'))
        except Exception as e:
            print("Visualization step failed:", e)

        # Best model report
        best_row = results_df.iloc[0]
        best_model_name = best_row['Model']
        best_model = trained_models[best_model_name]
        try:
            y_pred_best = best_model.predict(X_test_processed)
            print("\n" + "=" * 80)
            print(f"BEST MODEL: {best_model_name}")
            print(results_df[['Model', 'Test_Accuracy', 'Balanced_Accuracy', 'ROC_AUC']].to_string(index=False))
            print("\nClassification report (best model):")
            print(classification_report(y_test, y_pred_best, target_names=class_names, digits=4))
            print("\nConfusion matrix (best model):")
            print(pd.DataFrame(confusion_matrix(y_test, y_pred_best), index=class_names, columns=class_names))
        except Exception as e:
            print("Could not produce best-model report:", e)

        print("\nSaved outputs to:", OUTPUT_DIR)
        return trained_models, results_df, (X_test_processed, y_test)


# ---------------- Main ----------------
def main():
    DATA_PATH = './public/cvd_dataset.csv'  # change if needed
    pipeline = CVDModelPipeline(DATA_PATH, k_features=150)
    models, results, test = pipeline.prepare_and_train(test_size=0.2, optimize_lgbm=True, lgbm_trials=30)
    return pipeline, models, results


if __name__ == '__main__':
    main()
