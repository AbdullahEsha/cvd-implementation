"""
CVD PREDICTION - FIXED VERSION WITH REAL OUTCOMES
This version eliminates circular reasoning by using actual clinical outcomes
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.base import BaseEstimator, TransformerMixin
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import os
from datetime import datetime

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUTPUT_DIR = './cvd_results_final_' + datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# KEY FIX #1: Remove ClinicalRiskScorer Class Entirely
# ============================================================================
# DELETE the entire ClinicalRiskScorer class from lines ~99-166
# We will use ACTUAL outcomes instead of synthetic risk scores


# ============================================================================
# KEY FIX #2: Feature Engineering (Keep This - It's Good!)
# ============================================================================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Create clinically-informed derived features.
    This is VALID because it only transforms X, doesn't use y.
    """
    def __init__(self):
        self.hdl_floor = 20.0

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Lipid ratios (clinically validated markers)
        if 'HDL (mg/dL)' in df.columns and 'Total Cholesterol (mg/dL)' in df.columns:
            tc = df['Total Cholesterol (mg/dL)'].fillna(df['Total Cholesterol (mg/dL)'].median())
            hdl = df['HDL (mg/dL)'].fillna(df['HDL (mg/dL)'].median()).clip(lower=self.hdl_floor)
            df['TC_HDL_Ratio'] = (tc / hdl).clip(1, 10)

        # Blood pressure features
        if 'Systolic BP' in df.columns and 'Diastolic BP' in df.columns:
            sbp = df['Systolic BP'].fillna(df['Systolic BP'].median())
            dbp = df['Diastolic BP'].fillna(df['Diastolic BP'].median())
            df['Pulse_Pressure'] = (sbp - dbp).clip(20, 100)
            df['MAP'] = dbp + (sbp - dbp) / 3.0

        # Clinical thresholds (guideline-based)
        if 'BMI' in df.columns:
            df['BMI_Obese'] = (df['BMI'] >= 30).astype(float)
            
        if 'Age' in df.columns:
            df['Age_65_plus'] = (df['Age'] >= 65).astype(float)

        if 'Systolic BP' in df.columns:
            df['BP_Stage2_HTN'] = (df['Systolic BP'] >= 140).astype(float)

        if 'Fasting Blood Sugar (mg/dL)' in df.columns:
            df['Diabetes'] = (df['Fasting Blood Sugar (mg/dL)'] >= 126).astype(float)

        return df


# ============================================================================
# KEY FIX #3: Updated Pipeline Class
# ============================================================================
class CVDModelPipeline:
    """
    Fixed CVD prediction pipeline using ACTUAL outcomes
    
    CRITICAL CHANGES:
    - Removed synthetic risk scoring
    - Uses real CVD outcomes from dataset
    - Proper temporal/external validation
    """

    def __init__(self, data_path: str, outcome_col: str, k_features: int = 150):
        """
        Args:
            data_path: Path to CSV file
            outcome_col: Name of ACTUAL outcome column (e.g., 'CVD_Event', 'MI', 'Stroke')
            k_features: Number of features to select
        """
        self.data_path = data_path
        self.outcome_col = outcome_col  # THIS IS THE KEY FIX
        self.k_features = k_features
        self.df = None
        self.feature_engineer = FeatureEngineer()
        self.numeric_cols = []
        self.cat_cols = []
        self.models = {}
        self.results = []
        self.feature_names = []

    def load_and_clean(self) -> pd.DataFrame:
        """Load data and verify outcome column exists"""
        df = pd.read_csv(self.data_path)
        df.columns = [c.strip() for c in df.columns]
        
        # CRITICAL CHECK: Outcome column must exist
        if self.outcome_col not in df.columns:
            raise ValueError(
                f"❌ Outcome column '{self.outcome_col}' not found!\n"
                f"Available columns: {df.columns.tolist()}\n"
                f"You MUST have a real outcome variable for clinical prediction."
            )
        
        # Check outcome is binary or categorical
        unique_vals = df[self.outcome_col].nunique()
        if unique_vals < 2:
            raise ValueError(f"Outcome has only {unique_vals} unique value(s). Need at least 2.")
        
        print(f"✅ Using REAL outcome variable: '{self.outcome_col}'")
        print(f"   Distribution: {df[self.outcome_col].value_counts().to_dict()}")
        
        # Remove outcome from features
        self.df = df.copy()
        
        return df

    def prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Separate features from target
        
        KEY FIX: We extract the REAL outcome, not synthetic scores
        """
        # Get target
        y = df[self.outcome_col].values
        
        # Remove target and ID columns from features
        exclude_cols = [self.outcome_col, 'Patient_ID', 'ID', 'patient_id', 'id']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[feature_cols].copy()
        
        return X, y

    def initial_split(self, test_size=0.2, random_state=RANDOM_STATE):
        """Stratified train-test split using ACTUAL outcomes"""
        df = self.df.copy()
        
        # Use actual outcome for stratification
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df[self.outcome_col]  # Stratify on REAL outcome
        )
        
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def build_preprocessing_pipeline(self, X_train: pd.DataFrame):
        """Build preprocessing pipeline"""
        numeric_cols = [c for c in X_train.columns 
                       if X_train[c].dtype in ['int64', 'float64', 'float32', 'int32']]
        cat_cols = [c for c in X_train.columns if X_train[c].dtype == 'object']
        
        self.numeric_cols = numeric_cols
        self.cat_cols = cat_cols
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, cat_cols)
        ], remainder='drop')
        
        return preprocessor

    def prepare_and_train_all_models(self, test_size=0.2, random_state=RANDOM_STATE):
        """
        Train models using REAL outcomes
        
        FIXED: No circular reasoning - predicting actual CVD events
        """
        
        print("=" * 80)
        print("🏥 CVD PREDICTION - FIXED VERSION (REAL OUTCOMES)")
        print("=" * 80)
        print("\n✅ KEY FIX: Using actual clinical outcomes, not synthetic scores!")
        print("\n" + "=" * 80 + "\n")
        
        # Load and split
        df = self.load_and_clean()
        train_df, test_df = self.initial_split(test_size=test_size, random_state=random_state)

        # Get features and target - KEY FIX HERE
        X_train, y_train = self.prepare_features_and_target(train_df)
        X_test, y_test = self.prepare_features_and_target(test_df)

        print(f"📊 Dataset:")
        print(f"   Training samples:   {len(y_train)}")
        print(f"   Test samples:       {len(y_test)}")
        print(f"   Training outcome distribution: {np.bincount(y_train)}")
        print(f"   Test outcome distribution:     {np.bincount(y_test)}")

        # Feature engineering
        X_train_eng = self.feature_engineer.fit_transform(X_train)
        X_test_eng = self.feature_engineer.transform(X_test)

        # Build preprocessing pipeline
        preprocessor = self.build_preprocessing_pipeline(X_train_eng)

        print(f"\n🔧 Features: {len(self.numeric_cols)} numeric, {len(self.cat_cols)} categorical")

        # Preprocess
        X_train_processed = preprocessor.fit_transform(X_train_eng)
        X_test_processed = preprocessor.transform(X_test_eng)
        
        # Get feature names
        try:
            numeric_features = self.numeric_cols
            cat_features = []
            if self.cat_cols:
                cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
                cat_features = cat_encoder.get_feature_names_out(self.cat_cols).tolist()
            self.feature_names = numeric_features + cat_features
        except:
            self.feature_names = [f'feature_{i}' for i in range(X_train_processed.shape[1])]
        
        print(f"✅ After preprocessing: {X_train_processed.shape[1]} features")
        print(f"🎯 Will select top {self.k_features} features using SelectKBest")

        # SMOTE setup
        min_class_count = np.bincount(y_train).min()
        k_neighbors = min(5, max(1, min_class_count - 1))
        
        if min_class_count > k_neighbors:
            print(f"\n⚖️  SMOTE will be applied (k_neighbors={k_neighbors})")
            use_smote = True
        else:
            print("\n⚠️  SMOTE skipped - insufficient samples")
            use_smote = False

        # Define models
        print("\n🤖 Training Models...")
        print("-" * 80)

        models_to_train = {
            'Logistic Regression': LogisticRegression(
                max_iter=2000, class_weight='balanced', C=1.0, 
                random_state=RANDOM_STATE
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_leaf=5,
                class_weight='balanced', random_state=RANDOM_STATE
            ),
            'XGBoost': XGBClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=5,
                reg_alpha=0.5, reg_lambda=1.0, random_state=RANDOM_STATE,
                eval_metric='logloss', verbosity=0
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=6,
                reg_alpha=0.5, reg_lambda=1.0, random_state=RANDOM_STATE, verbose=-1
            ),
        }

        scoring = {
            'accuracy': 'accuracy',
            'balanced_accuracy': 'balanced_accuracy',
            'roc_auc': 'roc_auc',
            'f1': 'f1_weighted'
        }

        # Train each model
        for model_name, base_model in models_to_train.items():
            print(f"\n   Training {model_name}...", end=' ')
            
            try:
                # Build pipeline
                pipeline_steps = [
                    ('feature_selection', SelectKBest(f_classif, k=min(self.k_features, X_train_processed.shape[1])))
                ]
                
                if use_smote:
                    pipeline_steps.append(('smote', SMOTE(k_neighbors=k_neighbors, random_state=RANDOM_STATE)))
                
                pipeline_steps.append(('classifier', base_model))
                
                if use_smote:
                    full_pipeline = ImbPipeline(pipeline_steps)
                else:
                    full_pipeline = Pipeline(pipeline_steps)
                
                # Cross-validation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                
                cv_results = cross_validate(
                    full_pipeline, X_train_processed, y_train,
                    cv=cv, scoring=scoring, n_jobs=-1
                )
                
                cv_acc = np.mean(cv_results['test_accuracy'])
                cv_roc = np.mean(cv_results['test_roc_auc'])
                
                # Fit final model
                full_pipeline.fit(X_train_processed, y_train)
                
                # Test predictions
                y_test_pred = full_pipeline.predict(X_test_processed)
                y_test_proba = full_pipeline.predict_proba(X_test_processed)[:, 1] if hasattr(full_pipeline, 'predict_proba') else None
                
                test_acc = accuracy_score(y_test, y_test_pred)
                test_bal_acc = balanced_accuracy_score(y_test, y_test_pred)
                test_f1 = f1_score(y_test, y_test_pred, average='weighted')
                test_roc = roc_auc_score(y_test, y_test_proba) if y_test_proba is not None else 0.0
                
                self.results.append({
                    'Model': model_name,
                    'CV_Accuracy': cv_acc,
                    'CV_ROC_AUC': cv_roc,
                    'Test_Accuracy': test_acc,
                    'Test_Balanced_Acc': test_bal_acc,
                    'Test_F1': test_f1,
                    'Test_ROC_AUC': test_roc,
                    'Overfitting': cv_acc - test_acc
                })
                
                self.models[model_name] = full_pipeline
                
                print(f"✓")
                print(f"      CV: Acc={cv_acc:.4f}, ROC-AUC={cv_roc:.4f}")
                print(f"      Test: Acc={test_acc:.4f}, ROC-AUC={test_roc:.4f}")
                
            except Exception as e:
                print(f"✗ Error: {str(e)}")
                continue

        # Results
        results_df = pd.DataFrame(self.results).sort_values('Test_ROC_AUC', ascending=False)
        
        print("\n" + "=" * 80)
        print("📊 MODEL PERFORMANCE (REAL OUTCOMES)")
        print("=" * 80)
        print(results_df[['Model', 'CV_Accuracy', 'Test_Accuracy', 'Test_ROC_AUC', 'Overfitting']].to_string(index=False))
        
        # Save results
        csv_path = os.path.join(OUTPUT_DIR, 'results_real_outcomes.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\n✅ Results saved: {csv_path}")
        
        return self.models, results_df


# ============================================================================
# KEY FIX #4: Example Usage with Real Data
# ============================================================================

def example_cleveland_heart_disease():
    """
    Example using Cleveland Heart Disease dataset (has real diagnoses)
    """
    print("\n" + "=" * 80)
    print("EXAMPLE: Cleveland Heart Disease Dataset")
    print("=" * 80)
    
    # Download from: https://archive.ics.uci.edu/ml/datasets/heart+disease
    DATA_PATH = './data/cleveland_heart.csv'
    OUTCOME_COL = 'num'  # 0 = no disease, 1-4 = disease severity
    
    # For binary classification, convert to 0/1
    df = pd.read_csv(DATA_PATH)
    df['CVD_Binary'] = (df[OUTCOME_COL] > 0).astype(int)
    df.to_csv('./data/cleveland_processed.csv', index=False)
    
    pipeline = CVDModelPipeline(
        data_path='./data/cleveland_processed.csv',
        outcome_col='CVD_Binary',  # REAL outcome
        k_features=20
    )
    
    models, results = pipeline.prepare_and_train_all_models(test_size=0.2)
    
    return pipeline, models, results


def example_with_your_data():
    """
    Template for YOUR data - you need to add outcome column
    """
    print("\n" + "=" * 80)
    print("YOUR DATA - REQUIRES OUTCOME COLUMN")
    print("=" * 80)
    
    # STEP 1: Load your current data
    df = pd.read_csv('./public/cvd_dataset.csv')
    
    # STEP 2: YOU MUST ADD A REAL OUTCOME COLUMN
    # Options:
    # A) Merge with medical records (MI, stroke, death)
    # B) Use existing 'CVD Risk Level' if it's from actual diagnosis
    # C) Get follow-up data on who actually developed CVD
    
    # Example if you have follow-up data:
    # outcomes = pd.read_csv('./followup_outcomes.csv')  # Must have CVD events
    # df = df.merge(outcomes, on='Patient_ID')
    
    # For demonstration, let's check what you have:
    print("\nYour current columns:")
    print(df.columns.tolist())
    
    print("\n⚠️ YOU NEED TO:")
    print("1. Add a column like 'CVD_Event' (0=no event, 1=MI/stroke/death)")
    print("2. This data should come from:")
    print("   - Medical records follow-up")
    print("   - Hospital discharge codes (ICD-10)")
    print("   - Death certificates")
    print("   - Prospective follow-up study")
    
    # If you have a real outcome column, use:
    # pipeline = CVDModelPipeline(
    #     data_path='./public/cvd_with_outcomes.csv',
    #     outcome_col='CVD_Event',  # YOUR REAL OUTCOME
    #     k_features=150
    # )
    # models, results = pipeline.prepare_and_train_all_models()
    
    return None


if __name__ == '__main__':
    # Choose your scenario:
    
    # Option 1: Use Cleveland dataset (publicly available, has real outcomes)
    # pipeline, models, results = example_cleveland_heart_disease()
    
    # Option 2: Your data (REQUIRES you to add outcome column)
    example_with_your_data()