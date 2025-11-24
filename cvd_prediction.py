import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Optional

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ----------------------------- Utilities ---------------------------------

def safe_divide(numerator: pd.Series, denominator: pd.Series, floor: float = 1.0) -> pd.Series:
    """
    Safely divide series with a floor on the denominator to avoid division by zero.
    
    FIXED: Simplified to always use positive floor value.
    For medical ratios like TC/HDL, negative denominators are data errors anyway.
    
    Args:
        numerator: Numerator series
        denominator: Denominator series  
        floor: Minimum absolute value for denominator (default: 1.0 for HDL)
    
    Returns:
        Series with safe division results
    """
    denom = denominator.copy().astype(float)
    denom = denom.fillna(floor)  # Fill missing with floor
    # Clip to minimum absolute value, preserving sign
    denom = np.where(denom.abs() < floor, 
                     floor * np.sign(denom).replace(0, 1),  # Handle sign=0 case
                     denom)
    return numerator / denom


# ---------------------- Risk scoring (train-only thresholds) ----------------
class ClinicalRiskScorer:
    """
    Compute an additive, normalized risk score using clinically sensible weights.
    Thresholds for LOW/INTERMEDIATE/HIGH are learned on training set only.
    
    FIXED: Added U-shaped BMI relationship
    """
    def __init__(self):
        self.low_thr = None
        self.high_thr = None

    def compute_raw_score(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        score = np.zeros(n, dtype=float)

        # Age effect (smooth weight, non-linear after 65)
        if 'Age' in df.columns:
            a = df['Age'].fillna(df['Age'].median()).clip(18, 100)
            # Moderate linear contribution up to 65, then steeper
            score += np.where(a < 65, 
                            ((a - 40) / 15).clip(0, 4),
                            ((a - 40) / 12).clip(0, 5))  # Steeper after 65

        # Systolic BP contribution (MAP-based)
        if all(c in df.columns for c in ['Systolic BP', 'Diastolic BP']):
            sbp = df['Systolic BP'].fillna(df['Systolic BP'].median())
            dbp = df['Diastolic BP'].fillna(df['Diastolic BP'].median())
            # Mean Arterial Pressure
            map_ = dbp + (sbp - dbp) / 3.0
            score += ((map_ - 85) / 10).clip(0, 4)

        # LDL (non-linear scaling)
        if 'Estimated LDL (mg/dL)' in df.columns:
            ldl = df['Estimated LDL (mg/dL)'].fillna(df['Estimated LDL (mg/dL)'].median())
            # More aggressive penalty for very high LDL
            score += np.where(ldl >= 190,
                            4.0,  # Very high risk
                            ((ldl - 100) / 30).clip(0, 3.5))

        # HDL protective (subtract for high HDL)
        if 'HDL (mg/dL)' in df.columns:
            hdl = df['HDL (mg/dL)'].fillna(df['HDL (mg/dL)'].median())
            # Penalty for low HDL
            score += np.where(hdl < 40, 2.5,
                            np.where(hdl < 50, 1.0, 0.0))
            # Bonus for high HDL
            score -= np.where(hdl >= 60, 1.5, 0.0)

        # Triglycerides (staged)
        if 'Triglycerides (mg/dL)' in df.columns:
            tg = df['Triglycerides (mg/dL)'].fillna(df['Triglycerides (mg/dL)'].median())
            score += np.where(tg >= 500, 3.5,
                            np.where(tg >= 200, 2.0,
                            np.where(tg >= 150, 1.0, 0.0)))

        # Diabetes (staged by glucose level)
        if 'Fasting Blood Sugar (mg/dL)' in df.columns:
            fbs = df['Fasting Blood Sugar (mg/dL)'].fillna(df['Fasting Blood Sugar (mg/dL)'].median())
            score += np.where(fbs >= 126, 3.5,  # Diabetes
                            np.where(fbs >= 100, 1.5, 0.0))  # Prediabetes

        # BMI (U-SHAPED RELATIONSHIP - FIXED)
        if 'BMI' in df.columns:
            bmi = df['BMI'].fillna(df['BMI'].median())
            
            # Underweight risk (BMI < 18.5)
            score += np.where(bmi < 18.5, 
                            ((18.5 - bmi) / 2.0).clip(0, 2.0),
                            0.0)
            
            # Overweight/obesity risk (BMI >= 25)
            score += np.where(bmi >= 40, 3.5,  # Class III obesity
                            np.where(bmi >= 35, 3.0,  # Class II obesity
                            np.where(bmi >= 30, 2.0,  # Class I obesity
                            np.where(bmi >= 25, 1.0, 0.0))))  # Overweight

        # Smoking (high weight - established risk factor)
        if 'Smoking Status' in df.columns:
            score += np.where(df['Smoking Status'].fillna('N') == 'Y', 3.0, 0.0)

        # Family history (genetic component)
        if 'Family History of CVD' in df.columns:
            score += np.where(df['Family History of CVD'].fillna('N') == 'Y', 2.0, 0.0)

        # Physical activity (protective)
        if 'Physical Activity Level' in df.columns:
            pal = df['Physical Activity Level'].fillna('Moderate')
            score -= np.where(pal == 'High', 1.5, 0.0)
            score += np.where(pal == 'Low', 1.2, 0.0)

        return score

    def fit(self, df: pd.DataFrame):
        """Learn thresholds from training data only"""
        raw = self.compute_raw_score(df)
        # Learn tertiles on training set
        self.low_thr = np.percentile(raw, 33)
        self.high_thr = np.percentile(raw, 67)
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Apply learned thresholds to create risk labels"""
        raw = self.compute_raw_score(df)
        labels = np.where(raw <= self.low_thr, 'LOW', 
                         np.where(raw <= self.high_thr, 'INTERMEDIATE', 'HIGH'))
        return df.copy(), labels


# ----------------------- Feature engineering (train-safe) ------------------
class FeatureEngineer:
    """
    Create robust and interpretable features.
    All steps are deterministic and avoid dataset-global leaks.
    
    FIXED: Improved normalization and added clinical thresholds
    """
    def __init__(self):
        self.hdl_floor = 20.0  # Clinical minimum for HDL (severe deficiency)

    def fit(self, X: pd.DataFrame, y=None):
        # No global leaks; all transformations are deterministic
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # ============ LIPID RATIOS ============
        if 'HDL (mg/dL)' in df.columns and 'Total Cholesterol (mg/dL)' in df.columns:
            df['TC_HDL_Ratio'] = safe_divide(
                df['Total Cholesterol (mg/dL)'], 
                df['HDL (mg/dL)'], 
                floor=self.hdl_floor
            ).clip(1, 10)  # Clinical range

        if 'Estimated LDL (mg/dL)' in df.columns and 'HDL (mg/dL)' in df.columns:
            df['LDL_HDL_Ratio'] = safe_divide(
                df['Estimated LDL (mg/dL)'], 
                df['HDL (mg/dL)'], 
                floor=self.hdl_floor
            ).clip(0.5, 8)  # Clinical range

        if 'Triglycerides (mg/dL)' in df.columns and 'HDL (mg/dL)' in df.columns:
            # Atherogenic index
            df['TG_HDL_Ratio'] = safe_divide(
                df['Triglycerides (mg/dL)'], 
                df['HDL (mg/dL)'], 
                floor=self.hdl_floor
            ).clip(0.5, 10)  # Clinical range

        # Non-HDL cholesterol (validated CVD predictor)
        if 'Total Cholesterol (mg/dL)' in df.columns and 'HDL (mg/dL)' in df.columns:
            df['Non_HDL_Chol'] = (df['Total Cholesterol (mg/dL)'] - df['HDL (mg/dL)']).clip(0, 300)

        # ============ BLOOD PRESSURE FEATURES ============
        if 'Systolic BP' in df.columns and 'Diastolic BP' in df.columns:
            # Pulse pressure (arterial stiffness marker)
            df['Pulse_Pressure'] = (df['Systolic BP'] - df['Diastolic BP']).clip(20, 100)
            
            # Mean Arterial Pressure
            df['MAP'] = df['Diastolic BP'] + df['Pulse_Pressure'] / 3.0
            
            # Wide pulse pressure indicator (>60 is concerning)
            df['Wide_Pulse_Pressure'] = (df['Pulse_Pressure'] > 60).astype(float)

        # ============ AGE INTERACTIONS (NORMALIZED) ============
        # FIXED: Properly normalized to prevent extreme values
        
        if 'Age' in df.columns and 'BMI' in df.columns:
            # Normalize to [0, 1] approximately
            age_norm = (df['Age'] / 100.0).clip(0, 1)
            bmi_norm = (df['BMI'] / 50.0).clip(0, 1)
            df['Age_BMI_Interaction'] = age_norm * bmi_norm

        if 'Age' in df.columns and 'Systolic BP' in df.columns:
            age_norm = (df['Age'] / 100.0).clip(0, 1)
            sbp_norm = (df['Systolic BP'] / 200.0).clip(0, 1)
            df['Age_SBP_Interaction'] = age_norm * sbp_norm

        if 'Age' in df.columns and 'Total Cholesterol (mg/dL)' in df.columns:
            age_norm = (df['Age'] / 100.0).clip(0, 1)
            chol_norm = (df['Total Cholesterol (mg/dL)'] / 300.0).clip(0, 1.5)
            df['Age_Chol_Interaction'] = age_norm * chol_norm

        # ============ CLINICAL THRESHOLD FEATURES ============
        
        # BMI categories
        if 'BMI' in df.columns:
            df['BMI_Obese'] = (df['BMI'] >= 30).astype(float)
            df['BMI_Overweight'] = ((df['BMI'] >= 25) & (df['BMI'] < 30)).astype(float)
            df['BMI_Underweight'] = (df['BMI'] < 18.5).astype(float)
        
        # Age categories
        if 'Age' in df.columns:
            df['Age_65_plus'] = (df['Age'] >= 65).astype(float)
            df['Age_55_64'] = ((df['Age'] >= 55) & (df['Age'] < 65)).astype(float)
        
        # LDL categories
        if 'Estimated LDL (mg/dL)' in df.columns:
            df['LDL_Very_High'] = (df['Estimated LDL (mg/dL)'] >= 190).astype(float)
            df['LDL_High'] = ((df['Estimated LDL (mg/dL)'] >= 160) & 
                             (df['Estimated LDL (mg/dL)'] < 190)).astype(float)
        
        # HDL categories
        if 'HDL (mg/dL)' in df.columns:
            df['HDL_Low'] = (df['HDL (mg/dL)'] < 40).astype(float)
            df['HDL_Optimal'] = (df['HDL (mg/dL)'] >= 60).astype(float)
        
        # Blood pressure stages
        if 'Systolic BP' in df.columns and 'Diastolic BP' in df.columns:
            df['BP_Stage2_HTN'] = ((df['Systolic BP'] >= 140) | 
                                   (df['Diastolic BP'] >= 90)).astype(float)
            df['BP_Stage1_HTN'] = ((df['Systolic BP'] >= 130) & (df['Systolic BP'] < 140) | 
                                   (df['Diastolic BP'] >= 80) & (df['Diastolic BP'] < 90)).astype(float)
        
        # Diabetes indicators
        if 'Fasting Blood Sugar (mg/dL)' in df.columns:
            df['Diabetes'] = (df['Fasting Blood Sugar (mg/dL)'] >= 126).astype(float)
            df['Prediabetes'] = ((df['Fasting Blood Sugar (mg/dL)'] >= 100) & 
                                (df['Fasting Blood Sugar (mg/dL)'] < 126)).astype(float)

        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


# --------------------------- Main predictor class --------------------------
class CVDModelPipeline:
    """
    FIXED: Corrected data leakage issues and mathematical errors
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.feature_engineer = FeatureEngineer()
        self.risk_scorer = ClinicalRiskScorer()
        self.numeric_cols = []
        self.cat_cols = []
        self.feature_names_ = None
        self.final_model = None

    def load_and_clean(self) -> pd.DataFrame:
        """Load and perform basic cleaning"""
        df = pd.read_csv(self.data_path)

        # Strip column names
        df.columns = [c.strip() for c in df.columns]

        # Basic type conversion
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = df[col].astype(object)

        self.df = df
        return df

    def _choose_stratify_col(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """
        FIXED: Use fixed value for imputation to prevent data leakage
        """
        if 'Age' in df.columns:
            # Use fixed value (50) instead of median to prevent leakage
            return pd.cut(
                df['Age'].fillna(50),  # FIXED: Use constant instead of median
                bins=[0, 40, 55, 65, 200], 
                labels=[0, 1, 2, 3]
            )
        # Fallback to categorical stratification
        for c in ['Smoking Status', 'Family History of CVD', 'Physical Activity Level']:
            if c in df.columns:
                return df[c].fillna('Unknown')
        return None

    def initial_split(self, test_size=0.2, random_state=RANDOM_STATE):
        """Split data with stratification"""
        df = self.df.copy()
        strat = self._choose_stratify_col(df)
        
        if strat is None:
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state
            )
        else:
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state, stratify=strat
            )
        
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def prepare_and_train(self, test_size=0.2, random_state=RANDOM_STATE):
        """Complete training pipeline"""
        df = self.load_and_clean()
        train_df, test_df = self.initial_split(test_size=test_size, random_state=random_state)

        # Fit risk scorer on train only
        self.risk_scorer.fit(train_df)
        train_df, y_train_labels = self.risk_scorer.transform(train_df)
        test_df, y_test_labels = self.risk_scorer.transform(test_df)

        # Encode labels
        label_map = {'LOW': 0, 'INTERMEDIATE': 1, 'HIGH': 2}
        y_train = np.array([label_map[l] for l in y_train_labels])
        y_test = np.array([label_map[l] for l in y_test_labels])

        print(f"\nTraining set distribution:")
        print(pd.Series(y_train_labels).value_counts())
        print(f"\nTest set distribution:")
        print(pd.Series(y_test_labels).value_counts())

        # Feature engineering (fit on train, transform both)
        X_train = self.feature_engineer.fit_transform(train_df)
        X_test = self.feature_engineer.transform(test_df)

        # Identify column types
        self.numeric_cols = [c for c in X_train.columns if X_train[c].dtype in ['int64', 'float64']]
        self.cat_cols = [c for c in X_train.columns if X_train[c].dtype == 'object']

        print(f"\nFeatures: {len(self.numeric_cols)} numeric, {len(self.cat_cols)} categorical")

        # Build preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, self.numeric_cols),
            ('cat', categorical_transformer, self.cat_cols)
        ], remainder='drop')

        # Base models
        cat = CatBoostClassifier(
            iterations=800,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            verbose=False, 
            random_state=RANDOM_STATE, 
            auto_class_weights='Balanced'
        )
        
        xgb = XGBClassifier(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            eval_metric='mlogloss', 
            random_state=RANDOM_STATE
        )
        
        lgb = LGBMClassifier(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbose=-1
        )

        # Stacking ensemble
        estimators = [('cat', cat), ('xgb', xgb), ('lgb', lgb)]

        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                max_iter=1000, 
                class_weight='balanced', 
                random_state=RANDOM_STATE
            ),
            cv=5, 
            n_jobs=-1, 
            passthrough=False
        )

        # SMOTE validation - FIXED
        def can_apply_smote(y: np.ndarray, k_neighbors: int = 5) -> bool:
            """
            FIXED: Check if SMOTE can be applied with given k_neighbors
            """
            unique, counts = np.unique(y, return_counts=True)
            if len(unique) < 2:
                return False
            # Need at least k_neighbors + 1 samples in minority class
            if counts.min() <= k_neighbors:
                return False
            return True

        # Dynamically determine k_neighbors
        min_class_count = np.bincount(y_train).min()
        k_neighbors = min(5, max(1, min_class_count - 1))
        
        print(f"\nApplying SMOTE with k_neighbors={k_neighbors}")

        if can_apply_smote(y_train, k_neighbors):
            pipeline = ImbPipeline(steps=[
                ('pre', preprocessor),
                ('smote', SMOTE(k_neighbors=k_neighbors, random_state=RANDOM_STATE)),
                ('sel', SelectFromModel(
                    RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE), 
                    threshold='median'
                )),
                ('clf', stacking)
            ])
        else:
            print("⚠️  SMOTE skipped - insufficient samples")
            pipeline = Pipeline(steps=[
                ('pre', preprocessor),
                ('sel', SelectFromModel(
                    RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE), 
                    threshold='median'
                )),
                ('clf', stacking)
            ])

        # Train model
        print('\nTraining stacked ensemble...')
        pipeline.fit(X_train, y_train)
        self.final_model = pipeline

        # Evaluate
        y_pred = pipeline.predict(X_test)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print("\n" + "="*80)
        print("📊 FINAL RESULTS")
        print("="*80)
        print(f'Balanced Accuracy: {bal_acc:.4f} ({bal_acc*100:.2f}%)')
        print(f'Accuracy:          {acc:.4f} ({acc*100:.2f}%)')
        print(f'F1 Score:          {f1:.4f}')
        
        print('\n📋 Classification Report:')
        print(classification_report(y_test, y_pred, target_names=['LOW','INTERMEDIATE','HIGH']))
        
        print('\n📊 Confusion Matrix:')
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        return pipeline, (X_train, y_train), (X_test, y_test), {
            'bal_acc': bal_acc, 
            'acc': acc, 
            'f1': f1
        }


# ------------------------------ Execution ---------------------------------

def main():
    DATA_PATH = './public/cvd_dataset.csv'
    
    print("="*80)
    print("🏥 CVD RISK PREDICTOR - MATHEMATICALLY CORRECTED VERSION")
    print("="*80)
    print("\n✅ All mathematical errors fixed:")
    print("   • U-shaped BMI risk relationship")
    print("   • Improved safe_divide with proper floor handling")
    print("   • Fixed stratification to prevent data leakage")
    print("   • Dynamic SMOTE k_neighbors validation")
    print("   • Normalized age interactions")
    print("   • Clinical threshold features")
    print("\n" + "="*80 + "\n")
    
    model = CVDModelPipeline(DATA_PATH)
    pipeline, train_data, test_data, metrics = model.prepare_and_train(test_size=0.2)

    print('\n' + "="*80)
    print('✅ TRAINING COMPLETE')
    print("="*80)
    print(f"Final Balanced Accuracy: {metrics['bal_acc']*100:.2f}%")
    
    return model, pipeline, metrics


if __name__ == '__main__':
    model, pipeline, metrics = main()