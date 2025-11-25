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

def safe_divide(numerator: pd.Series, denominator: pd.Series, floor: float = 1e-2) -> pd.Series:
    """Safely divide series with a floor on the denominator to avoid huge ratios.
    floor is an absolute clinical floor (e.g., 1e-2) not percentile-based.
    """
    denom = denominator.copy().astype(float)
    denom = denom.fillna(0.0)
    denom = denom.where(denom.abs() >= floor, floor * np.sign(denom) + floor)
    return numerator / denom


# ---------------------- Risk scoring (train-only thresholds) ----------------
class ClinicalRiskScorer:
    """Compute an additive, normalized risk score using clinically sensible weights.
    Thresholds for LOW/INTERMEDIATE/HIGH are learned on training set only.
    """
    def __init__(self):
        self.low_thr = None
        self.high_thr = None

    def compute_raw_score(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        score = np.zeros(n, dtype=float)

        # Age effect (smooth weight)
        if 'Age' in df.columns:
            a = df['Age'].fillna(df['Age'].median()).clip(18, 100)
            score += ((a - 40) / 15).clip(0, 4)  # moderate linear contribution

        # Systolic BP contribution (capped)
        if all(c in df.columns for c in ['Systolic BP', 'Diastolic BP']):
            sbp = df['Systolic BP'].fillna(df['Systolic BP'].median())
            dbp = df['Diastolic BP'].fillna(df['Diastolic BP'].median())
            # MAP surrogate
            map_ = dbp + (sbp - dbp) / 3.0
            score += ((map_ - 85) / 10).clip(0, 4)

        # LDL
        if 'Estimated LDL (mg/dL)' in df.columns:
            ldl = df['Estimated LDL (mg/dL)'].fillna(df['Estimated LDL (mg/dL)'].median())
            score += ((ldl - 100) / 30).clip(0, 4)

        # HDL protective (subtract)
        if 'HDL (mg/dL)' in df.columns:
            hdl = df['HDL (mg/dL)'].fillna(df['HDL (mg/dL)'].median())
            score -= ((60 - hdl) / 10).clip(0, 3)

        # Triglycerides
        if 'Triglycerides (mg/dL)' in df.columns:
            tg = df['Triglycerides (mg/dL)'].fillna(df['Triglycerides (mg/dL)'].median())
            score += ((tg - 150) / 100).clip(0, 3)

        # Diabetes
        if 'Fasting Blood Sugar (mg/dL)' in df.columns:
            fbs = df['Fasting Blood Sugar (mg/dL)'].fillna(df['Fasting Blood Sugar (mg/dL)'].median())
            score += np.where(fbs >= 126, 3.0, np.where(fbs >= 100, 1.0, 0.0))

        # BMI
        if 'BMI' in df.columns:
            bmi = df['BMI'].fillna(df['BMI'].median())
            score += ((bmi - 25) / 5).clip(0, 3)

        # Smoking
        if 'Smoking Status' in df.columns:
            score += np.where(df['Smoking Status'].fillna('N') == 'Y', 2.5, 0.0)

        # Family history
        if 'Family History of CVD' in df.columns:
            score += np.where(df['Family History of CVD'].fillna('N') == 'Y', 1.5, 0.0)

        # Physical activity (protective)
        if 'Physical Activity Level' in df.columns:
            score -= np.where(df['Physical Activity Level'].fillna('Medium') == 'High', 1.0, 0.0)
            score += np.where(df['Physical Activity Level'].fillna('Medium') == 'Low', 0.8, 0.0)

        return score

    def fit(self, df: pd.DataFrame):
        raw = self.compute_raw_score(df)
        # Learn low/high using tertiles on training set
        self.low_thr = np.percentile(raw, 33)
        self.high_thr = np.percentile(raw, 67)
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        raw = self.compute_raw_score(df)
        labels = np.where(raw <= self.low_thr, 'LOW', np.where(raw <= self.high_thr, 'INTERMEDIATE', 'HIGH'))
        return df.copy(), labels


# ----------------------- Feature engineering (train-safe) ------------------
class FeatureEngineer:
    """Create robust and interpretable features.
    All steps are deterministic and avoid dataset-global leaks.
    """
    def __init__(self):
        # store any train-based stats if needed
        self.hdl_floor = 1.0  # absolute floor for HDL denominator

    def fit(self, X: pd.DataFrame, y=None):
        # No global leaks required; keep simple
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Lipid ratios
        if 'HDL (mg/dL)' in df.columns and 'Total Cholesterol (mg/dL)' in df.columns:
            df['TC_HDL_Ratio'] = safe_divide(df['Total Cholesterol (mg/dL)'], df['HDL (mg/dL)'], floor=self.hdl_floor)

        if 'Estimated LDL (mg/dL)' in df.columns and 'HDL (mg/dL)' in df.columns:
            df['LDL_HDL_Ratio'] = safe_divide(df['Estimated LDL (mg/dL)'], df['HDL (mg/dL)'], floor=self.hdl_floor)

        if 'Triglycerides (mg/dL)' in df.columns and 'HDL (mg/dL)' in df.columns:
            df['TG_HDL_Ratio'] = safe_divide(df['Triglycerides (mg/dL)'], df['HDL (mg/dL)'], floor=self.hdl_floor)

        # Non-HDL
        if 'Total Cholesterol (mg/dL)' in df.columns and 'HDL (mg/dL)' in df.columns:
            df['Non_HDL_Chol'] = df['Total Cholesterol (mg/dL)'] - df['HDL (mg/dL)']

        # Pulse pressure & MAP
        if 'Systolic BP' in df.columns and 'Diastolic BP' in df.columns:
            df['Pulse_Pressure'] = (df['Systolic BP'] - df['Diastolic BP']).clip(lower=0)
            df['MAP'] = df['Diastolic BP'] + df['Pulse_Pressure'] / 3.0

        # Age-BMI, Age-SBP interactions (normalized by plausible clinical maxes)
        if 'Age' in df.columns and 'BMI' in df.columns:
            df['Age_BMI_Interaction'] = (df['Age'] / 100.0) * (df['BMI'] / 50.0)

        if 'Age' in df.columns and 'Systolic BP' in df.columns:
            df['Age_SBP_Interaction'] = (df['Age'] / 100.0) * (df['Systolic BP'] / 200.0)

        # Simple bin features for BMI and Age
        if 'BMI' in df.columns:
            df['BMI_Obese'] = (df['BMI'] >= 30).astype(int)
        if 'Age' in df.columns:
            df['Age_65_plus'] = (df['Age'] >= 65).astype(int)

        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


# --------------------------- Main predictor class --------------------------
class CVDModelPipeline:
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
        df = pd.read_csv(self.data_path)

        # Basic cleaning: strip column names
        df.columns = [c.strip() for c in df.columns]

        # Impute basic missing values conservatively
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = df[col].astype(object)

        # Small, robust imputations will be done in transformers/pipeline
        self.df = df
        return df

    def _choose_stratify_col(self, df: pd.DataFrame) -> Optional[pd.Series]:
        # Prefer 'Age' buckets for stratification; fallback to categorical if present
        if 'Age' in df.columns:
            return pd.cut(df['Age'].fillna(df['Age'].median()), bins=[0,40,55,65,200], labels=[0,1,2,3])
        for c in ['Smoking Status', 'Family History of CVD', 'Physical Activity Level']:
            if c in df.columns:
                return df[c].fillna('M')
        return None

    def initial_split(self, test_size=0.2, random_state=RANDOM_STATE):
        df = self.df.copy()
        strat = self._choose_stratify_col(df)
        if strat is None:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        else:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=strat)
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def prepare_and_train(self, test_size=0.2, random_state=RANDOM_STATE):
        df = self.load_and_clean()
        train_df, test_df = self.initial_split(test_size=test_size, random_state=random_state)

        # Fit risk scorer on train only
        self.risk_scorer.fit(train_df)
        train_df, y_train_labels = self.risk_scorer.transform(train_df)
        test_df, y_test_labels = self.risk_scorer.transform(test_df)

        # Encode labels
        label_map = {lab: i for i, lab in enumerate(['LOW', 'INTERMEDIATE', 'HIGH'])}
        y_train = np.array([label_map[l] for l in y_train_labels])
        y_test = np.array([label_map[l] for l in y_test_labels])

        # Feature engineering
        X_train = self.feature_engineer.fit_transform(train_df.drop(columns=[]))
        X_test = self.feature_engineer.transform(test_df.drop(columns=[]))

        # Identify numeric and categorical columns for pipeline
        self.numeric_cols = [c for c in X_train.columns if X_train[c].dtype in ['int64', 'float64']]
        self.cat_cols = [c for c in X_train.columns if X_train[c].dtype == 'object']

        # Build preprocessing
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

        # Base learners
        cat = CatBoostClassifier(verbose=False, random_state=RANDOM_STATE, auto_class_weights='Balanced')
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=RANDOM_STATE)
        lgb = LGBMClassifier(random_state=RANDOM_STATE)

        # Create a stacking ensemble with logistic regression as final estimator
        estimators = [('cat', cat), ('xgb', xgb), ('lgb', lgb)]

        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
            cv=5, n_jobs=-1, passthrough=False
        )

        # Full pipeline with optional resampling (apply SMOTE only if valid)
        def can_apply_smote(y: np.ndarray) -> bool:
            """SMOTE requires at least 2 minority examples and >1 classes."""
            unique, counts = np.unique(y, return_counts=True)
            if len(unique) < 2:
                return False
            if counts.min() <= 1:
                return False
            return True

        if can_apply_smote(y_train):
            pipeline = ImbPipeline(steps=[
                ('pre', preprocessor),
                ('smote', SMOTE(random_state=RANDOM_STATE)),
                ('sel', SelectFromModel(RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE), threshold='median')),
                ('clf', stacking)
            ])
        else:
            pipeline = Pipeline(steps=[
                ('pre', preprocessor),
                ('sel', SelectFromModel(RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE), threshold='median')),
                ('clf', stacking)
            ])

        # Fit pipeline
        print('\nTraining model pipeline...')
        pipeline.fit(X_train, y_train)
        self.final_model = pipeline

        # Evaluate
        X_test_pre = X_test
        y_pred = pipeline.predict(X_test_pre)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f'Balanced Accuracy: {bal_acc:.4f}')
        print(f'Accuracy: {acc:.4f}')
        print(f'F1 (weighted): {f1:.4f}')
        print('\nClassification Report:')
        print(classification_report(y_test, y_pred, target_names=['LOW','INTERMEDIATE','HIGH']))

        self.feature_names_ = self._get_feature_names_from_preprocessor(pipeline.named_steps['pre'])
        return pipeline, (X_train, y_train), (X_test, y_test), {'bal_acc': bal_acc, 'acc': acc, 'f1': f1}

    def _get_feature_names_from_preprocessor(self, preprocessor: ColumnTransformer) -> List[str]:
        names: List[str] = []
        for name, trans, cols in preprocessor.transformers_:
            if name == 'remainder':
                continue
            if hasattr(trans, 'named_steps') and 'onehot' in trans.named_steps:
                o = trans.named_steps['onehot']
                cats = o.categories_
                for i, c in enumerate(cols):
                    for val in cats[i]:
                        names.append(f'{c}__{val}')
            else:
                names.extend(list(cols))
        return names


# ------------------------------ Execution ---------------------------------

def main():
    DATA_PATH = './public/cvd_dataset.csv'  # update if needed
    model = CVDModelPipeline(DATA_PATH)
    pipeline, train_data, test_data, metrics = model.prepare_and_train(test_size=0.2)

    print('\nDone. Metrics:')
    print(metrics)
    return model, pipeline


if __name__ == '__main__':
    main()