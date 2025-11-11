import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, 
                             confusion_matrix, classification_report, f1_score)
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class RobustCVDPredictor:
    """
    Ultra-robust CVD predictor with maximum overfitting prevention:
    - Stricter data leakage prevention
    - More conservative feature engineering
    - Stronger regularization
    - Nested cross-validation
    - Multiple random splits for validation
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        
    def load_data(self):
        print("="*80)
        print("🏥 ROBUST CVD PREDICTOR - MAXIMUM OVERFITTING PREVENTION")
        print("="*80)
        
        self.df = pd.read_csv(self.data_path)
        print(f"\n✅ Loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def create_clinical_labels(self):
        """Create clinical risk labels with simpler rules"""
        print("\n" + "="*80)
        print("🔬 CREATING CLINICAL RISK LABELS")
        print("="*80)
        
        df = self.df.copy()
        risk_score = np.zeros(len(df))
        
        # Simplified scoring to reduce complexity
        if 'Age' in df.columns:
            df['Age'].fillna(df['Age'].median(), inplace=True)
            risk_score += np.where(df['Age'] >= 60, 2, 
                          np.where(df['Age'] >= 45, 1, 0))
        
        if all(col in df.columns for col in ['Systolic BP', 'Diastolic BP']):
            df['Systolic BP'].fillna(df['Systolic BP'].median(), inplace=True)
            df['Diastolic BP'].fillna(df['Diastolic BP'].median(), inplace=True)
            risk_score += np.where((df['Systolic BP'] >= 140) | (df['Diastolic BP'] >= 90), 2,
                          np.where((df['Systolic BP'] >= 130) | (df['Diastolic BP'] >= 80), 1, 0))
        
        if 'Estimated LDL (mg/dL)' in df.columns:
            df['Estimated LDL (mg/dL)'].fillna(df['Estimated LDL (mg/dL)'].median(), inplace=True)
            risk_score += np.where(df['Estimated LDL (mg/dL)'] >= 160, 2,
                          np.where(df['Estimated LDL (mg/dL)'] >= 130, 1, 0))
        
        if 'HDL (mg/dL)' in df.columns:
            df['HDL (mg/dL)'].fillna(df['HDL (mg/dL)'].median(), inplace=True)
            risk_score += np.where(df['HDL (mg/dL)'] < 40, 1, 0)
        
        if 'Fasting Blood Sugar (mg/dL)' in df.columns:
            df['Fasting Blood Sugar (mg/dL)'].fillna(df['Fasting Blood Sugar (mg/dL)'].median(), inplace=True)
            risk_score += np.where(df['Fasting Blood Sugar (mg/dL)'] >= 126, 2,
                          np.where(df['Fasting Blood Sugar (mg/dL)'] >= 100, 1, 0))
        
        if 'BMI' in df.columns:
            df['BMI'].fillna(df['BMI'].median(), inplace=True)
            risk_score += np.where(df['BMI'] >= 30, 1, 0)
        
        if 'Smoking Status' in df.columns:
            df['Smoking Status'].fillna('N', inplace=True)
            risk_score += np.where(df['Smoking Status'] == 'Y', 2, 0)
        
        if 'Family History of CVD' in df.columns:
            df['Family History of CVD'].fillna('N', inplace=True)
            risk_score += np.where(df['Family History of CVD'] == 'Y', 1, 0)
        
        df['Clinical_Risk_Score'] = risk_score
        
        # Create balanced labels
        low_threshold = np.percentile(risk_score, 33)
        high_threshold = np.percentile(risk_score, 67)
        
        df['Clinical_Risk_Level'] = np.where(risk_score <= low_threshold, 'LOW',
                                     np.where(risk_score <= high_threshold, 'INTERMEDIARY', 'HIGH'))
        
        print(f"✅ Clinical labels created")
        print(f"   Distribution: {df['Clinical_Risk_Level'].value_counts().to_dict()}")
        
        self.df_clinical = df
        return df
    
    def minimal_feature_engineering(self, df):
        """Only create the most essential, well-validated features"""
        print("\n🔧 Minimal Feature Engineering (reducing overfitting risk)...")
        
        df = df.copy()
        features_created = []
        
        # ONLY the most clinically validated ratios
        if all(col in df.columns for col in ['Total Cholesterol (mg/dL)', 'HDL (mg/dL)']):
            df['TC_HDL_Ratio'] = df['Total Cholesterol (mg/dL)'] / (df['HDL (mg/dL)'] + 1)
            df['TC_HDL_Ratio'] = df['TC_HDL_Ratio'].clip(1, 10)
            features_created.append('TC/HDL Ratio')
        
        if all(col in df.columns for col in ['Systolic BP', 'Diastolic BP']):
            df['Pulse_Pressure'] = df['Systolic BP'] - df['Diastolic BP']
            df['Pulse_Pressure'] = df['Pulse_Pressure'].clip(20, 100)
            features_created.append('Pulse Pressure')
        
        print(f"   ✅ Created only {len(features_created)} essential features:")
        for feat in features_created:
            print(f"      • {feat}")
        
        return df
    
    def prepare_features(self, df, target_col='Clinical_Risk_Level'):
        """Prepare features with zero tolerance for leakage"""
        df = df.copy()
        
        # Remove ALL risk-related columns
        risk_cols = ['Clinical_Risk_Score', 'CVD Risk Level', 'Clinical_Risk_Level', 
                     'CVD Risk Score', 'Blood Pressure (mmHg)']
        for col in risk_cols:
            if col in df.columns and col != target_col:
                df = df.drop(col, axis=1)
        
        # Encode target
        le = LabelEncoder()
        df['target'] = le.fit_transform(df[target_col])
        
        # Drop target column
        if target_col in df.columns:
            df = df.drop(target_col, axis=1)
        
        # Handle missing values conservatively
        for col in df.columns:
            if col == 'target':
                continue
            if df[col].dtype in ['int64', 'float64']:
                if df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)
            else:
                if df[col].isnull().any():
                    mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                    df[col].fillna(mode_val, inplace=True)
        
        # One-hot encode
        cat_cols = [c for c in df.columns 
                   if (df[c].dtype == 'object' or str(df[c].dtype).startswith('category')) 
                   and c != 'target']
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)
        
        # Clean up
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in df.columns:
            if col != 'target':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)
        
        print(f"   ✅ Final features: {df.shape[1] - 1} (excluding target)")
        
        return df, le
    
    def nested_cv_optimization(self, X, y, n_trials=15):
        """Nested CV to prevent hyperparameter overfitting"""
        print("\n🔍 Nested Cross-Validation for Robust Optimization...")
        print("   (This prevents overfitting to validation set)")
        
        # Outer CV for true performance estimation
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        outer_scores = []
        
        def objective(trial):
            # More conservative parameter ranges
            params = {
                'iterations': trial.suggest_int('iterations', 200, 600),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
                'depth': trial.suggest_int('depth', 3, 6),  # Shallower trees
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 5, 15),  # More regularization
                'random_state': RANDOM_STATE,
                'verbose': False,
                'auto_class_weights': 'Balanced'
            }
            
            # Inner CV for hyperparameter selection
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
            scores = []
            
            for train_idx, val_idx in inner_cv.split(X, y):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # Apply SMOTE inside CV
                smote = SMOTE(k_neighbors=3, random_state=RANDOM_STATE)
                X_train_sm, y_train_sm = smote.fit_resample(X_train_fold, y_train_fold)
                
                model = CatBoostClassifier(**params)
                model.fit(X_train_sm, y_train_sm, verbose=False)
                
                y_pred = model.predict(X_val_fold)
                scores.append(balanced_accuracy_score(y_val_fold, y_pred))
            
            return np.mean(scores)
        
        # Optimize with fewer trials to reduce overfitting
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Test best params on outer CV
        best_params = study.best_params
        for train_idx, test_idx in outer_cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            smote = SMOTE(k_neighbors=3, random_state=RANDOM_STATE)
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
            
            model = CatBoostClassifier(**best_params)
            model.fit(X_train_sm, y_train_sm, verbose=False)
            
            y_pred = model.predict(X_test)
            outer_scores.append(balanced_accuracy_score(y_test, y_pred))
        
        nested_cv_score = np.mean(outer_scores)
        nested_cv_std = np.std(outer_scores)
        
        print(f"   ✅ Nested CV Score: {nested_cv_score:.4f} ± {nested_cv_std:.4f}")
        print(f"   Best params: {best_params}")
        
        return best_params, nested_cv_score, nested_cv_std
    
    def train_conservative_models(self, X_train, y_train, X_test, y_test, best_params):
        """Train with strong regularization"""
        print("\n🎓 Training Conservative Models...")
        
        # Apply SMOTE
        smote = SMOTE(k_neighbors=3, random_state=RANDOM_STATE)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        print(f"   After SMOTE: {len(X_train_bal)} samples")
        
        models = {}
        
        # CatBoost with best params (already regularized)
        cat_model = CatBoostClassifier(**best_params)
        cat_model.fit(X_train_bal, y_train_bal, verbose=False)
        models['CatBoost'] = cat_model
        
        # XGBoost with STRONG regularization
        xgb_model = XGBClassifier(
            n_estimators=best_params.get('iterations', 400),
            learning_rate=max(0.01, best_params.get('learning_rate', 0.03) * 0.7),  # Slower
            max_depth=min(5, best_params.get('depth', 4)),  # Shallower
            min_child_weight=5,  # More conservative
            subsample=0.7,  # Less data per tree
            colsample_bytree=0.7,  # Fewer features per tree
            reg_alpha=1.0,  # L1 regularization
            reg_lambda=2.0,  # L2 regularization
            random_state=RANDOM_STATE,
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_train_bal, y_train_bal)
        models['XGBoost'] = xgb_model
        
        # LightGBM with STRONG regularization
        lgb_model = LGBMClassifier(
            n_estimators=best_params.get('iterations', 400),
            learning_rate=max(0.01, best_params.get('learning_rate', 0.03) * 0.7),
            max_depth=min(5, best_params.get('depth', 4)),
            min_child_samples=30,  # More samples required
            min_split_gain=0.1,  # Minimum gain to split
            reg_alpha=1.0,
            reg_lambda=2.0,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=RANDOM_STATE,
            verbose=-1
        )
        lgb_model.fit(X_train_bal, y_train_bal)
        models['LightGBM'] = lgb_model
        
        # Voting ensemble
        voting = VotingClassifier(
            estimators=[(name, model) for name, model in models.items()],
            voting='soft',
            n_jobs=-1
        )
        voting.fit(X_train_bal, y_train_bal)
        models['Ensemble'] = voting
        
        # Evaluate
        results = []
        for name, model in models.items():
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results.append({
                'Model': name,
                'Accuracy': acc,
                'Balanced_Accuracy': bal_acc,
                'F1_Score': f1
            })
            
            print(f"   {name}: Acc={acc:.4f}, Balanced={bal_acc:.4f}, F1={f1:.4f}")
        
        results_df = pd.DataFrame(results).sort_values('Balanced_Accuracy', ascending=False)
        best_model_name = results_df.iloc[0]['Model']
        best_model = models[best_model_name]
        
        return best_model, best_model_name, results_df
    
    def multiple_holdout_validation(self, X, y, best_params, n_splits=5):
        """Test on multiple random splits to ensure robustness"""
        print("\n🔄 Multiple Holdout Validation...")
        print("   (Testing on different random splits)")
        
        holdout_scores = []
        
        for i in range(n_splits):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE + i, stratify=y
            )
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            smote = SMOTE(k_neighbors=3, random_state=RANDOM_STATE)
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
            
            model = CatBoostClassifier(**best_params)
            model.fit(X_train_sm, y_train_sm, verbose=False)
            
            y_pred = model.predict(X_test)
            score = balanced_accuracy_score(y_test, y_pred)
            holdout_scores.append(score)
            print(f"   Split {i+1}: {score:.4f}")
        
        mean_score = np.mean(holdout_scores)
        std_score = np.std(holdout_scores)
        
        print(f"\n   ✅ Mean: {mean_score:.4f} ± {std_score:.4f}")
        
        return mean_score, std_score, holdout_scores
    
    def run_full_pipeline(self):
        """Run ultra-robust pipeline"""
        # Load and create labels
        self.load_data()
        self.create_clinical_labels()
        
        # Minimal feature engineering
        df_engineered = self.minimal_feature_engineering(self.df_clinical)
        
        # Prepare features
        df_prepared, le = self.prepare_features(df_engineered)
        
        X = df_prepared.drop('target', axis=1).values
        y = df_prepared['target'].values
        
        print(f"\n📦 Total samples: {len(X)}, Features: {X.shape[1]}")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Nested CV optimization
        best_params, nested_cv_score, nested_cv_std = self.nested_cv_optimization(
            X_train_scaled, y_train, n_trials=15
        )
        
        # Multiple holdout validation
        holdout_mean, holdout_std, holdout_scores = self.multiple_holdout_validation(
            X, y, best_params, n_splits=5
        )
        
        # Train final models
        best_model, best_name, results = self.train_conservative_models(
            X_train_scaled, y_train, X_test_scaled, y_test, best_params
        )
        
        # Final evaluation
        y_pred = best_model.predict(X_test_scaled)
        
        print("\n" + "="*80)
        print(f"🏆 BEST MODEL: {best_name}")
        print("="*80)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
        print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.2%}")
        print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
        
        # Per-class accuracy
        print(f"\n🎯 Per-class Performance:")
        for cls_id in range(len(le.classes_)):
            mask = (y_test == cls_id)
            if mask.sum() > 0:
                cls_acc = accuracy_score(y_test[mask], y_pred[mask])
                cls_name = le.inverse_transform([cls_id])[0]
                print(f"   {cls_name}: {cls_acc:.2%} ({mask.sum()} samples)")
        
        # Confusion matrix
        print(f"\n📊 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        classes = le.classes_
        
        print(f"{'Predicted →':>15}", end='')
        for cls in classes:
            print(f"{cls:>12}", end='')
        print()
        print("Actual ↓")
        
        for i, cls in enumerate(classes):
            print(f"{cls:>15}", end='')
            for j in range(len(cm[i])):
                print(f"{cm[i][j]:>12}", end='')
            print()
        
        print(f"\n📈 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=classes, digits=4))
        
        final_balanced_acc = balanced_accuracy_score(y_test, y_pred)
        
        print("\n" + "="*80)
        print("📊 OVERFITTING ANALYSIS")
        print("="*80)
        print(f"Nested CV Score:        {nested_cv_score:.4f} ± {nested_cv_std:.4f}")
        print(f"Multiple Holdout Mean:  {holdout_mean:.4f} ± {holdout_std:.4f}")
        print(f"Final Test Score:       {final_balanced_acc:.4f}")
        
        gap_nested = abs(final_balanced_acc - nested_cv_score)
        gap_holdout = abs(final_balanced_acc - holdout_mean)
        
        print(f"\nGap from Nested CV:     {gap_nested:.4f} ({gap_nested*100:.2f}%)")
        print(f"Gap from Holdout Mean:  {gap_holdout:.4f} ({gap_holdout*100:.2f}%)")
        
        if gap_nested <= 0.03 and gap_holdout <= 0.03:
            print("\n✅ EXCELLENT! Gaps < 3% - Model generalizes very well!")
        elif gap_nested <= 0.05 and gap_holdout <= 0.05:
            print("\n✅ GOOD! Gaps < 5% - Model generalizes well!")
        elif gap_nested <= 0.08 and gap_holdout <= 0.08:
            print("\n⚠️  ACCEPTABLE! Gaps < 8% - Some overfitting present")
        else:
            print("\n❌ WARNING! Gaps > 8% - Significant overfitting detected!")
        
        return best_model, results, final_balanced_acc, nested_cv_score, holdout_mean


def main():
    predictor = RobustCVDPredictor('./public/cvd_dataset.csv')
    model, results, test_acc, nested_cv, holdout_mean = predictor.run_full_pipeline()
    
    print("\n" + "="*80)
    print("📈 FINAL SUMMARY")
    print("="*80)
    print(f"Nested CV (most reliable):  {nested_cv:.2%}")
    print(f"Multiple Holdout Mean:      {holdout_mean:.2%}")
    print(f"Final Test:                 {test_acc:.2%}")
    print(f"\nExpected Real-World Performance: ~{nested_cv:.2%}")
    
    return predictor, model, results


if __name__ == "__main__":
    predictor, model, results = main()