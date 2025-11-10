import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, 
                             confusion_matrix, classification_report, f1_score)
from sklearn.feature_selection import SelectFromModel
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
import optuna
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class RealisticCVDPredictor:
    """
    Realistic CVD predictor avoiding overfitting with:
    - Proper data leakage prevention
    - Conservative feature engineering
    - Regularized models
    - Realistic validation
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        
    def load_data(self):
        print("="*80)
        print("🏥 REALISTIC CVD PREDICTOR - AVOIDING OVERFITTING")
        print("="*80)
        
        self.df = pd.read_csv(self.data_path)
        print(f"\n✅ Loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def create_clinical_labels(self):
        """Create clinical risk labels"""
        print("\n" + "="*80)
        print("🔬 CREATING CLINICAL RISK LABELS")
        print("="*80)
        
        df = self.df.copy()
        risk_score = np.zeros(len(df))
        
        # Age risk
        if 'Age' in df.columns:
            df['Age'].fillna(df['Age'].median(), inplace=True)
            risk_score += np.where(df['Age'] >= 65, 3, 
                          np.where(df['Age'] >= 55, 2,
                          np.where(df['Age'] >= 45, 1, 0)))
        
        # Blood Pressure
        if all(col in df.columns for col in ['Systolic BP', 'Diastolic BP']):
            df['Systolic BP'].fillna(df['Systolic BP'].median(), inplace=True)
            df['Diastolic BP'].fillna(df['Diastolic BP'].median(), inplace=True)
            risk_score += np.where((df['Systolic BP'] >= 140) | (df['Diastolic BP'] >= 90), 3,
                          np.where((df['Systolic BP'] >= 130) | (df['Diastolic BP'] >= 80), 2,
                          np.where(df['Systolic BP'] >= 120, 1, 0)))
        
        # LDL Cholesterol
        if 'Estimated LDL (mg/dL)' in df.columns:
            df['Estimated LDL (mg/dL)'].fillna(df['Estimated LDL (mg/dL)'].median(), inplace=True)
            risk_score += np.where(df['Estimated LDL (mg/dL)'] >= 190, 3,
                          np.where(df['Estimated LDL (mg/dL)'] >= 160, 2,
                          np.where(df['Estimated LDL (mg/dL)'] >= 130, 1, 0)))
        
        # HDL (protective)
        if 'HDL (mg/dL)' in df.columns:
            df['HDL (mg/dL)'].fillna(df['HDL (mg/dL)'].median(), inplace=True)
            risk_score += np.where(df['HDL (mg/dL)'] < 40, 2,
                          np.where(df['HDL (mg/dL)'] < 50, 1, 0))
            risk_score -= np.where(df['HDL (mg/dL)'] >= 60, 1, 0)
        
        # Blood Sugar
        if 'Fasting Blood Sugar (mg/dL)' in df.columns:
            df['Fasting Blood Sugar (mg/dL)'].fillna(df['Fasting Blood Sugar (mg/dL)'].median(), inplace=True)
            risk_score += np.where(df['Fasting Blood Sugar (mg/dL)'] >= 126, 3,
                          np.where(df['Fasting Blood Sugar (mg/dL)'] >= 100, 1, 0))
        
        # BMI
        if 'BMI' in df.columns:
            df['BMI'].fillna(df['BMI'].median(), inplace=True)
            risk_score += np.where(df['BMI'] >= 35, 2,
                          np.where(df['BMI'] >= 30, 1, 0))
        
        # Smoking
        if 'Smoking Status' in df.columns:
            df['Smoking Status'].fillna('N', inplace=True)
            risk_score += np.where(df['Smoking Status'] == 'Y', 3, 0)
        
        # Family History
        if 'Family History of CVD' in df.columns:
            df['Family History of CVD'].fillna('N', inplace=True)
            risk_score += np.where(df['Family History of CVD'] == 'Y', 2, 0)
        
        # Physical Activity
        if 'Physical Activity Level' in df.columns:
            df['Physical Activity Level'].fillna('Moderate', inplace=True)
            risk_score -= np.where(df['Physical Activity Level'] == 'High', 1, 0)
            risk_score += np.where(df['Physical Activity Level'] == 'Low', 1, 0)
        
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
    
    def conservative_feature_engineering(self, df):
        """Create only medically validated features to avoid overfitting"""
        print("\n🔧 Conservative Feature Engineering...")
        
        df = df.copy()
        features_created = []
        
        # Only create well-established medical ratios
        if all(col in df.columns for col in ['Total Cholesterol (mg/dL)', 'HDL (mg/dL)']):
            df['TC_HDL_Ratio'] = df['Total Cholesterol (mg/dL)'] / (df['HDL (mg/dL)'] + 1)
            df['TC_HDL_Ratio'] = df['TC_HDL_Ratio'].clip(1, 10)
            features_created.append('TC/HDL Ratio')
        
        if all(col in df.columns for col in ['Estimated LDL (mg/dL)', 'HDL (mg/dL)']):
            df['LDL_HDL_Ratio'] = df['Estimated LDL (mg/dL)'] / (df['HDL (mg/dL)'] + 1)
            df['LDL_HDL_Ratio'] = df['LDL_HDL_Ratio'].clip(0.5, 8)
            features_created.append('LDL/HDL Ratio')
        
        if all(col in df.columns for col in ['Total Cholesterol (mg/dL)', 'HDL (mg/dL)']):
            df['Non_HDL_Chol'] = df['Total Cholesterol (mg/dL)'] - df['HDL (mg/dL)']
            features_created.append('Non-HDL Cholesterol')
        
        # Pulse pressure (validated CVD marker)
        if all(col in df.columns for col in ['Systolic BP', 'Diastolic BP']):
            df['Pulse_Pressure'] = df['Systolic BP'] - df['Diastolic BP']
            df['Pulse_Pressure'] = df['Pulse_Pressure'].clip(20, 100)
            features_created.append('Pulse Pressure')
        
        # Mean Arterial Pressure
        if all(col in df.columns for col in ['Systolic BP', 'Diastolic BP']):
            df['MAP'] = df['Diastolic BP'] + ((df['Systolic BP'] - df['Diastolic BP']) / 3)
            features_created.append('MAP')
        
        print(f"   ✅ Created {len(features_created)} validated features:")
        for feat in features_created:
            print(f"      • {feat}")
        
        return df
    
    def prepare_features(self, df, target_col='Clinical_Risk_Level'):
        """Prepare features with NO data leakage"""
        df = df.copy()
        
        # CRITICAL: Remove the clinical risk score to prevent leakage!
        if 'Clinical_Risk_Score' in df.columns:
            print("   ⚠️  Dropping Clinical_Risk_Score to prevent data leakage")
            df = df.drop('Clinical_Risk_Score', axis=1)
        
        # Encode target
        le = LabelEncoder()
        df['target'] = le.fit_transform(df[target_col])
        
        # Drop all label-related columns
        drop_cols = ['CVD Risk Level', 'Clinical_Risk_Level', 'CVD Risk Score', 
                     'Blood Pressure (mmHg)', target_col]
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(col, axis=1)
        
        # Handle missing values
        for col in df.columns:
            if col == 'target':
                continue
            if df[col].dtype in ['int64', 'float64']:
                if df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)
            else:
                if df[col].isnull().any():
                    df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
        
        # One-hot encode categorical
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
    
    def optimize_with_cv(self, X, y, n_trials=30):
        """Optimize using proper cross-validation"""
        print("\n🔍 Hyperparameter Optimization with CV...")
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 300, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'depth': trial.suggest_int('depth', 4, 8),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3, 10),
                'random_state': RANDOM_STATE,
                'verbose': False,
                'auto_class_weights': 'Balanced'
            }
            
            # Use cross-validation for unbiased evaluation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            scores = []
            
            for train_idx, val_idx in cv.split(X, y):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                model = CatBoostClassifier(**params)
                model.fit(X_train_fold, y_train_fold, verbose=False)
                
                y_pred = model.predict(X_val_fold)
                scores.append(balanced_accuracy_score(y_val_fold, y_pred))
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"   ✅ Best CV score: {study.best_value:.4f}")
        print(f"   Best params: {study.best_params}")
        
        return study.best_params, study.best_value
    
    def train_final_models(self, X_train, y_train, X_test, y_test, best_params):
        """Train final ensemble with regularization"""
        print("\n🎓 Training Final Models...")
        
        # Apply SMOTE only to training data
        smote = SMOTE(k_neighbors=3, random_state=RANDOM_STATE)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        
        print(f"   After SMOTE: {len(X_train_bal)} samples")
        
        # Train regularized models
        models = {}
        
        # CatBoost with optimized params
        cat_model = CatBoostClassifier(**best_params)
        cat_model.fit(X_train_bal, y_train_bal, verbose=False)
        models['CatBoost'] = cat_model
        
        # XGBoost with regularization
        xgb_model = XGBClassifier(
            n_estimators=best_params.get('iterations', 500),
            learning_rate=best_params.get('learning_rate', 0.05),
            max_depth=best_params.get('depth', 6),
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_train_bal, y_train_bal)
        models['XGBoost'] = xgb_model
        
        # LightGBM with regularization
        lgb_model = LGBMClassifier(
            n_estimators=best_params.get('iterations', 500),
            learning_rate=best_params.get('learning_rate', 0.05),
            max_depth=best_params.get('depth', 6),
            min_child_samples=20,
            reg_alpha=0.5,
            reg_lambda=1.0,
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
        
        # Evaluate all models
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
    
    def run_full_pipeline(self):
        """Run realistic pipeline"""
        # Load and create labels
        self.load_data()
        self.create_clinical_labels()
        
        # Conservative feature engineering
        df_engineered = self.conservative_feature_engineering(self.df_clinical)
        
        # Prepare features (no leakage!)
        df_prepared, le = self.prepare_features(df_engineered)
        
        X = df_prepared.drop('target', axis=1).values
        y = df_prepared['target'].values
        
        print(f"\n📦 Total samples: {len(X)}, Features: {X.shape[1]}")
        
        # Split ONCE - no feature selection on test set!
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Scale (fit on train only!)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Optimize hyperparameters
        best_params, cv_score = self.optimize_with_cv(X_train_scaled, y_train, n_trials=20)
        
        print(f"\n📊 Expected performance from CV: {cv_score:.2%}")
        
        # Train final models
        best_model, best_name, results = self.train_final_models(
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
        print("📊 RESULTS INTERPRETATION")
        print("="*80)
        
        if final_balanced_acc >= 0.90:
            print("🎉 EXCELLENT! 90%+ balanced accuracy!")
        elif final_balanced_acc >= 0.85:
            print("✅ VERY GOOD! 85%+ balanced accuracy!")
        elif final_balanced_acc >= 0.80:
            print("✅ GOOD! 80%+ balanced accuracy!")
        elif final_balanced_acc >= 0.75:
            print("✅ ACCEPTABLE! 75%+ balanced accuracy!")
        else:
            print(f"⚠️  Current: {final_balanced_acc:.2%}")
        
        if final_balanced_acc > cv_score + 0.05:
            print("\n⚠️  WARNING: Test performance >> CV performance")
            print("   This suggests possible overfitting or lucky split")
        elif final_balanced_acc < cv_score - 0.05:
            print("\n⚠️  Test performance << CV performance")
            print("   This is normal - test set is unseen data")
        else:
            print("\n✅ Test performance matches CV expectations")
            print("   Model generalizes well!")
        
        return best_model, results, final_balanced_acc, cv_score


def main():
    predictor = RealisticCVDPredictor('./public/cvd_dataset.csv')
    model, results, test_acc, cv_acc = predictor.run_full_pipeline()
    
    print("\n" + "="*80)
    print("📈 SUMMARY")
    print("="*80)
    print(f"CV Balanced Accuracy:   {cv_acc:.2%}")
    print(f"Test Balanced Accuracy: {test_acc:.2%}")
    print(f"Difference: {abs(test_acc - cv_acc):.2%}")
    
    return predictor, model, results


if __name__ == "__main__":
    predictor, model, results = main()