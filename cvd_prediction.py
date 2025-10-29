import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, VotingClassifier,
                              StackingClassifier, ExtraTreesClassifier)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class EnhancedCVDPredictor:
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = None
        self.pca = None
        self.models = {}
        self.results = {}
        self.feature_names = None
        
    def load_data(self):
        print("Loading dataset...")
        self.df = pd.read_csv("./public/cvd_dataset.csv")
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print(f"\nTarget distribution:\n{self.df['CVD Risk Level'].value_counts()}")
        return self.df
    
    def advanced_preprocessing(self):
        print("\n" + "="*60)
        print("ADVANCED DATA PREPROCESSING")
        print("="*60)
        
        df = self.df.copy()
        
        print("\nPhase 1: Advanced Missing Value Imputation...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if 'CVD Risk Level' in categorical_cols:
            categorical_cols.remove('CVD Risk Level')
        for col in ['CVD Risk Score', 'CVD Risk Level']:
            if col in numeric_cols:
                numeric_cols.remove(col)
        
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['CVD_Risk_Encoded'] = le.fit_transform(df['CVD Risk Level'])
        self.label_encoder = le
        
        knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
        df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])
        
        cat_imputer = SimpleImputer(strategy='most_frequent')
        if len(categorical_cols) > 0:
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
        
        print(f"Missing values handled: {df.isnull().sum().sum()} remaining")
        
        print("\nPhase 2: Advanced Feature Engineering...")
        
        if 'Age' in df.columns and 'BMI' in df.columns:
            df['Age_Squared'] = df['Age'] ** 2
            df['BMI_Squared'] = df['BMI'] ** 2
            df['Age_BMI'] = df['Age'] * df['BMI']
        
        if all(col in df.columns for col in ['Total Cholesterol (mg/dL)', 'HDL (mg/dL)']):
            df['Total_HDL_Ratio'] = df['Total Cholesterol (mg/dL)'] / (df['HDL (mg/dL)'] + 1)
            df['Total_HDL_Ratio'] = df['Total_HDL_Ratio'].clip(upper=15)
        
        if all(col in df.columns for col in ['Estimated LDL (mg/dL)', 'HDL (mg/dL)']):
            df['LDL_HDL_Ratio'] = df['Estimated LDL (mg/dL)'] / (df['HDL (mg/dL)'] + 1)
            df['LDL_HDL_Ratio'] = df['LDL_HDL_Ratio'].clip(lower=-5, upper=15)
        
        if all(col in df.columns for col in ['Systolic BP', 'Diastolic BP']):
            df['Pulse_Pressure'] = df['Systolic BP'] - df['Diastolic BP']
            df['Mean_Arterial_Pressure'] = df['Diastolic BP'] + (df['Pulse_Pressure'] / 3)
            df['BP_Product'] = df['Systolic BP'] * df['Diastolic BP']
        
        if all(col in df.columns for col in ['Weight (kg)', 'Height (m)']):
            df['Weight_Height_Ratio'] = df['Weight (kg)'] / (df['Height (m)'] ** 2)
        
        if 'Waist-to-Height Ratio' in df.columns and 'BMI' in df.columns:
            df['Waist_BMI_Interaction'] = df['Waist-to-Height Ratio'] * df['BMI']
        
        if 'Age' in df.columns:
            df['Age_Group_Young'] = (df['Age'] < 35).astype(int)
            df['Age_Group_Middle'] = ((df['Age'] >= 35) & (df['Age'] < 55)).astype(int)
            df['Age_Group_Senior'] = (df['Age'] >= 55).astype(int)
        
        if all(col in df.columns for col in ['BMI', 'Fasting Blood Sugar (mg/dL)', 'Systolic BP']):
            df['Metabolic_Risk_Score'] = (
                (df['BMI'] > 30).astype(int) +
                (df['Fasting Blood Sugar (mg/dL)'] > 126).astype(int) +
                (df['Systolic BP'] > 140).astype(int)
            )
        
        print(f"Created advanced engineered features")
        
        print("\nPhase 3: Encoding Categorical Variables...")
        categorical_features = ['Sex', 'Smoking Status', 'Diabetes Status', 
                                'Physical Activity Level', 'Family History of CVD',
                                'Blood Pressure Category']
        
        for col in categorical_features:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
                df = pd.concat([df, dummies], axis=1)
                df.drop(col, axis=1, inplace=True)
        
        print("\nPhase 4: Robust Outlier Handling...")
        numeric_cols_current = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'CVD_Risk_Encoded' in numeric_cols_current:
            numeric_cols_current.remove('CVD_Risk_Encoded')
        if 'CVD Risk Score' in numeric_cols_current:
            numeric_cols_current.remove('CVD Risk Score')
        
        for col in numeric_cols_current:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 2.5 * IQR
            upper_bound = Q3 + 2.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        print("\nPhase 5: Handling Extreme Values...")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        for col in numeric_cols_current:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val if not pd.isna(median_val) else 0, inplace=True)
        
        features_to_drop = ['CVD Risk Level', 'CVD Risk Score', 'Blood Pressure (mmHg)']
        df = df.drop([col for col in features_to_drop if col in df.columns], axis=1)
        
        print("\nPhase 6: Final Type Conversion...")
        for col in df.columns:
            if col != 'CVD_Risk_Encoded':
                if df[col].dtype == 'bool':
                    df[col] = df[col].astype(int)
                elif df[col].dtype == 'object':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if df[col].isnull().any():
                        df[col].fillna(df[col].median(), inplace=True)
        
        self.df_processed = df
        print(f"\nFinal dataset shape: {df.shape}")
        print(f"Number of features: {df.shape[1] - 1}")
        
        return df
    
    def split_and_balance_data(self, test_size=0.15, val_size=0.15, use_smote=True):
        print("\n" + "="*60)
        print("DATA SPLITTING & BALANCING")
        print("="*60)
        
        df = self.df_processed.copy()
        X = df.drop('CVD_Risk_Encoded', axis=1)
        y = df['CVD_Risk_Encoded']
        
        self.feature_names = X.columns.tolist()
        
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=RANDOM_STATE, stratify=y_temp
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Validation set: {self.X_val.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        print("\nOriginal class distribution in training:")
        print(pd.Series(self.y_train).value_counts().sort_index())
        
        if use_smote:
            print("\nApplying SMOTE for class balancing...")
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
            self.X_train_balanced, self.y_train_balanced = smote.fit_resample(self.X_train, self.y_train)
            
            print("\nBalanced class distribution:")
            print(pd.Series(self.y_train_balanced).value_counts().sort_index())
            print(f"Training samples after SMOTE: {self.X_train_balanced.shape[0]}")
        else:
            self.X_train_balanced = self.X_train
            self.y_train_balanced = self.y_train
    
    def advanced_feature_scaling(self, use_power_transform=True):
        print("\n" + "="*60)
        print("ADVANCED FEATURE SCALING")
        print("="*60)
        
        if use_power_transform:
            print("Using PowerTransformer (Yeo-Johnson) for normalization...")
            self.scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        else:
            print("Using RobustScaler...")
            self.scaler = RobustScaler()
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train_balanced)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Scaled training data shape: {self.X_train_scaled.shape}")
        print("Features scaled successfully")
    
    def feature_selection(self, method='combined', n_features=30):
        print("\n" + "="*60)
        print("FEATURE SELECTION")
        print("="*60)
        
        if method == 'combined':
            print(f"Selecting top {n_features} features using combined approach...")
            
            selector1 = SelectKBest(score_func=f_classif, k=min(n_features, self.X_train_scaled.shape[1]))
            selector1.fit(self.X_train_scaled, self.y_train_balanced)
            scores1 = selector1.scores_
            
            selector2 = SelectKBest(score_func=mutual_info_classif, k=min(n_features, self.X_train_scaled.shape[1]))
            selector2.fit(self.X_train_scaled, self.y_train_balanced)
            scores2 = selector2.scores_
            
            combined_scores = (scores1 + scores2) / 2
            top_indices = np.argsort(combined_scores)[-n_features:]
            
            self.X_train_selected = self.X_train_scaled[:, top_indices]
            self.X_val_selected = self.X_val_scaled[:, top_indices]
            self.X_test_selected = self.X_test_scaled[:, top_indices]
            
            self.selected_features = [self.feature_names[i] for i in top_indices]
            
            print(f"Selected {len(self.selected_features)} features")
            print(f"\nTop 15 selected features:")
            feature_scores = sorted(zip(self.selected_features, combined_scores[top_indices]), 
                                   key=lambda x: x[1], reverse=True)
            for feat, score in feature_scores[:15]:
                print(f"  {feat}: {score:.4f}")
        
        elif method == 'pca':
            print(f"Applying PCA (n_components={n_features})...")
            self.pca = PCA(n_components=min(n_features, self.X_train_scaled.shape[1]), random_state=RANDOM_STATE)
            self.X_train_selected = self.pca.fit_transform(self.X_train_scaled)
            self.X_val_selected = self.pca.transform(self.X_val_scaled)
            self.X_test_selected = self.pca.transform(self.X_test_scaled)
            
            explained_var = np.sum(self.pca.explained_variance_ratio_)
            print(f"Explained variance: {explained_var:.4f}")
        
        else:
            print("Skipping feature selection...")
            self.X_train_selected = self.X_train_scaled
            self.X_val_selected = self.X_val_scaled
            self.X_test_selected = self.X_test_scaled
        
        return self.X_train_selected
    
    def define_advanced_models(self):
        print("\n" + "="*60)
        print("DEFINING ADVANCED MODELS")
        print("="*60)
        
        self.model_configs = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'saga'],
                    'class_weight': ['balanced', None]
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=RANDOM_STATE),
                'params': {
                    'n_estimators': [200, 300, 500],
                    'max_depth': [15, 20, 25, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'max_features': ['sqrt', 'log2'],
                    'class_weight': ['balanced', 'balanced_subsample']
                }
            },
            'Extra Trees': {
                'model': ExtraTreesClassifier(random_state=RANDOM_STATE),
                'params': {
                    'n_estimators': [200, 300, 500],
                    'max_depth': [15, 20, 25],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'class_weight': ['balanced', 'balanced_subsample']
                }
            },
            'XGBoost': {
                'model': XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss'),
                'params': {
                    'n_estimators': [200, 300, 500],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [4, 6, 8],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'gamma': [0, 0.1, 0.2],
                    'reg_alpha': [0, 0.1, 0.5],
                    'reg_lambda': [1, 1.5, 2]
                }
            },
            'LightGBM': {
                'model': LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
                'params': {
                    'n_estimators': [200, 300, 500],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [4, 6, 8, -1],
                    'num_leaves': [31, 50, 70],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.1],
                    'reg_lambda': [0, 0.1],
                    'class_weight': ['balanced', None]
                }
            },
            'CatBoost': {
                'model': CatBoostClassifier(random_state=RANDOM_STATE, verbose=False),
                'params': {
                    'iterations': [200, 300, 500],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'depth': [4, 6, 8, 10],
                    'l2_leaf_reg': [1, 3, 5],
                    'border_count': [32, 64, 128],
                    'auto_class_weights': ['Balanced', 'SqrtBalanced', None]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
                'params': {
                    'n_estimators': [200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [4, 6, 8],
                    'subsample': [0.8, 0.9, 1.0],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'SVM': {
                'model': SVC(probability=True, random_state=RANDOM_STATE),
                'params': {
                    'C': [1, 10, 100],
                    'kernel': ['rbf', 'poly'],
                    'gamma': ['scale', 'auto'],
                    'class_weight': ['balanced', None]
                }
            },
            'MLP': {
                'model': MLPClassifier(random_state=RANDOM_STATE, max_iter=1000, early_stopping=True),
                'params': {
                    'hidden_layer_sizes': [(100, 50), (128, 64, 32), (200, 100)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['adaptive'],
                    'batch_size': [32, 64]
                }
            }
        }
        
        print(f"Defined {len(self.model_configs)} advanced models")
    
    def train_with_randomized_search(self, n_iter=50, cv_folds=5):
        print("\n" + "="*60)
        print("MODEL TRAINING (Randomized Search)")
        print("="*60)
        
        X_tr = self.X_train_selected
        X_v = self.X_val_selected
        
        for name, config in self.model_configs.items():
            print(f"\nTraining {name}...")
            
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
            
            random_search = RandomizedSearchCV(
                config['model'],
                config['params'],
                n_iter=n_iter,
                cv=cv,
                scoring='f1_weighted',
                n_jobs=None,
                verbose=0,
                random_state=RANDOM_STATE
            )
            
            random_search.fit(X_tr, self.y_train_balanced)
            
            best_model = random_search.best_estimator_
            print(f"Best params: {random_search.best_params_}")
            print(f"Best CV score: {random_search.best_score_:.4f}")
            
            self.models[name] = best_model
            
            y_val_pred = best_model.predict(X_v)
            val_acc = accuracy_score(self.y_val, y_val_pred)
            val_f1 = f1_score(self.y_val, y_val_pred, average='weighted')
            
            print(f"Validation Accuracy: {val_acc:.4f}")
            print(f"Validation F1-Score: {val_f1:.4f}")
        
        print(f"\nAll {len(self.models)} models trained successfully")
    
    def evaluate_models(self):
        print("\n" + "="*60)
        print("MODEL EVALUATION ON TEST SET")
        print("="*60)
        
        results_list = []
        X_test = self.X_test_selected
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            
            try:
                roc_auc = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                roc_auc = 0.0
            
            results_list.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc
            })
            
            self.results[name] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc
                }
            }
        
        self.results_df = pd.DataFrame(results_list).sort_values('Accuracy', ascending=False)
        
        print("\n" + "="*60)
        print("RESULTS SUMMARY (Sorted by Accuracy)")
        print("="*60)
        print(self.results_df.to_string(index=False))
        
        best_model_name = self.results_df.iloc[0]['Model']
        best_acc = self.results_df.iloc[0]['Accuracy']
        print(f"\nBest Model: {best_model_name} (Accuracy: {best_acc:.4f})")
        
        return self.results_df
    
    def build_meta_ensemble(self):
        print("\n" + "="*60)
        print("BUILDING META-ENSEMBLE")
        print("="*60)
        
        top_models = self.results_df.head(5)['Model'].tolist()
        print(f"\nTop 5 models for ensemble: {top_models}")
        
        estimators = [(name, self.models[name]) for name in top_models if name in self.models]
        
        X_tr = self.X_train_selected
        X_te = self.X_test_selected
        
        print("\n1. Training Weighted Voting Ensemble...")
        weights = []
        for name in top_models:
            if name in self.models:
                val_pred = self.models[name].predict(self.X_val_selected)
                weight = accuracy_score(self.y_val, val_pred)
                weights.append(weight)
        
        voting_weighted = VotingClassifier(estimators=estimators, voting='soft', weights=weights)
        voting_weighted.fit(X_tr, self.y_train_balanced)
        
        y_pred_weighted = voting_weighted.predict(X_te)
        acc_weighted = accuracy_score(self.y_test, y_pred_weighted)
        f1_weighted = f1_score(self.y_test, y_pred_weighted, average='weighted')
        
        print(f"Weighted Voting - Accuracy: {acc_weighted:.4f}, F1: {f1_weighted:.4f}")
        
        print("\n2. Training Stacking Ensemble...")
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=XGBClassifier(random_state=RANDOM_STATE, n_estimators=100),
            cv=5
        )
        stacking.fit(X_tr, self.y_train_balanced)
        
        y_pred_stack = stacking.predict(X_te)
        acc_stack = accuracy_score(self.y_test, y_pred_stack)
        f1_stack = f1_score(self.y_test, y_pred_stack, average='weighted')
        
        print(f"Stacking - Accuracy: {acc_stack:.4f}, F1: {f1_stack:.4f}")
        
        self.ensemble_models = {
            'Weighted_Voting': voting_weighted,
            'Stacking': stacking
        }
        
        ensemble_results = [
            {'Model': 'Weighted Voting Ensemble', 'Accuracy': acc_weighted, 'F1-Score': f1_weighted},
            {'Model': 'Stacking Ensemble', 'Accuracy': acc_stack, 'F1-Score': f1_stack}
        ]
        
        print("\nEnsemble models created")
        
        return pd.DataFrame(ensemble_results)
    
    def plot_comprehensive_results(self):
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        ax1 = axes[0, 0]
        top_models = self.results_df.head(10)
        ax1.barh(top_models['Model'], top_models['Accuracy'], color='skyblue')
        ax1.set_xlabel('Accuracy', fontsize=12)
        ax1.set_ylabel('Model', fontsize=12)
        ax1.set_title('Model Accuracy Comparison (Top 10)', fontsize=14, fontweight='bold')
        ax1.axvline(x=0.8, color='red', linestyle='--', label='80% Target')
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)
        
        ax2 = axes[0, 1]
        metrics_df = self.results_df.head(8).set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score']]
        metrics_df.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Multi-Metric Comparison (Top 8)', fontsize=14, fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        
        ax3 = axes[1, 0]
        best_model_name = self.results_df.iloc[0]['Model']
        y_pred = self.results[best_model_name]['y_pred']
        cm = confusion_matrix(self.y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar=True)
        ax3.set_xlabel('Predicted', fontsize=12)
        ax3.set_ylabel('Actual', fontsize=12)
        ax3.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
        
        ax4 = axes[1, 1]
        roc_df = self.results_df.head(10)[['Model', 'ROC-AUC']].sort_values('ROC-AUC', ascending=True)
        ax4.barh(roc_df['Model'], roc_df['ROC-AUC'], color='lightcoral')
        ax4.set_xlabel('ROC-AUC Score', fontsize=12)
        ax4.set_ylabel('Model', fontsize=12)
        ax4.set_title('ROC-AUC Comparison', fontsize=14, fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('enhanced_cvd_results.png', dpi=300, bbox_inches='tight')
        print("Saved: enhanced_cvd_results.png")
        plt.show()
    
    def detailed_classification_report(self):
        print("\n" + "="*60)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*60)
        
        best_model_name = self.results_df.iloc[0]['Model']
        print(f"\nBest Model: {best_model_name}")
        print("-" * 60)
        
        y_pred = self.results[best_model_name]['y_pred']
        target_names = self.label_encoder.classes_
        
        report = classification_report(self.y_test, y_pred, target_names=target_names)
        print(report)
        
        cm = confusion_matrix(self.y_test, y_pred)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        print("\nPer-Class Accuracy:")
        for i, class_name in enumerate(target_names):
            print(f"  {class_name}: {per_class_acc[i]:.4f}")


def main():
    print("="*60)
    print("ENHANCED CVD RISK PREDICTION SYSTEM")
    print("Target: 80%+ Accuracy")
    print("="*60)
    
    predictor = EnhancedCVDPredictor('./public/cvd_dataset.csv')
    
    predictor.load_data()
    
    predictor.advanced_preprocessing()
    
    predictor.split_and_balance_data(use_smote=True)
    
    predictor.advanced_feature_scaling(use_power_transform=True)
    
    predictor.feature_selection(method='combined', n_features=35)
    
    predictor.define_advanced_models()
    
    predictor.train_with_randomized_search(n_iter=50, cv_folds=5)
    
    results_df = predictor.evaluate_models()
    
    ensemble_df = predictor.build_meta_ensemble()
    print("\n" + "="*60)
    print("ENSEMBLE MODEL RESULTS")
    print("="*60)
    print(ensemble_df.to_string(index=False))
    
    predictor.plot_comprehensive_results()
    
    predictor.detailed_classification_report()
    
    results_df.to_csv('enhanced_cvd_results.csv', index=False)
    print("\nResults saved to enhanced_cvd_results.csv")
    
    print("\n" + "="*60)
    print("ENHANCED PIPELINE COMPLETED")
    print("="*60)
    
    best_acc = results_df.iloc[0]['Accuracy']
    if best_acc >= 0.80:
        print(f"\nSUCCESS! Achieved {best_acc:.2%} accuracy (Target: 80%)")
    else:
        print(f"\nAchieved {best_acc:.2%} accuracy. Consider:")
        print("  - Increasing n_iter in RandomizedSearchCV")
        print("  - Trying different n_features in feature_selection")
        print("  - Experimenting with feature engineering")
        print("  - Using ensemble models")
    
    return predictor


def experiment_configurations():
    print("\n" + "="*60)
    print("RUNNING EXPERIMENTS WITH DIFFERENT CONFIGS")
    print("="*60)
    
    configs = [
        {'name': 'Config 1: No Feature Selection',
         'feature_selection': 'none', 'power_transform': True, 'smote': True, 'n_features': 0},
        
        {'name': 'Config 2: PCA 40 components',
         'feature_selection': 'pca', 'n_features': 40, 'power_transform': False, 'smote': True},
        
        {'name': 'Config 3: Combined Selection 30 features',
         'feature_selection': 'combined', 'n_features': 30, 'power_transform': True, 'smote': True},
    ]
    
    best_config = None
    best_score = 0
    
    for config in configs:
        print(f"\n\nTesting: {config['name']}")
        print("-" * 60)
        
        predictor = EnhancedCVDPredictor('./public/cvd_dataset.csv')
        predictor.load_data()
        predictor.advanced_preprocessing()
        predictor.split_and_balance_data(use_smote=config['smote'])
        predictor.advanced_feature_scaling(use_power_transform=config['power_transform'])
        
        if config['feature_selection'] == 'pca':
            predictor.feature_selection(method='pca', n_features=config.get('n_features', 30))
        elif config['feature_selection'] == 'combined':
            predictor.feature_selection(method='combined', n_features=config.get('n_features', 30))
        else:
            predictor.feature_selection(method='none')
        
        predictor.model_configs = {
            'XGBoost': {
                'model': XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss'),
                'params': {
                    'n_estimators': [200, 300],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [4, 6, 8],
                    'subsample': [0.8, 1.0]
                }
            },
            'LightGBM': {
                'model': LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
                'params': {
                    'n_estimators': [200, 300],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [4, 6, 8],
                    'num_leaves': [31, 50]
                }
            },
            'CatBoost': {
                'model': CatBoostClassifier(random_state=RANDOM_STATE, verbose=False),
                'params': {
                    'iterations': [200, 300],
                    'learning_rate': [0.01, 0.1],
                    'depth': [4, 6, 8]
                }
            }
        }
        
        predictor.train_with_randomized_search(n_iter=20, cv_folds=3)
        results = predictor.evaluate_models()
        
        top_acc = results.iloc[0]['Accuracy']
        print(f"Best accuracy: {top_acc:.4f}")
        
        if top_acc > best_score:
            best_score = top_acc
            best_config = config
    
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Best Configuration: {best_config['name']}")
    print(f"Best Accuracy: {best_score:.4f}")
    
    return best_config


if __name__ == "__main__":
    
    predictor = main()
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS TO FURTHER IMPROVE ACCURACY:")
    print("="*60)
    print("1. Increase RandomizedSearchCV iterations (n_iter=100)")
    print("2. Try different feature selection methods")
    print("3. Experiment with different SMOTE strategies")
    print("4. Use ensemble of ensembles")
    print("5. Fine-tune best model with GridSearchCV")
    print("6. Add more domain-specific features")
    print("7. Try neural network architectures")
    print("="*60)