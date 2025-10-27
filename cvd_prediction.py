import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve, auc)

# ML Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, VotingClassifier,
                              StackingClassifier)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Explainability
import shap
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class CVDRiskPredictor:
    """Complete CVD Risk Prediction Pipeline"""
    
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
        self.models = {}
        self.results = {}
        self.feature_names = None
        
    def load_data(self):
        """Load the CVD dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv("./public/cvd_dataset.csv")
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print(f"\nTarget distribution:\n{self.df['CVD Risk Level'].value_counts()}")
        return self.df
    
    def explore_data(self):
        """Basic exploratory data analysis"""
        print("\n" + "="*60)
        print("DATA EXPLORATION")
        print("="*60)
        
        print("\nDataset Info:")
        print(self.df.info())
        
        print("\nMissing Values:")
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing) > 0:
            print(missing)
        else:
            print("No missing values found")
        
        print("\nNumerical Statistics:")
        print(self.df.describe())
        
    def preprocess_data(self):
        """Complete data preprocessing pipeline"""
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        
        df = self.df.copy()
        
        # Phase 1: Handle Missing Values
        print("\nPhase 1: Handling Missing Values...")
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from features
        if 'CVD Risk Level' in numeric_cols:
            numeric_cols.remove('CVD Risk Level')
        if 'CVD Risk Score' in numeric_cols:
            numeric_cols.remove('CVD Risk Score')
        if 'CVD Risk Level' in categorical_cols:
            categorical_cols.remove('CVD Risk Level')
        
        # Impute numeric features with median
        numeric_imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
        
        # Impute categorical features with mode
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        if len(categorical_cols) > 0:
            df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
        
        print(f"Missing values handled: {df.isnull().sum().sum()} remaining")
        
        # Phase 2: Outlier Detection and Handling
        print("\nPhase 2: Outlier Detection (IQR Method)...")
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                # Cap outliers instead of removing
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                print(f"  {col}: {outliers} outliers capped")
        
        # Encode categorical variables BEFORE feature engineering
        print("\nPhase 3: Encoding Categorical Variables...")
        le = LabelEncoder()
        
        # Encode target variable
        df['CVD_Risk_Encoded'] = le.fit_transform(df['CVD Risk Level'])
        self.label_encoder = le
        print(f"Target classes: {list(le.classes_)}")
        
        # One-hot encode other categorical features
        categorical_features = ['Sex', 'Smoking Status', 'Diabetes Status', 
                            'Physical Activity Level', 'Family History of CVD',
                            'Blood Pressure Category']
        
        for col in categorical_features:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
                df = pd.concat([df, dummies], axis=1)
                df.drop(col, axis=1, inplace=True)
        
        print(f"Categorical encoding complete. Shape: {df.shape}")
        
        # Phase 4: Feature Engineering
        print("\nPhase 4: Feature Engineering...")
        
        # Get current numeric columns (after encoding)
        current_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Interaction features
        if 'BMI' in current_numeric_cols and 'Age' in current_numeric_cols:
            df['BMI_Age_Interaction'] = df['BMI'] * df['Age']
        
        if 'Systolic BP' in current_numeric_cols and 'Total Cholesterol (mg/dL)' in current_numeric_cols:
            df['BP_Cholesterol_Interaction'] = df['Systolic BP'] * df['Total Cholesterol (mg/dL)']
        
        if 'HDL (mg/dL)' in current_numeric_cols and 'Estimated LDL (mg/dL)' in current_numeric_cols:
            # Avoid division by zero and handle edge cases
            ldl_safe = df['Estimated LDL (mg/dL)'].replace(0, 1)
            df['HDL_LDL_Ratio'] = df['HDL (mg/dL)'] / (ldl_safe + 1)
            # Cap extreme ratios
            df['HDL_LDL_Ratio'] = df['HDL_LDL_Ratio'].clip(upper=10)
        
        print(f"Created interaction features")
        
        # Phase 5: Handle any remaining infinite or extreme values
        print("\nPhase 5: Handling Infinite and Extreme Values...")
        
        # Replace infinite values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Check for any remaining NaN values after feature engineering
        nan_cols = df.isnull().sum()
        nan_cols = nan_cols[nan_cols > 0]
        
        if len(nan_cols) > 0:
            print(f"Found {len(nan_cols)} columns with NaN values after feature engineering")
            # Impute with median for numeric columns
            for col in nan_cols.index:
                if col != 'CVD_Risk_Encoded' and col != 'CVD Risk Level':
                    median_val = df[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    df[col].fillna(median_val, inplace=True)
            print("NaN values imputed")
        
        # Cap extreme values using percentiles (99th percentile)
        numeric_cols_fe = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'CVD_Risk_Encoded' in numeric_cols_fe:
            numeric_cols_fe.remove('CVD_Risk_Encoded')
        if 'CVD Risk Score' in numeric_cols_fe:
            numeric_cols_fe.remove('CVD Risk Score')
            
        for col in numeric_cols_fe:
            upper_limit = df[col].quantile(0.99)
            lower_limit = df[col].quantile(0.01)
            df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
        
        print("Extreme values capped at 1st and 99th percentiles")
        
        # Drop original target and non-feature columns
        features_to_drop = ['CVD Risk Level', 'CVD Risk Score', 'Blood Pressure (mmHg)']
        df = df.drop([col for col in features_to_drop if col in df.columns], axis=1)
        
        # CRITICAL: Ensure ALL columns are numeric (including boolean dummies)
        print("\nPhase 6: Final Type Conversion...")
        for col in df.columns:
            if col != 'CVD_Risk_Encoded':
                if df[col].dtype == 'bool':
                    df[col] = df[col].astype(int)
                elif df[col].dtype == 'object':
                    print(f"  Converting object column '{col}' to numeric...")
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if df[col].isnull().any():
                        df[col].fillna(df[col].median(), inplace=True)
        
        # Final verification
        print("\nFinal Data Type Check:")
        print(df.dtypes.value_counts())
        
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if 'CVD_Risk_Encoded' in non_numeric:
            non_numeric.remove('CVD_Risk_Encoded')
        
        if non_numeric:
            print(f"\nWARNING: Non-numeric columns still present: {non_numeric}")
            for col in non_numeric:
                print(f"  {col}: {df[col].dtype}, sample values: {df[col].head()}")
        
        self.df_processed = df
        print(f"\nFinal dataset shape: {df.shape}")
        print(f"All features are numeric: {all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns if col != 'CVD_Risk_Encoded')}")
        
        return df
    
    def split_data(self, test_size=0.15, val_size=0.15):
        """Split data into train, validation, and test sets"""
        print("\n" + "="*60)
        print("DATA SPLITTING")
        print("="*60)
        
        df = self.df_processed.copy()
        
        # Separate features and target
        X = df.drop('CVD_Risk_Encoded', axis=1)
        y = df['CVD_Risk_Encoded']
        
        self.feature_names = X.columns.tolist()
        
        # First split: train+val and test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )
        
        # Second split: train and validation
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=RANDOM_STATE, stratify=y_temp
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Validation set: {self.X_val.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        print("\nClass distribution:")
        print(f"Train: {dict(pd.Series(self.y_train).value_counts())}")
        print(f"Val: {dict(pd.Series(self.y_val).value_counts())}")
        print(f"Test: {dict(pd.Series(self.y_test).value_counts())}")
        
    def scale_features(self):
        """Scale features using StandardScaler"""
        print("\n" + "="*60)
        print("FEATURE SCALING")
        print("="*60)
        
        # Validate data before scaling
        print("\nValidating data before scaling...")
        
        # First, ensure all columns are numeric
        print("Converting all columns to numeric...")
        for col in self.X_train.columns:
            if self.X_train[col].dtype == 'object':
                print(f"  Warning: Column '{col}' has object dtype. Converting to numeric...")
                self.X_train[col] = pd.to_numeric(self.X_train[col], errors='coerce')
                self.X_val[col] = pd.to_numeric(self.X_val[col], errors='coerce')
                self.X_test[col] = pd.to_numeric(self.X_test[col], errors='coerce')
        
        # Now check for infinite values (after ensuring numeric types)
        if np.isinf(self.X_train.values).any():
            print("WARNING: Found infinite values in training data. Replacing with max finite values...")
            self.X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Check for NaN values
        if self.X_train.isnull().any().any():
            print("WARNING: Found NaN values in training data. Imputing with median...")
            for col in self.X_train.columns:
                if self.X_train[col].isnull().any():
                    median_val = self.X_train[col].median()
                    if pd.isna(median_val):
                        # If median is also NaN (all values are NaN), use 0
                        median_val = 0
                    self.X_train[col].fillna(median_val, inplace=True)
                    print(f"  Imputed {col} with {median_val}")
    
        # Apply same fixes to validation and test sets
        for dataset_name, dataset in [('validation', self.X_val), ('test', self.X_test)]:
            if np.isinf(dataset.values).any() or dataset.isnull().any().any():
                dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
                for col in dataset.columns:
                    if dataset[col].isnull().any():
                        # Use training set median for consistency
                        train_median = self.X_train[col].median()
                        if pd.isna(train_median):
                            train_median = 0
                        dataset[col].fillna(train_median, inplace=True)

        print("Data validation completed")

        # Verify all data is numeric and finite before scaling
        print("\nFinal validation before scaling...")
        for dataset_name, dataset in [('train', self.X_train), ('val', self.X_val), ('test', self.X_test)]:
            # Check data types
            non_numeric = dataset.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric:
                print(f"ERROR: Non-numeric columns in {dataset_name}: {non_numeric}")
                raise ValueError(f"All columns must be numeric before scaling. Found: {non_numeric}")
            
            # Check for infinite values
            if np.isinf(dataset.values).any():
                print(f"ERROR: Infinite values still present in {dataset_name}")
                raise ValueError(f"Infinite values detected in {dataset_name} set")
            
            # Check for NaN values
            if dataset.isnull().any().any():
                print(f"ERROR: NaN values still present in {dataset_name}")
                nan_cols = dataset.columns[dataset.isnull().any()].tolist()
                print(f"Columns with NaN: {nan_cols}")
                raise ValueError(f"NaN values detected in {dataset_name} set")
            
        print("All data is clean and numeric. Proceeding with scaling...")
        
        # Now scale
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print("Features scaled using StandardScaler")
        print(f"Scaled data shape: {self.X_train_scaled.shape}")

    def define_models(self):
        """Define all ML models with hyperparameter grids"""
        print("\n" + "="*60)
        print("DEFINING MODELS")
        print("="*60)
        
        self.model_configs = {
            # Traditional ML
            'Logistic Regression': {
                'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['lbfgs']
                },
                'scaled': True
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier(random_state=RANDOM_STATE),
                'params': {
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'scaled': False
            },
            'Naive Bayes': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7]
                },
                'scaled': True
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },
                'scaled': True
            },
            'SVM': {
                'model': SVC(probability=True, random_state=RANDOM_STATE),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                },
                'scaled': True
            },
            # Ensemble Methods
            'Random Forest': {
                'model': RandomForestClassifier(random_state=RANDOM_STATE),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                'scaled': False
            },
            'AdaBoost': {
                'model': AdaBoostClassifier(random_state=RANDOM_STATE),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0]
                },
                'scaled': False
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5, 7]
                },
                'scaled': False
            },
            'XGBoost': {
                'model': XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                },
                'scaled': False
            },
            'LightGBM': {
                'model': LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5, 7],
                    'num_leaves': [31, 50]
                },
                'scaled': False
            },
            'CatBoost': {
                'model': CatBoostClassifier(random_state=RANDOM_STATE, verbose=False),
                'params': {
                    'iterations': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'depth': [4, 6, 8]
                },
                'scaled': False
            },
            # Neural Network
            'MLP': {
                'model': MLPClassifier(random_state=RANDOM_STATE, max_iter=500),
                'params': {
                    'hidden_layer_sizes': [(64, 32), (128, 64)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001],
                    'learning_rate': ['constant', 'adaptive']
                },
                'scaled': True
            }
        }
        
        print(f"Defined {len(self.model_configs)} models")
        
    def train_models(self, use_grid_search=True, cv_folds=5):
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)
        
        for name, config in self.model_configs.items():
            print(f"\nTraining {name}...")
            
            if config['scaled']:
                X_tr = self.X_train_scaled
                X_v = self.X_val_scaled
            else:
                X_tr = self.X_train
                X_v = self.X_val
            
            if use_grid_search:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=cv,
                    scoring='f1_weighted',
                    n_jobs=1,
                    verbose=0
                )
                grid_search.fit(X_tr, self.y_train)
                
                best_model = grid_search.best_estimator_
                print(f"Best params: {grid_search.best_params_}")
                print(f"Best CV score: {grid_search.best_score_:.4f}")
            else:
                best_model = config['model']
                best_model.fit(X_tr, self.y_train)
            
            self.models[name] = {
                'model': best_model,
                'scaled': config['scaled']
            }
            
            y_val_pred = best_model.predict(X_v)
            val_acc = accuracy_score(self.y_val, y_val_pred)
            val_f1 = f1_score(self.y_val, y_val_pred, average='weighted')
            
            print(f"Validation Accuracy: {val_acc:.4f}")
            print(f"Validation F1-Score: {val_f1:.4f}")
        
        print(f"\n✓ All {len(self.models)} models trained successfully")
    
    def evaluate_models(self):
        """Comprehensive evaluation of all models"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        results_list = []
        
        for name, model_dict in self.models.items():
            print(f"\nEvaluating {name}...")
            
            model = model_dict['model']
            
            # Select test data
            if model_dict['scaled']:
                X_test = self.X_test_scaled
            else:
                X_test = self.X_test
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            
            # ROC-AUC (for multiclass)
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
            
            # Store predictions for later analysis
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
        
        # Create results dataframe
        self.results_df = pd.DataFrame(results_list)
        self.results_df = self.results_df.sort_values('F1-Score', ascending=False)
        
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(self.results_df.to_string(index=False))
        
        # Best model
        best_model_name = self.results_df.iloc[0]['Model']
        best_f1 = self.results_df.iloc[0]['F1-Score']
        print(f"\n🏆 Best Model: {best_model_name} (F1-Score: {best_f1:.4f})")
        
        return self.results_df
    
    def plot_results(self):
        """Visualize model comparison"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Accuracy Comparison
        ax1 = axes[0, 0]
        self.results_df.plot(x='Model', y='Accuracy', kind='barh', ax=ax1, legend=False, color='skyblue')
        ax1.set_xlabel('Accuracy', fontsize=12)
        ax1.set_ylabel('Model', fontsize=12)
        ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. F1-Score Comparison
        ax2 = axes[0, 1]
        self.results_df.plot(x='Model', y='F1-Score', kind='barh', ax=ax2, legend=False, color='lightcoral')
        ax2.set_xlabel('F1-Score', fontsize=12)
        ax2.set_ylabel('Model', fontsize=12)
        ax2.set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Multi-metric Comparison
        ax3 = axes[1, 0]
        metrics_df = self.results_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score']]
        metrics_df.head(8).plot(kind='bar', ax=ax3, width=0.8)
        ax3.set_ylabel('Score', fontsize=12)
        ax3.set_title('Multi-Metric Comparison (Top 8 Models)', fontsize=14, fontweight='bold')
        ax3.legend(loc='lower right')
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # 4. Confusion Matrix for Best Model
        ax4 = axes[1, 1]
        best_model_name = self.results_df.iloc[0]['Model']
        y_pred = self.results[best_model_name]['y_pred']
        cm = confusion_matrix(self.y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
        ax4.set_xlabel('Predicted', fontsize=12)
        ax4.set_ylabel('Actual', fontsize=12)
        ax4.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('cvd_model_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: cvd_model_comparison.png")
        plt.show()
    
    def feature_importance_analysis(self, top_n=15):
        """Analyze feature importance using the best tree-based model"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Find best tree-based model
        tree_models = ['Random Forest', 'XGBoost', 'LightGBM', 'CatBoost', 'Gradient Boosting']
        
        best_tree_model = None
        best_score = 0
        
        for model_name in tree_models:
            if model_name in self.models:
                score = self.results[model_name]['metrics']['f1']
                if score > best_score:
                    best_score = score
                    best_tree_model = model_name
        
        if best_tree_model is None:
            print("No tree-based models found for feature importance")
            return
        
        print(f"Using {best_tree_model} for feature importance")
        
        model = self.models[best_tree_model]['model']
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Create dataframe
            feature_imp_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(top_n)
            
            print(f"\nTop {top_n} Important Features:")
            print(feature_imp_df.to_string(index=False))
            
            # Plot
            plt.figure(figsize=(10, 8))
            plt.barh(feature_imp_df['Feature'], feature_imp_df['Importance'], color='teal')
            plt.xlabel('Importance', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.title(f'Top {top_n} Feature Importances - {best_tree_model}', 
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("\n✓ Saved: feature_importance.png")
            plt.show()
    
    def generate_classification_report(self):
        """Generate detailed classification reports"""
        print("\n" + "="*60)
        print("CLASSIFICATION REPORTS")
        print("="*60)
        
        best_model_name = self.results_df.iloc[0]['Model']
        print(f"\nClassification Report - {best_model_name}")
        print("-" * 60)
        
        y_pred = self.results[best_model_name]['y_pred']
        
        target_names = self.label_encoder.classes_
        report = classification_report(self.y_test, y_pred, target_names=target_names)
        print(report)
    
    def save_results(self, filename='cvd_results.csv'):
        """Save results to CSV"""
        self.results_df.to_csv(filename, index=False)
        print(f"\n✓ Results saved to {filename}")
    
    def shap_analysis(self, model_name=None, sample_size=100):
        """SHAP analysis for model explainability"""
        print("\n" + "="*60)
        print("SHAP EXPLAINABILITY ANALYSIS")
        print("="*60)
        
        # Use best model if not specified
        if model_name is None:
            model_name = self.results_df.iloc[0]['Model']
        
        print(f"\nAnalyzing {model_name}...")
        
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return
        
        model = self.models[model_name]['model']
        
        # Select appropriate data
        if self.models[model_name]['scaled']:
            X_test = pd.DataFrame(self.X_test_scaled, columns=self.feature_names)
            X_train = pd.DataFrame(self.X_train_scaled, columns=self.feature_names)
        else:
            X_test = self.X_test
            X_train = self.X_train
        
        # Sample data for faster computation
        if len(X_test) > sample_size:
            X_test_sample = X_test.sample(n=sample_size, random_state=RANDOM_STATE)
        else:
            X_test_sample = X_test
        
        try:
            # Create SHAP explainer
            if model_name in ['Random Forest', 'XGBoost', 'LightGBM', 'CatBoost', 'Gradient Boosting']:
                explainer = shap.TreeExplainer(model)
            else:
                # Use KernelExplainer for other models (slower)
                background = shap.sample(X_train, 100)
                explainer = shap.KernelExplainer(model.predict_proba, background)
            
            # Calculate SHAP values
            print("Calculating SHAP values...")
            shap_values = explainer.shap_values(X_test_sample)
            
            # For multiclass, take values for class 1 or average
            if isinstance(shap_values, list):
                shap_values_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_values_plot = shap_values
            
            # SHAP Summary Plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values_plot, X_test_sample, show=False)
            plt.title(f'SHAP Summary Plot - {model_name}', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: shap_summary.png")
            plt.show()
            
            # SHAP Bar Plot (Feature Importance)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values_plot, X_test_sample, plot_type="bar", show=False)
            plt.title(f'SHAP Feature Importance - {model_name}', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig('shap_importance.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: shap_importance.png")
            plt.show()
            
            print("\n✓ SHAP analysis completed")
            
        except Exception as e:
            print(f"Error in SHAP analysis: {e}")
            print("Note: SHAP analysis works best with tree-based models")
    
    def build_ensemble_models(self):
        """Build stacking and voting ensembles"""
        print("\n" + "="*60)
        print("BUILDING ENSEMBLE MODELS")
        print("="*60)
        
        # Get top 3 models
        top_models = self.results_df.head(3)['Model'].tolist()
        print(f"\nTop 3 models for ensemble: {top_models}")
        
        estimators = []
        for model_name in top_models:
            if model_name in self.models:
                estimators.append((model_name, self.models[model_name]['model']))
        
        if len(estimators) < 2:
            print("Not enough models for ensemble")
            return
        
        # Determine if we need scaled data
        scaled_needed = any(self.models[name]['scaled'] for name in top_models)
        
        if scaled_needed:
            X_tr = self.X_train_scaled
            X_te = self.X_test_scaled
        else:
            X_tr = self.X_train
            X_te = self.X_test
        
        # 1. Voting Classifier (Soft Voting)
        print("\n1. Training Voting Ensemble (Soft)...")
        voting_soft = VotingClassifier(estimators=estimators, voting='soft')
        voting_soft.fit(X_tr, self.y_train)
        
        y_pred_soft = voting_soft.predict(X_te)
        acc_soft = accuracy_score(self.y_test, y_pred_soft)
        f1_soft = f1_score(self.y_test, y_pred_soft, average='weighted')
        
        print(f"Voting (Soft) - Accuracy: {acc_soft:.4f}, F1: {f1_soft:.4f}")
        
        # 2. Voting Classifier (Hard Voting)
        print("\n2. Training Voting Ensemble (Hard)...")
        voting_hard = VotingClassifier(estimators=estimators, voting='hard')
        voting_hard.fit(X_tr, self.y_train)
        
        y_pred_hard = voting_hard.predict(X_te)
        acc_hard = accuracy_score(self.y_test, y_pred_hard)
        f1_hard = f1_score(self.y_test, y_pred_hard, average='weighted')
        
        print(f"Voting (Hard) - Accuracy: {acc_hard:.4f}, F1: {f1_hard:.4f}")
        
        # 3. Stacking Classifier
        print("\n3. Training Stacking Ensemble...")
        final_estimator = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5
        )
        stacking.fit(X_tr, self.y_train)
        
        y_pred_stack = stacking.predict(X_te)
        acc_stack = accuracy_score(self.y_test, y_pred_stack)
        f1_stack = f1_score(self.y_test, y_pred_stack, average='weighted')
        
        print(f"Stacking - Accuracy: {acc_stack:.4f}, F1: {f1_stack:.4f}")
        
        # Store ensemble models
        self.ensemble_models = {
            'Voting_Soft': voting_soft,
            'Voting_Hard': voting_hard,
            'Stacking': stacking
        }
        
        # Add to results
        ensemble_results = [
            {'Model': 'Voting Ensemble (Soft)', 'Accuracy': acc_soft, 'F1-Score': f1_soft},
            {'Model': 'Voting Ensemble (Hard)', 'Accuracy': acc_hard, 'F1-Score': f1_hard},
            {'Model': 'Stacking Ensemble', 'Accuracy': acc_stack, 'F1-Score': f1_stack}
        ]
        
        print("\n✓ Ensemble models created successfully")
        
        return pd.DataFrame(ensemble_results)
    
    def cross_validation_analysis(self, cv_folds=10):
        print("\n" + "="*60)
        print("CROSS-VALIDATION ANALYSIS")
        print("="*60)
        
        cv_results = []
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        
        for name, model_dict in self.models.items():
            print(f"\nCross-validating {name}...")
            
            model = model_dict['model']
            
            if model_dict['scaled']:
                X = self.X_train_scaled
            else:
                X = self.X_train
            
            scores = cross_val_score(model, X, self.y_train, cv=cv, 
                                    scoring='f1_weighted', n_jobs=1)
            
            cv_results.append({
                'Model': name,
                'Mean_CV_F1': scores.mean(),
                'Std_CV_F1': scores.std(),
                'Min_CV_F1': scores.min(),
                'Max_CV_F1': scores.max()
            })
            
            print(f"CV F1-Score: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        cv_df = pd.DataFrame(cv_results).sort_values('Mean_CV_F1', ascending=False)
        
        print("\n" + "="*60)
        print("CROSS-VALIDATION RESULTS")
        print("="*60)
        print(cv_df.to_string(index=False))
        
        plt.figure(figsize=(12, 8))
        plt.barh(cv_df['Model'], cv_df['Mean_CV_F1'], xerr=cv_df['Std_CV_F1'], 
                color='steelblue', capsize=5)
        plt.xlabel('Mean F1-Score', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.title(f'{cv_folds}-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('cross_validation_results.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: cross_validation_results.png")
        plt.show()
        
        return cv_df
    
    def learning_curves(self, model_name=None):
        print("\n" + "="*60)
        print("LEARNING CURVE ANALYSIS")
        print("="*60)
        
        from sklearn.model_selection import learning_curve
        
        if model_name is None:
            model_name = self.results_df.iloc[0]['Model']
        
        print(f"\nGenerating learning curves for {model_name}...")
        
        model = self.models[model_name]['model']
        
        if self.models[model_name]['scaled']:
            X = self.X_train_scaled
        else:
            X = self.X_train
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, self.y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5, scoring='f1_weighted', n_jobs=1
        )
        
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training Score', color='blue', marker='o')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.15, color='blue')
        
        plt.plot(train_sizes, val_mean, label='Validation Score', color='red', marker='s')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.15, color='red')
        
        plt.xlabel('Training Set Size', fontsize=12)
        plt.ylabel('F1-Score', fontsize=12)
        plt.title(f'Learning Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: learning_curve.png")
        plt.show()
    
    def roc_curve_analysis(self):
        """Plot ROC curves for all models"""
        print("\n" + "="*60)
        print("ROC CURVE ANALYSIS")
        print("="*60)
        
        plt.figure(figsize=(12, 8))
        
        for name in self.results_df.head(8)['Model']:
            if name not in self.results:
                continue
                
            y_pred_proba = self.results[name]['y_pred_proba']
            
            # For multiclass, compute ROC for each class and average
            n_classes = y_pred_proba.shape[1]
            
            if n_classes == 2:
                fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
            else:
                # Macro-average ROC curve
                from sklearn.preprocessing import label_binarize
                y_test_bin = label_binarize(self.y_test, classes=range(n_classes))
                
                fpr_list, tpr_list = [], []
                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                    fpr_list.append(fpr)
                    tpr_list.append(tpr)
                
                # Compute macro-average
                all_fpr = np.unique(np.concatenate(fpr_list))
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(n_classes):
                    mean_tpr += np.interp(all_fpr, fpr_list[i], tpr_list[i])
                mean_tpr /= n_classes
                
                roc_auc = auc(all_fpr, mean_tpr)
                plt.plot(all_fpr, mean_tpr, label=f'{name} (AUC={roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: roc_curves.png")
        plt.show()
    
    def predict_new_patient(self, patient_data):
        """Predict CVD risk for a new patient"""
        print("\n" + "="*60)
        print("NEW PATIENT PREDICTION")
        print("="*60)
        
        best_model_name = self.results_df.iloc[0]['Model']
        model = self.models[best_model_name]['model']
        
        # Convert patient data to dataframe
        patient_df = pd.DataFrame([patient_data])
        
        # Apply same preprocessing
        # Note: This is simplified - you'd need to apply the exact same transformations
        
        if self.models[best_model_name]['scaled']:
            patient_scaled = self.scaler.transform(patient_df)
            prediction = model.predict(patient_scaled)[0]
            probability = model.predict_proba(patient_scaled)[0]
        else:
            prediction = model.predict(patient_df)[0]
            probability = model.predict_proba(patient_df)[0]
        
        risk_level = self.label_encoder.inverse_transform([prediction])[0]
        
        print(f"\nModel: {best_model_name}")
        print(f"Predicted Risk Level: {risk_level}")
        print(f"Prediction Probabilities:")
        for i, prob in enumerate(probability):
            class_name = self.label_encoder.classes_[i]
            print(f"  {class_name}: {prob:.2%}")
        
        return risk_level, probability


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("="*60)
    print("CVD RISK PREDICTION SYSTEM")
    print("CAIR-CVD-2025 Dataset")
    print("="*60)
    
    # Initialize predictor
    predictor = CVDRiskPredictor('sample_cvr_data.csv')
    
    # Load and explore data
    predictor.load_data()
    predictor.explore_data()
    
    # Preprocess
    predictor.preprocess_data()
    
    # Split data
    predictor.split_data()
    
    # Scale features
    predictor.scale_features()
    
    # Define models
    predictor.define_models()
    
    # Train models (set use_grid_search=False for faster training without tuning)
    predictor.train_models(use_grid_search=True, cv_folds=5)
    
    # Evaluate models
    predictor.evaluate_models()
    
    # Generate visualizations
    predictor.plot_results()
    
    # Feature importance
    predictor.feature_importance_analysis()
    
    # Classification report
    predictor.generate_classification_report()
    
    # Save results
    predictor.save_results()
    
    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    
    return predictor


def advanced_analysis(predictor):
    """Run advanced analysis functions"""
    print("\n" + "="*60)
    print("RUNNING ADVANCED ANALYSIS")
    print("="*60)
    
    # 1. Cross-validation analysis
    cv_results = predictor.cross_validation_analysis(cv_folds=10)
    
    # 2. Learning curves
    predictor.learning_curves()
    
    # 3. ROC curve analysis
    predictor.roc_curve_analysis()
    
    # 4. Build ensemble models
    ensemble_df = predictor.build_ensemble_models()
    print("\n" + "="*60)
    print("ENSEMBLE MODEL RESULTS")
    print("="*60)
    print(ensemble_df.to_string(index=False))
    
    # 5. SHAP analysis (for best model)
    predictor.shap_analysis(sample_size=100)
    
    print("\n" + "="*60)
    print("✓ ADVANCED ANALYSIS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    # =========================================================================
    # BASIC PIPELINE - Run this first
    # =========================================================================
    predictor = main()
    
    # =========================================================================
    # ADVANCED ANALYSIS - Uncomment to run additional analyses
    # =========================================================================
    # advanced_analysis(predictor)
    
    # =========================================================================
    # EXAMPLE: Predict for a new patient
    # =========================================================================
    # new_patient = {
    #     'Age': 45,
    #     'BMI': 28.5,
    #     'Systolic BP': 135,
    #     'Diastolic BP': 85,
    #     'Total Cholesterol (mg/dL)': 220,
    #     'HDL (mg/dL)': 45,
    #     'Estimated LDL (mg/dL)': 150,
    #     'Fasting Blood Sugar (mg/dL)': 110,
    #     # Add all other required features...
    # }
    # predictor.predict_new_patient(new_patient)
    
    # =========================================================================
    # Access trained models and results
    # =========================================================================
    # predictor.results_df          # Results dataframe
    # predictor.models              # Dictionary of trained models
    # predictor.feature_names       # List of feature names
    # predictor.X_test, predictor.y_test  # Test data
    
    print("\n" + "="*60)
    print("ALL ANALYSES COMPLETE")
    print("="*60)
    print("\nGenerated Files:")
    print("  • cvd_model_comparison.png")
    print("  • feature_importance.png")
    print("  • cvd_results.csv")
    print("\nOptional (if advanced_analysis run):")
    print("  • cross_validation_results.png")
    print("  • learning_curve.png")
    print("  • roc_curves.png")
    print("  • shap_summary.png")
    print("  • shap_importance.png")
    print("="*60)