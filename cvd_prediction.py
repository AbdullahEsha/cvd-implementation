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
        
        # Phase 3: Feature Engineering
        print("\nPhase 3: Feature Engineering...")
        
        # Interaction features
        if 'BMI' in df.columns and 'Age' in df.columns:
            df['BMI_Age_Interaction'] = df['BMI'] * df['Age']
        
        if 'Systolic BP' in df.columns and 'Total Cholesterol (mg/dL)' in df.columns:
            df['BP_Cholesterol_Interaction'] = df['Systolic BP'] * df['Total Cholesterol (mg/dL)']
        
        if 'HDL (mg/dL)' in df.columns and 'Estimated LDL (mg/dL)' in df.columns:
            # Avoid division by zero
            df['HDL_LDL_Ratio'] = df['HDL (mg/dL)'] / (df['Estimated LDL (mg/dL)'] + 1)
        
        print(f"Created interaction features: {df.shape[1] - self.df.shape[1]} new features")
        
        # Encode categorical variables
        print("\nPhase 4: Encoding Categorical Variables...")
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
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df.drop(col, axis=1, inplace=True)
        
        # Drop original target and non-feature columns
        features_to_drop = ['CVD Risk Level', 'CVD Risk Score', 'Blood Pressure (mmHg)']
        df = df.drop([col for col in features_to_drop if col in df.columns], axis=1)
        
        self.df_processed = df
        print(f"\nFinal dataset shape: {df.shape}")
        
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
        
        self.scaler = StandardScaler()
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Features scaled using StandardScaler")
        
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
        """Train all models with optional hyperparameter tuning"""
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)
        
        for name, config in self.model_configs.items():
            print(f"\nTraining {name}...")
            
            # Select scaled or unscaled data
            if config['scaled']:
                X_tr = self.X_train_scaled
                X_v = self.X_val_scaled
            else:
                X_tr = self.X_train
                X_v = self.X_val
            
            if use_grid_search:
                # Hyperparameter tuning with GridSearchCV
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=cv,
                    scoring='f1_weighted',
                    n_jobs=-1,
                    verbose=0
                )
                grid_search.fit(X_tr, self.y_train)
                
                best_model = grid_search.best_estimator_
                print(f"Best params: {grid_search.best_params_}")
                print(f"Best CV score: {grid_search.best_score_:.4f}")
            else:
                # Train with default parameters
                best_model = config['model']
                best_model.fit(X_tr, self.y_train)
            
            # Store the trained model
            self.models[name] = {
                'model': best_model,
                'scaled': config['scaled']
            }
            
            # Evaluate on validation set
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


if __name__ == "__main__":
    # Run the complete pipeline
    predictor = main()
    
    # Access results
    # predictor.results_df  # Results dataframe
    # predictor.models  # Trained models
    # predictor.feature_names  # Feature names
