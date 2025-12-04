import pandas as pd
import numpy as np
import warnings, joblib
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

SEED = 42
np.random.seed(SEED)

# ====================== LOAD DATA ======================
df = pd.read_csv('./public/cvd_dataset.csv')
df.columns = [c.strip() for c in df.columns]

# Target
df['is_high_risk'] = (df['CVD Risk Level'].astype(str).str.upper().str.strip() == 'HIGH').astype(int)
y = df['is_high_risk'].values
X = df.drop(columns=['is_high_risk', 'CVD Risk Level'], errors='ignore')

print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"High-risk prevalence: {y.mean():.2%}")

# ====================== YOUR ORIGINAL FUNCTIONS (EXACTLY) ======================
def full_categorical_encoding(X):
    X = X.copy()
    cat_maps = {
        'Sex': {'M':1, 'F':0, 'Male':1, 'Female':0},
        'Smoking Status': {'Y':1, 'N':0, 'Yes':1, 'No':0},
        'Diabetes Status': {'Y':1, 'N':0, 'Yes':1, 'No':0},
        'Family History of CVD': {'Y':1, 'N':0, 'Yes':1, 'No':0},
        'Physical Activity Level': {'Low':0, 'Moderate':1, 'High':2},
        'Blood Pressure Category': {
            'Normal':0, 'Elevated':1,
            'Hypertension Stage 1':2, 'Hypertension Stage 2':3,
            'Hypertensive Crisis':4
        }
    }
    for col, mapping in cat_maps.items():
        if col in X.columns:
            X[col] = X[col].astype(str).str.strip().map(mapping).fillna(-1)
    return X

def parse_bp_height(X):
    X = X.copy()
    if 'Blood Pressure (mmHg)' in X.columns:
        def parse(x):
            try:
                s, d = str(x).split('/')
                return float(s.strip()), float(d.strip())
            except:
                return np.nan, np.nan
        bp = X['Blood Pressure (mmHg)'].apply(lambda x: pd.Series(parse(x)))
        X['Systolic BP'] = X.get('Systolic BP', np.nan).combine_first(bp[0])
        X['Diastolic BP'] = X.get('Diastolic BP', np.nan).combine_first(bp[1])
        X = X.drop(columns=['Blood Pressure (mmHg)'], errors='ignore')
    if 'Height (cm)' in X.columns:
        X['Height (m)'] = X.get('Height (m)', np.nan).fillna(X['Height (cm)'] / 100)
        X = X.drop(columns=['Height (cm)'], errors='ignore')
    return X

def engineer_features(X):
    X = X.copy()
    if all(c in X.columns for c in ['Weight (kg)', 'Height (m)']):
        h = X['Height (m)'].replace(0, np.nan)
        X['BMI'] = X['Weight (kg)'] / (h**2)
        X['BMI'] = X['BMI'].replace([np.inf, -np.inf], np.nan)
    if all(c in X.columns for c in ['Total Cholesterol (mg/dL)', 'HDL (mg/dL)']):
        hdl = X['HDL (mg/dL)'].replace(0, np.nan).fillna(1)
        X['TC_HDL_Ratio'] = np.clip(X['Total Cholesterol (mg/dL)'] / hdl, 0, 30)
        X['Non_HDL'] = X['Total Cholesterol (mg/dL)'] - X['HDL (mg/dL)']
    if all(c in X.columns for c in ['Systolic BP', 'Diastolic BP']):
        X['Pulse_Pressure'] = X['Systolic BP'] - X['Diastolic BP']
        X['MAP'] = X['Diastolic BP'] + X['Pulse_Pressure']/3
    if 'Age' in X.columns and 'Systolic BP' in X.columns:
        X['Age_SBP_Interaction'] = X['Age'] * X['Systolic BP']
    if 'BMI' in X.columns:
        X['Obese'] = (X['BMI'] >= 30).astype(float)
    if 'Systolic BP' in X.columns:
        X['Hypertensive'] = (X['Systolic BP'] >= 140).astype(float)
    return X

def drop_string_columns(X):
    X = X.copy()
    string_cols = X.select_dtypes(include=['object', 'string']).columns
    if len(string_cols) > 0:
        print(f"Dropping remaining string columns: {list(string_cols)}")
        X = X.drop(columns=string_cols)
    return X

# ====================== ENHANCED FEATURE ENGINEERING ======================
def enhanced_engineer_features(X):
    X = engineer_features(X)  # Keep original features
    
    # Add only the MOST impactful new features
    if all(c in X.columns for c in ['Age', 'Systolic BP']):
        X['Age_BP_Risk'] = X['Age'] * (X['Systolic BP'] - 120).clip(0, None) / 100
    
    if all(c in X.columns for c in ['Age', 'BMI']):
        X['Age_BMI_Risk'] = X['Age'] * (X['BMI'] - 25).clip(0, None) / 100
    
    if all(c in X.columns for c in ['Smoking Status', 'Diabetes Status', 'Family History of CVD']):
        X['Total_Risk_Factors'] = X['Smoking Status'] + X['Diabetes Status'] + X['Family History of CVD']
    
    # Simple but effective: Risk score based on key factors
    risk_score = 0
    if 'Age' in X.columns:
        risk_score += (X['Age'] > 50).astype(int)
    if 'Systolic BP' in X.columns:
        risk_score += (X['Systolic BP'] > 140).astype(int)
    if 'Diabetes Status' in X.columns:
        risk_score += X['Diabetes Status']
    if 'Smoking Status' in X.columns:
        risk_score += X['Smoking Status']
    
    X['Simple_Risk_Score'] = risk_score
    
    return X

# ====================== STRATEGY 1: YOUR ORIGINAL PIPELINE ======================
print("\n" + "="*85)
print("STRATEGY 1: YOUR ORIGINAL PIPELINE (Baseline: 79.33%)")
print("="*85)

original_pipeline = Pipeline([
    ('encode', FunctionTransformer(full_categorical_encoding, validate=False)),
    ('parse',  FunctionTransformer(parse_bp_height, validate=False)),
    ('engineer', FunctionTransformer(engineer_features, validate=False)),
    ('drop_strings', FunctionTransformer(drop_string_columns, validate=False)),
    ('impute', IterativeImputer(estimator=BayesianRidge(), max_iter=50, random_state=SEED)),
    ('scale', StandardScaler()),
    ('model', StackingClassifier(
        estimators=[
            ('cat', CatBoostClassifier(iterations=1200, depth=7, learning_rate=0.03, 
                                       verbose=False, random_state=SEED)),
            ('xgb', XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.03, 
                                  random_state=SEED, use_label_encoder=False, eval_metric='logloss')),
            ('lgb', LGBMClassifier(n_estimators=1000, max_depth=8, learning_rate=0.03, 
                                   random_state=SEED, verbose=-1))
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3,
        n_jobs=-1,
        passthrough=False
    ))
])

# ====================== STRATEGY 2: ENHANCED PIPELINE ======================
print("\n" + "="*85)
print("STRATEGY 2: ENHANCED PIPELINE WITH SMART FEATURES")
print("="*85)

enhanced_pipeline = Pipeline([
    ('encode', FunctionTransformer(full_categorical_encoding, validate=False)),
    ('parse',  FunctionTransformer(parse_bp_height, validate=False)),
    ('engineer', FunctionTransformer(enhanced_engineer_features, validate=False)),
    ('drop_strings', FunctionTransformer(drop_string_columns, validate=False)),
    ('impute', IterativeImputer(estimator=BayesianRidge(), max_iter=50, random_state=SEED)),
    ('scale', StandardScaler()),
    ('model', StackingClassifier(
        estimators=[
            ('cat', CatBoostClassifier(iterations=1500, depth=6, learning_rate=0.025,
                                      l2_leaf_reg=3, verbose=False, random_state=SEED)),
            ('xgb', XGBClassifier(n_estimators=1200, max_depth=5, learning_rate=0.025,
                                 subsample=0.8, colsample_bytree=0.8,
                                 random_state=SEED, use_label_encoder=False)),
            ('lgb', LGBMClassifier(n_estimators=1200, max_depth=7, learning_rate=0.025,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=SEED, verbose=-1))
        ],
        final_estimator=LogisticRegression(max_iter=1000, C=0.1, random_state=SEED),
        cv=5,
        n_jobs=-1,
        passthrough=True  # Keep original features for meta-learner
    ))
])

# ====================== STRATEGY 3: OPTIMIZED XGBOOST ONLY ======================
print("\n" + "="*85)
print("STRATEGY 3: HEAVILY OPTIMIZED XGBOOST")
print("="*85)

xgb_optimized = Pipeline([
    ('encode', FunctionTransformer(full_categorical_encoding, validate=False)),
    ('parse',  FunctionTransformer(parse_bp_height, validate=False)),
    ('engineer', FunctionTransformer(enhanced_engineer_features, validate=False)),
    ('drop_strings', FunctionTransformer(drop_string_columns, validate=False)),
    ('impute', IterativeImputer(estimator=BayesianRidge(), max_iter=50, random_state=SEED)),
    ('scale', StandardScaler()),
    ('model', XGBClassifier(
        n_estimators=2000,  # More trees but shallower
        max_depth=4,        # Prevent overfitting
        learning_rate=0.01, # Slower learning
        subsample=0.7,      # Dropout
        colsample_bytree=0.7,
        reg_alpha=0.5,      # L1 regularization
        reg_lambda=1.0,     # L2 regularization
        min_child_weight=5,
        gamma=0.1,
        random_state=SEED,
        use_label_encoder=False,
        eval_metric='logloss'
    ))
])

# ====================== CROSS-VALIDATION COMPARISON ======================
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

pipelines = {
    'Original (79.33% baseline)': original_pipeline,
    'Enhanced Stacking': enhanced_pipeline,
    'Optimized XGBoost': xgb_optimized
}

print("\nRunning 10-fold cross-validation...")
results = {}

for name, pipeline in pipelines.items():
    print(f"\nEvaluating {name}...")
    
    try:
        acc_scores = cross_val_score(pipeline, X, y, cv=skf, 
                                    scoring='accuracy', n_jobs=1)  # Reduced to 1 for memory
        auc_scores = cross_val_score(pipeline, X, y, cv=skf,
                                    scoring='roc_auc', n_jobs=1)
        
        results[name] = {
            'accuracy_mean': acc_scores.mean(),
            'accuracy_std': acc_scores.std(),
            'auc_mean': auc_scores.mean(),
            'auc_std': auc_scores.std()
        }
        
        print(f"  Accuracy: {acc_scores.mean():.4f} (±{acc_scores.std():.4f})")
        print(f"  ROC-AUC:  {auc_scores.mean():.4f} (±{auc_scores.std():.4f})")
        
    except Exception as e:
        print(f"  Error: {e}")
        results[name] = None

# ====================== IDENTIFY BEST PIPELINE ======================
valid_results = {k: v for k, v in results.items() if v is not None}
if valid_results:
    best_model_name = max(valid_results, key=lambda x: valid_results[x]['accuracy_mean'])
    best_result = valid_results[best_model_name]
    
    print("\n" + "="*85)
    print("BEST MODEL IDENTIFIED")
    print("="*85)
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Accuracy: {best_result['accuracy_mean']:.4f} (±{best_result['accuracy_std']:.4f})")
    print(f"   ROC-AUC:  {best_result['auc_mean']:.4f} (±{best_result['auc_std']:.4f})")
    
    # Compare with baseline
    baseline_acc = 0.7933
    improvement = ((best_result['accuracy_mean'] - baseline_acc) / baseline_acc) * 100
    print(f"\n📈 Improvement over baseline ({baseline_acc:.4f}): {improvement:+.2f}%")
    
    # ====================== TRAIN BEST MODEL ======================
    print("\n" + "="*85)
    print("TRAINING FINAL BEST MODEL")
    print("="*85)
    
    best_pipeline = pipelines[best_model_name]
    final_model = best_pipeline.fit(X, y)
    print(f"✅ Best model trained on all data")
    
    # ====================== FINAL EVALUATION WITH THRESHOLD OPTIMIZATION ======================
    print("\n" + "="*85)
    print("FINAL MODEL OPTIMIZATION")
    print("="*85)
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix
    
    # Split for threshold optimization
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, 
                                                     random_state=SEED, stratify=y)
    
    # Train on 80% for threshold tuning
    temp_pipeline = Pipeline(best_pipeline.steps[:-1])  # All but model
    temp_pipeline.fit(X_train)
    
    # Get the actual model
    model_steps = dict(best_pipeline.named_steps)
    model = model_steps['model']
    
    # Transform the data
    X_train_transformed = temp_pipeline.transform(X_train)
    X_val_transformed = temp_pipeline.transform(X_val)
    
    # Train model
    model.fit(X_train_transformed, y_train)
    
    # Optimize threshold
    y_val_proba = model.predict_proba(X_val_transformed)[:, 1]
    thresholds = np.arange(0.3, 0.7, 0.01)
    f1_scores = []
    
    for th in thresholds:
        y_pred = (y_val_proba >= th).astype(int)
        f1_scores.append(f1_score(y_val, y_pred))
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    print(f"Optimal threshold for F1: {optimal_threshold:.3f}")
    print(f"Optimal F1 score: {optimal_f1:.4f}")
    
    # Test with optimal threshold on validation set
    y_val_pred_opt = (y_val_proba >= optimal_threshold).astype(int)
    val_acc_opt = accuracy_score(y_val, y_val_pred_opt)
    val_bal_opt = balanced_accuracy_score(y_val, y_val_pred_opt)
    
    print(f"\nValidation with optimal threshold ({optimal_threshold:.3f}):")
    print(f"  Accuracy: {val_acc_opt:.4f}")
    print(f"  Balanced Accuracy: {val_bal_opt:.4f}")
    
    # ====================== VISUALIZATION ======================
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Model Comparison
    plt.subplot(2, 2, 1)
    model_names = list(valid_results.keys())
    acc_means = [valid_results[name]['accuracy_mean'] for name in model_names]
    colors = ['skyblue', 'lightgreen', 'salmon']
    
    bars = plt.bar(range(len(model_names)), acc_means, color=colors, alpha=0.8)
    plt.axhline(y=baseline_acc, color='red', linestyle='--', 
               label=f'Baseline ({baseline_acc:.4f})', alpha=0.7)
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison (10-Fold CV)')
    plt.xticks(range(len(model_names)), [name.split()[0] for name in model_names])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, acc_means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', fontsize=9)
    
    # Plot 2: Threshold Optimization
    plt.subplot(2, 2, 2)
    plt.plot(thresholds, f1_scores, 'b-', linewidth=2)
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', 
               label=f'Optimal: {optimal_threshold:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('Threshold Optimization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Performance Metrics
    plt.subplot(2, 2, 3)
    metrics = ['CV Accuracy', 'Val Accuracy', 'F1 Score', 'Improvement']
    values = [best_result['accuracy_mean'], val_acc_opt, optimal_f1, improvement]
    colors_metrics = ['blue', 'green', 'orange', 'red' if improvement < 0 else 'green']
    
    bars2 = plt.bar(metrics, values, color=colors_metrics, alpha=0.7)
    plt.ylabel('Score')
    plt.title(f'Best Model: {best_model_name.split()[0]}')
    plt.grid(True, alpha=0.3)
    
    for bar, val in zip(bars2, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', fontsize=9)
    
    # Plot 4: Confusion Matrix with optimal threshold
    plt.subplot(2, 2, 4)
    cm = confusion_matrix(y_val, y_val_pred_opt)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (Threshold: {optimal_threshold:.2f})')
    plt.colorbar()
    
    classes = ['Low Risk', 'High Risk']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig('final_model_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✅ Analysis visualization saved as 'final_model_analysis.png'")
    
    # ====================== SAVE FINAL MODEL WITH THRESHOLD ======================
    model_to_save = {
        'pipeline': final_model,
        'optimal_threshold': optimal_threshold,
        'preprocessing_functions': {
            'full_categorical_encoding': full_categorical_encoding,
            'parse_bp_height': parse_bp_height,
            'engineer_features': enhanced_engineer_features if 'Enhanced' in best_model_name else engineer_features,
            'drop_string_columns': drop_string_columns
        },
        'performance': {
            'cv_accuracy': float(best_result['accuracy_mean']),
            'cv_accuracy_std': float(best_result['accuracy_std']),
            'cv_roc_auc': float(best_result['auc_mean']),
            'validation_accuracy': float(val_acc_opt),
            'optimal_f1': float(optimal_f1),
            'improvement_over_baseline': float(improvement)
        }
    }
    
    # Create filename
    if improvement > 0:
        filename = f'CVD_HighRisk_{best_result["accuracy_mean"]:.4f}_IMPROVED.pkl'
    else:
        filename = f'CVD_HighRisk_{best_result["accuracy_mean"]:.4f}_BEST_EFFORT.pkl'
    
    joblib.dump(model_to_save, filename)
    print(f"\n✅ FINAL MODEL SAVED AS: '{filename}'")
    
    # ====================== FINAL SUMMARY ======================
    print("\n" + "="*85)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*85)
    print(f"\n1. Baseline Accuracy:        {baseline_acc:.4f}")
    print(f"2. Best Model:              {best_model_name}")
    print(f"3. CV Accuracy Achieved:    {best_result['accuracy_mean']:.4f}")
    print(f"4. Validation Accuracy:     {val_acc_opt:.4f}")
    print(f"5. Optimal Threshold:       {optimal_threshold:.3f}")
    print(f"6. Optimal F1 Score:        {optimal_f1:.4f}")
    print(f"7. Improvement:             {improvement:+.2f}%")
    
    if val_acc_opt > baseline_acc:
        print(f"\n🎉 SUCCESS! Validation accuracy improved to {val_acc_opt:.4f}")
    else:
        print(f"\n📊 Note: Using threshold {optimal_threshold:.3f} gives F1={optimal_f1:.4f}")
    
    # Show the plots
    plt.show()
    
else:
    print("\n❌ No valid models to compare. Check for errors above.")

print("\n" + "="*85)
print("RECOMMENDATIONS IF STILL NOT IMPROVED:")
print("="*85)
print("1. Check data quality - ensure no data leakage")
print("2. Try collecting more samples if possible")
print("3. Consider more advanced feature engineering")
print("4. Try neural networks or deep learning")
print("5. Ensemble multiple different algorithms")
print("6. Use automated hyperparameter tuning (Optuna, Hyperopt)")