# Xgboost + Regularization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('/Users/vishwa/final_spacy.csv')

# Define columns
numerical_cols = ['Price', 'Weight_lbs', 'Volume']
categorical_cols = ['Manufacturer']
text_col = 'Processed_Text'
target_col = 'Category'

# Ensure numerical columns are numeric.
df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')

# Handle missing values
df[numerical_cols] = SimpleImputer(strategy='median').fit_transform(df[numerical_cols])
df[categorical_cols] = df[categorical_cols].fillna('missing')
df[text_col] = df[text_col].fillna('')
df = df.dropna(subset=[target_col])

# Encode target variable
label_encoder = LabelEncoder()
df[target_col] = label_encoder.fit_transform(df[target_col])

# Shuffle DataFrame
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split data
X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols),
        ('text', TfidfVectorizer(max_features=1000), text_col)
    ]
)

# Transform training data before passing to XGBoost
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Ensure correct data format after transformation
print(f"Transformed Train Shape: {X_train_transformed.shape}")
print(f"Transformed Test Shape: {X_test_transformed.shape}")

# Train XGBoost classifier
classifier = XGBClassifier(
    eval_metric='mlogloss',
    random_state=42,
    n_estimators=200,
    early_stopping_rounds=10,
    use_label_encoder=False,
    scale_pos_weight=(len(y_train) - np.sum(y_train)) / np.sum(y_train),  # Xgboost Class Imbalance
    max_depth=5, # Regularization
    subsample=0.8, # Regularization
    colsample_bytree=0.8 # Regularization
)

# Train with early stopping, NTS: early stopping did not work
classifier.fit(
    X_train_transformed, y_train,
    eval_set=[(X_test_transformed, y_test)],
    verbose=False
)

# Predictions
y_pred = classifier.predict(X_test_transformed)

# Convert back to original category labels for evaluation
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Evaluation metrics
print("=== Imbalanced Class Handling ===")
print(f"Class distribution:\n{np.bincount(y_train) / len(y_train)}")
print("\n=== Metrics ===")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_labels))
#print("\nConfusion Matrix:")
# print(confusion_matrix(y_test_labels, y_pred_labels))

# # Cross-validation
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cv_scores = cross_val_score(classifier, X_train_transformed, y_train, cv=cv, scoring='balanced_accuracy')
# print(f"\nCross-Validation Scores: {cv_scores}")
# print(f"Mean CV Balanced Accuracy: {cv_scores.mean():.4f}")
