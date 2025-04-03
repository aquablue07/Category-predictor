# Random forest with regularization

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv('/Users/vishwa/final_spacy.csv')

numerical_cols = ['Price', 'Weight_lbs', 'Volume']
categorical_cols = ['Manufacturer', 'SKU']
text_col = 'Processed_Text'

# Nan handling
num_imputer = SimpleImputer(strategy='median')
df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
df[categorical_cols] = df[categorical_cols].fillna('missing')
df[text_col] = df[text_col].fillna('')
df = df.dropna(subset=['Category'])

# Shuffle the DataFrame before splitting
df = df.sample(frac=1, random_state=42).reset_index(drop=True).  #shuffle data

# Separate features and target
X = df.drop('Category', axis=1)
y = df['Category']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7, shuffle=True)

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols),
        ('text', TfidfVectorizer(max_features=1000), text_col)
    ])

# Model Pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=7, n_estimators=400, class_weight='balanced',  # 100-200 is the sweet spot
                                           max_depth=10, min_samples_split=2, min_samples_leaf=2)) #added regularization
])

# Train model
model.fit(X_train, y_train)

train_accuracy = model.score(X_train, y_train)
val_accuracy = model.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Validation Accuracy: {val_accuracy:.2f}")

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=7)
cv_scores = cross_val_score(model, X, y, cv=kf)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.2f}")

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
