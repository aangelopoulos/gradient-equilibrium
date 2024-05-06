import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import pdb

classifier = "gradient_boosting"
if classifier == "gradient_boosting":
    classifier_function = HistGradientBoostingClassifier(max_depth=5)
elif classifier == "logistic_regression":
    classifier_function = LogisticRegression(max_iter=10)
else:
    raise ValueError(f"Invalid classifier: {classifier}")

# Load the data
data = pd.read_csv('./raw_data/application_train.csv')

columns = data.columns

# Separate features and target
X = data.drop('TARGET', axis=1)
y = data['TARGET']

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Create transformers for numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine transformers into a preprocessor with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Choose classifier

# Create a pipeline with preprocessor and a classifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier_function)])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Fine-tune the model online using a growing dataset
y_predict_proba = []
idxs = [ len(X_test) // 11 * i for i in range(11) ]
idxs[-1] = len(X_test)
for i in tqdm(range(10)):
    idx = idxs[i]
    _X_train = pd.concat([X_train, X_test[:idx]])
    _y_train = pd.concat([y_train, y_test[:idx]])
    model.fit(_X_train, _y_train)
    y_predict_proba.append(model.predict_proba(X_test[idx:idxs[i+1]])[:, 1])
y_predict_proba = np.concatenate(y_predict_proba)

# Save the X_test, y_test, y_predict_proba as a dataframe, with the original columns plus 'target' and 'prediction' columns
os.makedirs('./.cache', exist_ok=True)
df = pd.DataFrame(np.column_stack([X_test, y_test, y_predict_proba]), columns=list(X_test.columns) + ['target', 'f'])
df.to_pickle(f"./.cache/{classifier}.pkl")