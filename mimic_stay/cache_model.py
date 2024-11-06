# %%
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_array
from tqdm import tqdm
import pdb
import argparse
import mimic_utils

# Get the parent directory of this file
current_directory = os.path.dirname(os.path.abspath(__file__))

# Get the data directory
data_directory = os.path.join(current_directory, 'data')

mimic_utils.process_mimic_data(data_directory)

df = pd.read_csv(data_directory + '/processed_mimic_data.csv')

# Split the 'diagnoses' column by the separator '<sep>'
df['diagnoses'] = df['diagnoses'].str.split(' <sep> ')
df['procedure'] = df['procedure'].str.split(' <sep> ')
df['admittime'] = pd.to_datetime(df['admittime'])
df['dischtime'] = pd.to_datetime(df['dischtime'])
temp_df = df[['diagnoses', 'procedure', 'subject_id']]
df = df.drop(['diagnoses', 'procedure'], axis=1)

# Explode the list-like column into rows
temp_df = temp_df.explode('diagnoses')
temp_df = temp_df.explode('procedure')

# Create one-hot encoding for top 10
top_diagnoses = temp_df.diagnoses.value_counts().head(50).index
temp_df.diagnoses[~temp_df.diagnoses.isin(top_diagnoses)] = 'other diagnosis'
top_procedures = [c for c in temp_df.procedure.value_counts().head(50).index if c not in top_diagnoses]
temp_df.procedure[~temp_df.procedure.isin(top_procedures)] = 'other procedure'

temp_df = pd.get_dummies(temp_df, columns=['diagnoses', 'procedure'], prefix='', prefix_sep='')

# Convert to floats
bool_cols = temp_df.select_dtypes(include='bool').columns
temp_df = temp_df.astype({
    c : float for c in bool_cols
})

# %%

# re-collapse by subject_id
temp_df = temp_df.groupby(['subject_id']).max(numeric_only=True)

df = df.merge(temp_df, on="subject_id", how="inner")

# Calculate the percentage of NaN values in each column
na_percentage = df.isnull().sum() / df.shape[0]

# Find columns with more than 90% NaN values
drop_cols = na_percentage[na_percentage > .9].index.tolist()

# Remove the columns from the dataframe
df = df.drop(columns=drop_cols)

print(df.head())

df['length_of_stay'] = df.dischtime - df.admittime
df['length_of_stay_float'] = (df.dischtime - df.admittime).dt.days
df['dischtime_float'] = (df.dischtime - df.admittime.min()) / np.timedelta64(1, 'D')
df['admittime_float'] = (df.admittime - df.admittime.min()) / np.timedelta64(1, 'D')

df.sort_values(by='admittime', inplace=True)

columns = df.columns
# Clean up for prediction
df = df[df.ethnicity.isin(['ASIAN', 'WHITE', 'BLACK/AFRICAN AMERICAN', 'HISPANIC OR LATINO'])]
# Cut rows with nan
df = df.reset_index().drop("index", axis=1)

# %%

regressor = "gradient_boosting"
if regressor == "gradient_boosting":
    regressor_function = HistGradientBoostingRegressor(max_depth=20)
elif regressor == "logistic_regression":
    regressor_function = LogisticRegression(max_iter=10)
else:
    raise ValueError(f"Invalid regrssor: {regressor}")

# Separate features and target
X_test = df.drop(['length_of_stay_float', 'mortality', 'readmission', 'admittime', 'dischtime', 'dischtime_float', 'admittime_float', 'length_of_stay'], axis=1)
y_test = df['length_of_stay_float']

# Identify numeric and categorical columns
numeric_features = X_test.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_test.select_dtypes(include=['object']).columns

# Fit the encoder to all possible categories
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(X_test[categorical_features])

# Create transformers for numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', encoder)])

# Combine transformers into a preprocessor with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)],
        remainder='passthrough')

# Create a pipeline with preprocessor and a regressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', regressor_function)])

# Fine-tune the model online using a growing dataset
fs = []
idxs = [ len(X_test) // 12 * i for i in range(12) ]
idxs = idxs[1:]
idxs[-1] = len(X_test)

for i in tqdm(range(10)):
    idx = idxs[i]
    _X_train = X_test[:idx]
    _y_train = y_test[:idx]
    
    model.fit(_X_train, _y_train)
    preds = model.predict(X_test[idx:idxs[i+1]])
    preds_mse = np.mean(np.square(preds-y_test[idx:idxs[i+1]]))
    mean_mse = np.mean(np.square(y_test[idx:idxs[i+1]].mean()-y_test[idx:idxs[i+1]]))
    print(i, preds_mse, mean_mse)
    fs.append(preds)
    
fs = np.concatenate(fs)
# Fill in zeros for the first batch of predictions
fs = np.concatenate([np.zeros(len(X_test) - len(fs)), fs])

# Save the fs in the dataframe
os.makedirs('./.cache', exist_ok=True)
df['f'] = fs
print(df)
df.to_pickle(f"./.cache/{regressor}.pkl")
# %%
