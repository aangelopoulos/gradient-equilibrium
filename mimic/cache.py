import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_array
from tqdm import tqdm

# Get the parent directory of this file
current_directory = os.path.dirname(os.path.abspath(__file__))

# Get the data directory
data_directory = os.path.join(current_directory, 'data')

# Get the CSV files
csv_files = [f for f in os.listdir(data_directory) if f.endswith('.csv')]

# Iterate over each CSV file and merge them by subject_id
combined_data = pd.DataFrame()
for file in csv_files:
    file_path = os.path.join(data_directory, file)
    data = pd.read_csv(file_path)
    combined_data = pd.concat([combined_data, data], ignore_index=True)

combined_data.admittime = pd.to_datetime(combined_data.admittime)
combined_data.dischtime = pd.to_datetime(combined_data.dischtime)
combined_data.deathtime = pd.to_datetime(combined_data.deathtime)

# Convert to days
combined_data['length_of_stay'] = (combined_data.dischtime - combined_data.admittime).dt.days
combined_data.deathtime = (combined_data.deathtime - combined_data.admittime.min()) / np.timedelta64(1, 'D')
combined_data.dischtime = (combined_data.dischtime - combined_data.admittime.min()) / np.timedelta64(1, 'D')
combined_data.admittime = (combined_data.admittime - combined_data.admittime.min()) / np.timedelta64(1, 'D')

combined_data.sort_values(by='admittime', inplace=True)

# Clean up for prediction
data = combined_data[['admittime', 'ethnicity', 'marital_status', 'insurance', 'language','length_of_stay']]
# Cut rows with nan
data = data.dropna()
data = data[data.ethnicity.isin(['ASIAN', 'WHITE', 'BLACK/AFRICAN AMERICAN', 'HISPANIC OR LATINO'])]

regressor = "gradient_boosting"
if regressor == "gradient_boosting":
    regressor_function = HistGradientBoostingRegressor(max_depth=5)
elif regressor == "logistic_regression":
    regressor_function = LogisticRegression(max_iter=10)
else:
    raise ValueError(f"Invalid regrssor: {regressor}")

columns = data.columns

# Separate features and target
X_test = data.drop('length_of_stay', axis=1)
y_test = data['length_of_stay']

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

# Choose regressor

# Create a pipeline with preprocessor and a regressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', regressor_function)])

# Fine-tune the model online using a growing dataset
yhats = []
idxs = [ len(X_test) // 12 * i for i in range(12) ]
idxs = idxs[1:]
idxs[-1] = len(X_test)

for i in tqdm(range(10)):
    idx = idxs[i]
    _X_train = X_test[:idx]
    _y_train = y_test[:idx]
    
    model.fit(_X_train, _y_train)
    yhats.append(model.predict(X_test[idx:idxs[i+1]]))
    
yhats = np.concatenate(yhats)
# Fill in zeros for the first batch of predictions
yhats = np.concatenate([np.zeros(len(X_test) - len(yhats)), yhats])

# Save the X_test, y_test, y_predict_proba as a dataframe, with the original columns plus 'target' and 'prediction' columns
os.makedirs('./.cache', exist_ok=True)
df = pd.DataFrame(np.column_stack([X_test, y_test, yhats]), columns=list(X_test.columns) + ['target', 'prediction'])
df.to_pickle(f"./.cache/{regressor}.pkl")