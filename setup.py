from collections import defaultdict
import json, math, gdown
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm
import os
import requests
pd.options.display.float_format = '{:.2f}'.format

if __name__ == "__main__":
    """
        ARENA DATA
    """
    # Download the data using HTTP request
    url = "https://storage.googleapis.com/arena_external_data/public/clean_battle_20240826_public.json"
    response = requests.get(url)

    os.makedirs('./arena/raw_data', exist_ok=True)
    with open('./arena/raw_data/public_data.json', 'wb') as file:
        file.write(response.content)

    # load the JSON data from the local file
    with open('./arena/raw_data/public_data.json', 'r') as file:
        battles = pd.read_json(file).sort_values(ascending=True, by=["tstamp"])

    # we use anony battles only for leaderboard
    battles = battles[battles["anony"] == True]
    # we de-duplicate top 0.1% redudant prompts
    battles = battles[battles["dedup_tag"].apply(lambda x: x.get("sampled", False))]

    # Convert the unix tstamp column to datetime
    battles['datetime'] = pd.to_datetime(battles['tstamp'], unit='s')

    battles = battles[~battles["winner"].isin(["tie", "tie (bothbad)"])]

    # Sort the battles DataFrame by datetime
    battles_sorted = battles.sort_values('datetime').reset_index(drop=True)

    # Get unique models and create an ordering
    models = pd.unique(battles_sorted[['model_a', 'model_b']].values.ravel('K'))
    models.sort()  # Sort for consistency

    # Create a dictionary to map model names to indices
    model_to_index = {model: index for index, model in enumerate(models)}

    print("Number of unique models:", len(models))
    print("First few models:", models[:5])
    print("model_to_index sample:", dict(list(model_to_index.items())[:5]))

    # Initialize X matrix
    X = np.zeros((len(battles_sorted), len(models)))

    # Populate X matrix
    for i, row in tqdm(enumerate(battles_sorted.itertuples()), total=len(battles_sorted)):
        model_a = row.model_a
        model_b = row.model_b
        
        if model_a not in model_to_index or model_b not in model_to_index:
            print(f"Error at row {i}: model_a = {model_a}, model_b = {model_b}")
            continue
        
        X[i, model_to_index[model_a]] = 1
        X[i, model_to_index[model_b]] = -1
        
    # Create Y vector
    Y = (battles_sorted['winner'] == 'model_a').astype(int).values

    # Datetime vector
    datetimes = battles_sorted['datetime'].values

    # Save models, X, Y, and datetimes to a file
    os.makedirs('./arena/.cache', exist_ok=True)
    np.savez('./arena/.cache/models_X_Y.npz', models=models, X=X, Y=Y, datetimes=datetimes)