# Gradient Equilibrium in Online Learning
## Theory and Applications
### Anastasios N. Angelopoulos, Michael I. Jordan, and Ryan J. Tibshirani

This repository includes code for reproducing our experiments.

To install the requirements for this code, ideally within a virtual environment, run
```
pip install -r requirements.txt
```

To download the required data for reproducing the experiments, run
```
python setup.py
```

## Repository Structure

The core algorithms, including gradient descent, mirror descent, and the model debiasing wrappers, are in `core/algorithms.py`. The file is short and should be easy to parse.

All other folders are dedicated to experiments. The most important files in each folder are a set of Jupyter notebooks in the root directory, each of which is dedicated to a separate experiment (they are largely self-explanatory). Each folder includes also a `plots` folder for saved PDF files of plots. 

`arena/` reproduces the Chatbot Arena experiments. The notebooks can be run in any order.

`compas/` reproduces the COMPAS experiments. The notebooks can be run in any order.

`helpsteer/` reproduces the HelpSteer2 experiments. From this directory, you can run `debias.ipynb` to reproduce our plots immediately, or `python train_and_generate_rewards.py` to reproduce the reward model training.

`mimic_stay/` reproduces the MIMIC-IV length-of-stay experiments. This dataset cannot be fully open-sourced due to patient confidentiality issues. The dataset is available at [this link](https://physionet.org/content/mimiciv/3.1/). To reproduce the MIMIC IV predictions, perform the required trainings and then download the files {`admissions.csv`, `diagnoses_icd.csv`, `patients.csv`, `procedures_icd.csv`} into the folder `mimic_stay/raw_data`. Then run `cache_model.py`. Thereafter, run any of the notebooks to reproduce the experiments.

`simulation/` reproduces experiments on simulated data.
