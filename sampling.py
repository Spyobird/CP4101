# import relevant libraries
# data management
import numpy as np
import pandas as pd

# helper
import argparse
import math
import os
import statistics
import time
from tqdm import tqdm

# plotting
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt

# scipy
from scipy.spatial.distance import squareform
from scipy.stats import wilcoxon

# scikit-learn
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score, f1_score, pairwise_distances, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# scikit-optimize
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real

# persistence
from joblib import dump, load

# OpenML
import openml as oml
from openml.datasets import get_dataset

## HELPER FUNCTIONS

# load CV splits individually
def load_dataset_split(id, split):
  print(f"[{id}:{split}] Loading dataset split")
  with np.load(f"datasets/{id}/split{split:02d}.npz") as data:
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    train = data["train"]
    test = data["test"]
    return X_train, X_test, y_train, y_test, train, test

# save a model
def save_sampling_model(id, nu, model, name):
  if not os.path.exists(f"models/sampling/{id}/{nu}"):
    print(f"[{id}] Creating directory for models")
    os.makedirs(f"models/sampling/{id}/{nu}", exist_ok=True)

  print(f"[{id}] Saving model")
  dump(model, f"models/sampling/{id}/{nu}/{name}.joblib")
  print(f"[{id}] Model saved")

# load a model
def load_baseline_model(id, name):
  print(f"[{id}] Loading model")
  model = load(f"models/baseline/{id}/{name}.joblib")
  print(f"[{id}] Model loaded")
  return model

## FITTING CODE

# fetch best gamma
def fetch_best_gamma(id):
  print(f"[{id}] Fetching best gamma")
  bayes_search = load_baseline_model(id, "bayes")

   # Best gamma
  best_gamma = bayes_search.best_params_["gamma"]
  print(f"[{id}] Best gamma: {best_gamma}")
  return best_gamma

def fit_sampling_models(id, gamma, random_state=4101):
  print(f"[{id}] Begin fitting sampling models")

  for split in range(100):
    print(f"[{id}:{split}] Begin fitting split")
    # load dataset split
    X_train, X_test, y_train, y_test, train, test = load_dataset_split(id, split)

    # set seed
    print(f"[{id}:{split}] Setting random seed")
    current_seed = (random_state + hash(split * 27)) % 2**16
    np.random.seed(current_seed)
    random_idx = np.arange(len(y_train))
    np.random.shuffle(random_idx)

    # trying all proportions in intervals of 5%
    for i in range(19):
      nu = (i + 1) * 5
      print(f"[{id}:{split}:{nu}] Begin fitting at proportion")
      m = int(nu / 100 * len(y_train))

      print(f"[{id}:{split}:{nu}] Choosing subset")
      X_s = X_train[random_idx[:m]]
      y_s = y_train[random_idx[:m]]

      clf = SVC(C=1, gamma=gamma)

      print(f"[{id}:{split}:{nu}] Fit model")
      clf.fit(X_s, y_s)
      print(f"[{id}:{split}:{nu}] Model fitting completed")

      print(f"[{id}:{split}] Save fitted model")
      save_sampling_model(id, nu, clf, f"model_{id}_split{split:02d}")

      print(f"[{id}:{split}:{nu}] Completed proportion")

    print(f"[{id}:{split}] Completed split")

  print(f"[{id}] End fitting sampling models")

def main(args):
  if args.id:
    id = args.id
    print(f"SAMPLING STRATEGY {id}")
    gamma = fetch_best_gamma(id)
    fit_sampling_models(id, gamma)
    print(f"COMPLETED {id}")
  
def get_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('id', type=int, help='OpenML dataset ID to process')
  return parser.parse_args()

if __name__ == "__main__":
  args = get_arguments()
  main(args)