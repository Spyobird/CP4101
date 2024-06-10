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

# convert variables to np arrays
def variable_as_array(var):
  if var is None:
    return []
  return np.asarray(var)

# check if there are strings or date
def has_strings_or_date(id):
  data = get_dataset(dataset_id=id, download_data=False, download_qualities=False, download_features_meta_data=False)
  if len(data.get_features_by_type("string")) != 0 or len(data.get_features_by_type("date")) != 0:
    return True
  return False

# drop NaN values
def drop_nulls(data):
  data[1].dropna(inplace=True)

# get X and y
def get_Xy(data):
  df = data[1]
  X = df.loc[:, df.columns.drop(data[5])]
  y = df.loc[:, data[5]]
  return X, y

# load CV splits individually
def load_dataset_split(id, split):
  print(f"[{id}:{split}] Loading dataset split")
  with np.load(f"datasets/{id}/split{split:02d}.npz", allow_pickle=True) as data:
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    train = data["train"]
    test = data["test"]
    return X_train, X_test, y_train, y_test, train, test

# save a model
def save_baseline_model(id, model, name):
  if not os.path.exists(f"models/baseline/{id}"):
    print(f"[{id}] Creating directory for models")
    os.makedirs(f"models/baseline/{id}", exist_ok=True)

  print(f"[{id}] Saving model")
  dump(model, f"models/baseline/{id}/{name}.joblib")
  print(f"[{id}] Model saved")

# load a model
def load_baseline_model(id, name):
  print(f"[{id}] Loading model")
  model = load(f"models/baseline/{id}/{name}.joblib")
  print(f"[{id}] Model loaded")
  return model

def save_numpy(path, name, arr):
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
  np.save(f"{path}/{name}", arr)

# fetch data from OpenML
def fetch_data(id):
  data = get_dataset(dataset_id=id, download_data=False, download_qualities=True, download_features_meta_data=True)
  frame, _, mask, names = data.get_data(dataset_format='dataframe')
  mask = np.array(mask)
  names = np.array(names)

  target = data.default_target_attribute

  num_idx = data.get_features_by_type("numeric")
  feature_names = names[num_idx]

  cat_idx = data.get_features_by_type("nominal")
  category_names = []
  category_values = []
  target_names = []
  for i in cat_idx:
    feature = data.features[i]
    if feature.name == target:
      target_names = feature.nominal_values
      continue
    category_names.append(feature.name)
    category_values.append(feature.nominal_values)
  return id, frame, category_names, category_values, feature_names, target, target_names, data.name

## PREPROCESSING CODE

# Preprocess data
def preprocess(id, random_state=4101):
  print(f"[{id}] Begin preprocessing")

  # fetch data
  print(f"[{id}] Fetching data from OpenML")
  data = fetch_data(id)

  # drop NaN values
  print(f"[{id}] Drop null values")
  drop_nulls(data)

  # standardization of numerical features
  scaler = StandardScaler()

  # one-hot encoding of categorical features
  encoder = OneHotEncoder(categories=data[3], drop='if_binary')

  # label encoding
  labeller = LabelEncoder()

  # column transformer
  preprocess = ColumnTransformer([
      ('num', scaler, data[4]),
      ('cat', encoder, data[2]),
  ], n_jobs=-1)

  X, y = get_Xy(data)

  print(f"[{id}] Processing features")
  Xt = preprocess.fit_transform(X)

  print(f"[{id}] Processing labels")
  yt = labeller.fit_transform(y)
  
  print(f"[{id}] End preprocessing")
  return Xt, yt

# Split data and save it (10x10 CV)
def dataset_splitter(id, Xt, yt, random_state=4101):
  print(f"[{id}] Begin data splitting")

  # CV split
  cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=random_state)

  if not os.path.exists(f"datasets/{id}"):
    print(f"[{id}] Creating directory for splits")
    os.makedirs(f"datasets/{id}", exist_ok=True)

  for i, (train, test) in enumerate(cv.split(Xt, yt)):
    X_train = Xt[train]
    y_train = yt[train]
    X_test = Xt[test]
    y_test = yt[test]
    print(f"[{id}] Saving split {i}")
    np.savez(f"datasets/{id}/split{i:02d}", X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test, train=train, test=test)
  
  print(f"[{id}] End data splitting")
  return cv

## HP TUNING CODE

# HP space
param_space = {
    # "C": Real(2e-15, 2e15, prior="log-uniform", base=2, name="C"),
    "gamma": Real(2e-15, 2e15, prior="log-uniform", base=2, name="gamma")
}

def fit_bayes_model(id, Xt, yt, cv, random_state=4101):
  print(f"[{id}] Begin HP tuning")

  # Initialise model
  print(f"[{id}] Initialise SMBO model")
  bayes_search = BayesSearchCV(
    SVC(C=1),
    param_space,
    n_iter=100,
    cv=cv,
    n_jobs=-1,
    refit=False, # probably don't need to refit model
    return_train_score=True,
    random_state=random_state)
  
  # Fit model
  print(f"[{id}] Fit SMBO model")
  bayes_search.fit(Xt, yt)
  print(f"[{id}] Model fitting completed")

  # Best gamma
  best_gamma = bayes_search.best_params_["gamma"]
  print(f"[{id}] Best gamma: {best_gamma}")

  # Save model
  print(f"[{id}] Save SMBO model")
  save_baseline_model(id, bayes_search, "bayes")
  
  print(f"[{id}] End HP tuning")
  return best_gamma

# REFIT CODE

def refit_split_models(id, gamma, knn=False, random_state=4101):
  print(f"[{id}] Begin baseline model refitting")

  timings = np.zeros(100)
  for split in range(100):
    print(f"[{id}:{split}] Begin fitting split")
    # load dataset split
    X_train, X_test, y_train, y_test, train, test = load_dataset_split(id, split)

    if knn:
      clf = KNeighborsClassifier(n_neighbors=5)
    else:
      clf = SVC(C=1, gamma=gamma, cache_size=1000)

    # Fit model
    print(f"[{id}:{split}] Fit model")
    start = time.process_time()
    clf.fit(X_train, y_train)
    end = time.process_time()
    exec_time = end - start
    timings[split] = exec_time
    print(f"[{id}:{split}] Model fitting completed")

    print(f"[{id}:{split}] Save refitted model")
    if knn:
      save_baseline_model(id, clf, f"model_knn_{id}_split{split:02d}")
    else:
      save_baseline_model(id, clf, f"model_{id}_split{split:02d}")

    print(f"[{id}:{split}] Completed split")
  if knn:
    save_numpy(f"results/baseline/{id}", "timings_knn", timings)
  else:
    save_numpy(f"results/baseline/{id}", "timings", timings)
  
  print(f"[{id}:{split}] End baseline model refitting")

def main(args):
  if args.id:
    id = args.id
    print(f"BASELINE PROCESSING {id}")
    Xt, yt = preprocess(id)
    cv = dataset_splitter(id, Xt, yt)
    gamma = fit_bayes_model(id, Xt, yt, cv)
    refit_split_models(id, gamma, args.knn)
    print(f"COMPLETED {id}")
  
def get_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('id', type=int, help='OpenML dataset ID to process')
  parser.add_argument('--knn', action='store_true')
  return parser.parse_args()

if __name__ == "__main__":
  args = get_arguments()
  main(args)