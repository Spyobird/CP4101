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
from scipy.spatial.distance import pdist, squareform
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
  with np.load(f"datasets/{id}/split{split:02d}.npz", allow_pickle=True) as data:
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    train = data["train"]
    test = data["test"]
    return X_train, X_test, y_train, y_test, train, test
  
# save a dataframe
def save_dataframe(id, path, name, opt, df):
  if not os.path.exists(f"results/{path}/{id}{opt}"):
    print(f"[{id}] Creating directory for results")
    os.makedirs(f"results/{path}/{id}{opt}", exist_ok=True)

  print(f"[{id}] Saving dataframe to csv")
  df.to_csv(f"results/{path}/{id}{opt}/{name}.csv")
  print(f"[{id}] Dataframe saved")

# fetch best gamma
def fetch_best_gamma(id):
  print(f"[{id}] Fetching best gamma")
  bayes_search = load_baseline_model(id, "bayes")

   # Best gamma
  best_gamma = bayes_search.best_params_["gamma"]
  print(f"[{id}] Best gamma: {best_gamma}")
  return best_gamma

# load a model
def load_baseline_model(id, name):
  print(f"[{id}] Loading model")
  model = load(f"models/baseline/{id}/{name}.joblib")
  print(f"[{id}] Model loaded")
  return model

# save a model
def save_lssm_model(id, model, name):
  if not os.path.exists(f"models/local_lssm/{id}"):
    print(f"[{id}] Creating directory for models")
    os.makedirs(f"models/local_lssm/{id}", exist_ok=True)

  print(f"[{id}] Saving model")
  dump(model, f"models/local_lssm/{id}/{name}.joblib")
  print(f"[{id}] Model saved")

def save_numpy(path, name, arr):
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
  np.save(f"{path}/{name}", arr)

# calculate local sets
def local_sets(X, y):
  # calculate distances between all points: O(n^2)
  distances = pairwise_distances(X)

  # find nearest enemies: O(n^2)
  ne = np.zeros(y.size)
  ne_dist = np.zeros(y.size)
  for i in range(len(y)):
    ne_dist[i] = np.inf
    for j in range(len(y)):
      if i == j:
        continue
      if y[i] == y[j]:
        continue
      if distances[i][j] < ne_dist[i]:
        ne_dist[i] = distances[i][j]
        ne[i] = j
  
  # create local sets: O(n^2)
  ls = [set() for i in range(len(y))]
  for i in range(len(y)):
    for j in range(len(y)):
      if i == j:
        continue
      if distances[i][j] < ne_dist[i]:
        ls[i].add(j)

  return distances, ne, ne_dist, ls

# create S (ref: https://github.com/waashk/instanceselection/blob/main/src/main/python/iSel/lssm.py)
def LSSm(X, y, distances, ne, ne_dist, ls):
  u = np.zeros(y.size)
  h = np.zeros(y.size)
  mask = np.zeros(y.size, dtype=bool)

  # calculate u and h
  for i in range(len(y)):
    # calculate usefulness
    for j in range(len(y)):
      count = 0
      if i == j:
        continue
      if i in ls[j]:
        count += 1
    u[i] = count

    # calculate harmfulness
    h[i] = len(np.where(ne == i)[0])

  # check criteria
  for i in range(len(y)):
    if u[i] >= h[i]:
      mask[i] = True
  
  X_s = np.asarray(X[mask])
  y_s = np.asarray(y[mask])

  return X_s, y_s


## main code

def process(id, gamma, knn=False, random_state=4101):
  print(f"[{id}] Begin LSSm")

  info = {}

  timings = np.zeros((100, 3))
  for split in range(100):

    print(f"[{id}:{split}] Begin fitting split")
    # load dataset split
    X_train, X_test, y_train, y_test, train, test = load_dataset_split(id, split)

    # get local sets
    print(f"[{id}:{split}] Calculating local sets")
    start = time.process_time()
    distances, ne, ne_dist, ls = local_sets(X_train, y_train)
    end = time.process_time()
    exec_time = end - start
    timings[split,0] = exec_time
    print(f"[{id}:{split}] Completed local sets")

    # Construct reduced set
    print(f"[{id}:{split}] Constructing subset")
    start = time.process_time()
    X_s, y_s = LSSm(X_train, y_train, distances, ne, ne_dist, ls)
    end = time.process_time()
    exec_time = end - start
    timings[split,1] = exec_time

    if knn:
      clf = KNeighborsClassifier(n_neighbors=5)
    else:
      clf = SVC(C=1, gamma=gamma, cache_size=1000)

    print(f"[{id}:{split}] Fit model")
    start = time.process_time()
    clf.fit(X_s, y_s)
    end = time.process_time()
    exec_time = end - start
    timings[split,2] = exec_time
    print(f"[{id}:{split}] Model fitting completed")

    print(f"[{id}:{split}] Save fitted model")
    if knn:
      save_lssm_model(id, clf, f"model_knn_{id}_split{split:02d}")
    else:
      save_lssm_model(id, clf, f"model_{id}_split{split:02d}")

    if not os.path.exists(f"datasets/LSSm/{id}"):
      print(f"[{id}:{split}] Creating directory for splits")
      os.makedirs(f"datasets/LSSm/{id}", exist_ok=True)

    print(f"[{id}:{split}] Saving split {split}")
    np.savez(f"datasets/LSSm/{id}/split{split:02d}", X_train=X_s, y_train=y_s,
            X_test=X_test, y_test=y_test)
  
  if knn:
    save_numpy(f"results/lssm/{id}", "timings_knn", timings)
  else:
    save_numpy(f"results/lssm/{id}", "timings", timings)
  
  print(f"[{id}] End LSSm")

def main(args):
  if args.id:
    id = args.id
    print(f"LOCAL SET BASED SMOOTHER {id}")
    gamma = fetch_best_gamma(id)
    process(id, gamma, args.knn)
    print(f"COMPLETED {id}")
  
def get_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('id', type=int, help='OpenML dataset ID to process')
  parser.add_argument('--knn', action='store_true')
  return parser.parse_args()

if __name__ == "__main__":
  args = get_arguments()
  main(args)