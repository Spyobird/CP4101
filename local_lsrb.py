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
def load_lssm_dataset_split(id, split):
  print(f"[{id}:{split}] Loading dataset split")
  with np.load(f"datasets/LSSm/{id}/split{split:02d}.npz", allow_pickle=True) as data:
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    return X_train, X_test, y_train, y_test
  
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

# save a model
def save_lsrb_model(id, model, name):
  if not os.path.exists(f"models/local_lssm_lsrb/{id}"):
    print(f"[{id}] Creating directory for models")
    os.makedirs(f"models/local_lssm_lsrb/{id}", exist_ok=True)

  print(f"[{id}] Saving model")
  dump(model, f"models/local_lssm_lsrb/{id}/{name}.joblib")
  print(f"[{id}] Model saved")

# load a model
def load_baseline_model(id, name):
  print(f"[{id}] Loading model")
  model = load(f"models/baseline/{id}/{name}.joblib")
  print(f"[{id}] Model loaded")
  return model

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
  ls = [set() for _ in range(len(y))]
  for i in range(len(y)):
    for j in range(len(y)):
      if i == j:
        continue
      if distances[i][j] < ne_dist[i]:
        ls[i].add(j)

  return distances, ne, ne_dist, ls

# create S (ref: https://github.com/waashk/instanceselection/blob/main/src/main/python/iSel/lsbo.py)
def LSRB(X, y, distances, ne, ne_dist, ls):
  lsc = np.zeros(y.size)
  mask = np.zeros(y.size, dtype=bool)

  S = set()

  # calculate LSC
  for i in range(len(y)):
    lsc[i] = len(ls[i])

  # sort by LSC
  lsc_sort = np.argsort(lsc)

  # check intersection and add to set
  for i in lsc_sort:
    count = 0
    for j in S:
      if i in ls[j]:
        count += 1
    if count == 0:
      S.add(i)
      mask[i] = True
  
  X_s = np.asarray(X[mask])
  y_s = np.asarray(y[mask])

  return X_s, y_s

## main code
def process(id, gamma, random_state=4101):
  print(f"[{id}] Begin LSRB")

  info = {}

  for split in range(100):

    print(f"[{id}:{split}] Begin fitting split")
    # load dataset split
    X_train, X_test, y_train, y_test = load_lssm_dataset_split(id, split)

    # get local sets
    print(f"[{id}:{split}] Calculating local sets")
    distances, ne, ne_dist, ls = local_sets(X_train, y_train)
    print(f"[{id}:{split}] Completed local sets")

    # Construct reduced set
    print(f"[{id}:{split}] Constructing subset")
    X_s, y_s = LSRB(X_train, y_train, distances, ne, ne_dist, ls)
    print(f"[{id}:{split}] Completed subset construction")

    info[f"split{split}"] = [len(y_s), len(y_train), len(y_s)/len(y_train)]

    clf = SVC(C=1, gamma=gamma)

    print(f"[{id}:{split}] Fit model")
    clf.fit(X_s, y_s)
    print(f"[{id}:{split}] Model fitting completed")

    print(f"[{id}:{split}] Save fitted model")
    save_lsrb_model(id, clf, f"model_{id}_split{split:02d}")

    print(f"[{id}:{split}] Completed split")

  info_df = pd.DataFrame(info)
  info_df["mean"] = info_df.mean(numeric_only=True, axis=1)
  save_dataframe(id, "lssm_lsrb", "lsrb_info", "", info_df)
  
  print(f"[{id}] End LSRB")

def main(args):
  if args.id:
    id = args.id
    print(f"LOCAL SET REPRESENTATIVE BORDER {id}")
    gamma = fetch_best_gamma(id)
    process(id, gamma)
    print(f"COMPLETED {id}")
  
def get_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('id', type=int, help='OpenML dataset ID to process')
  return parser.parse_args()

if __name__ == "__main__":
  args = get_arguments()
  main(args)