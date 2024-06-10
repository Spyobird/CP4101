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

# save a model
def save_beta_model(id, nu, model, name):
  if not os.path.exists(f"models/beta/{id}/{nu}"):
    print(f"[{id}] Creating directory for models")
    os.makedirs(f"models/beta/{id}/{nu}", exist_ok=True)

  print(f"[{id}] Saving model")
  dump(model, f"models/beta/{id}/{nu}/{name}.joblib")
  print(f"[{id}] Model saved")

# load a model
def load_baseline_model(id, name):
  print(f"[{id}] Loading model")
  model = load(f"models/baseline/{id}/{name}.joblib")
  print(f"[{id}] Model loaded")
  return model

# fetch best gamma
def fetch_best_gamma(id):
  print(f"[{id}] Fetching best gamma")
  bayes_search = load_baseline_model(id, "bayes")

   # Best gamma
  best_gamma = bayes_search.best_params_["gamma"]
  print(f"[{id}] Best gamma: {best_gamma}")
  return best_gamma

## HELPERS for splitting and iteration

# returns a tuple of (a dict of labels as keys and split values as values, the split dictionary of indexes)
def class_splitter(y, values):
  split = {}
  labels = np.unique(y)
  for l in labels:
    split[l] = []
  for i in range(len(y)):
    split[y[i]].append(i)
  for k, v in split.items():
    split[k] = np.array(v)
  return {k: values[v] for k, v in split.items()}, split

class BalancedIterator():
  def __init__(self, sort):
    self.sort = sort
    self.classes = list(sort.keys())
    self.len_classes = list(len(sort[c]) for c in self.classes)
    self.num_classes = len(self.classes)
    self.counters = list(0 for i in range(self.num_classes))
    self.flags = list(True for i in range(self.num_classes))
    self.next_class = 0

  def __iter__(self):
    self.counters = list(0 for i in range(self.num_classes))
    self.flags = list(True for i in range(self.num_classes))
    self.next_class = 0
    return self

  def __next__(self):
    flag = True
    for f in self.flags:
      flag = flag and not f
    if flag:
      raise StopIteration
    while (not self.flags[self.next_class]):
      self.next_class = (self.next_class + 1) % self.num_classes
    x = self.sort[self.classes[self.next_class]][self.counters[self.next_class]]
    self.counters[self.next_class] += 1
    if self.counters[self.next_class] == self.len_classes[self.next_class]:
      self.flags[self.next_class] = False
    self.next_class = (self.next_class + 1) % self.num_classes
    return x

# calculate beta-values
def beta_values(X, y):
  # calculates the invd value of the distance between two vectors
  def invd(distance):
    return 1/(1 + distance)

  features = X
  labels = y

  label_counts = {}
  for label in labels:
    if label not in label_counts:
      label_counts[label] = 1
    else:
      label_counts[label] += 1

  dist_mat = pairwise_distances(features)

  filtered_distances = []
  for i, row in enumerate(dist_mat):
    row_distances = [(dist, labels[j] == labels[i]) for j, dist in enumerate(row) if i != j]
    row_distances.sort(key=lambda x: x[0])
    filtered_distances.append(row_distances)

  beta_values = []
  for i in range(len(dist_mat)):
    denominator = 0.0
    k = label_counts[labels[i]] - 1
    numerator = sum([invd(filtered_distances[i][j][0]) for j in range(k) if filtered_distances[i][j][1]])
    denominator = sum([invd(distance[0]) for distance in filtered_distances[i]])
    beta_values.append(numerator/denominator)

  return np.array(beta_values), dist_mat

## FITTING CODE

def fit_beta_models(id, gamma, random_state=4101):
  print(f"[{id}] Begin fitting beta models")

  for split in range(100):
    print(f"[{id}:{split}] Begin fitting split")
    # load dataset split
    X_train, X_test, y_train, y_test, train, test = load_dataset_split(id, split)

    # construct beta-values
    print(f"[{id}:{split}] Calculating beta-values")
    beta, dist_mat = beta_values(X_train, y_train)
    beta_split, y_split = class_splitter(y_train, beta)
    beta_split_sort = {k: y_split[k][np.argsort(v)] for k, v in beta_split.items()}
    beta_iter = BalancedIterator(beta_split_sort)
    print(f"[{id}:{split}] Completed beta-values")

    X_sorted = X_train[[*beta_iter]]
    y_sorted = y_train[[*beta_iter]]

    # trying all proportions in intervals of 5%
    for i in range(19):
      nu = (i + 1) * 5
      print(f"[{id}:{split}:{nu}] Begin fitting at proportion")
      m = int(nu / 100 * len(y_train))

      print(f"[{id}:{split}:{nu}] Choosing subset")
      # choose reduced set
      X_s = X_sorted[:m]
      y_s = y_sorted[:m]

      clf = SVC(C=1, gamma=gamma)

      print(f"[{id}:{split}:{nu}] Fit model")
      clf.fit(X_s, y_s)
      print(f"[{id}:{split}:{nu}] Model fitting completed")

      print(f"[{id}:{split}] Save fitted model")
      save_beta_model(id, nu, clf, f"model_{id}_split{split:02d}")

      print(f"[{id}:{split}:{nu}] Completed proportion")

    print(f"[{id}:{split}] Completed split")

  print(f"[{id}] End fitting beta models")

def main(args):
  if args.id:
    id = args.id
    print(f"BETA BASELINE STRATEGY {id}")
    gamma = fetch_best_gamma(id)
    fit_beta_models(id, gamma)
    print(f"COMPLETED {id}")
  
def get_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('id', type=int, help='OpenML dataset ID to process')
  return parser.parse_args()

if __name__ == "__main__":
  args = get_arguments()
  main(args)