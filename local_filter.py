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

# save a model
def save_local_model(id, model, name):
  if not os.path.exists(f"models/local_filter/{id}"):
    print(f"[{id}] Creating directory for models")
    os.makedirs(f"models/local_filter/{id}", exist_ok=True)

  print(f"[{id}] Saving model")
  dump(model, f"models/local_filter/{id}/{name}.joblib")
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

# calculate local sets
def local_sets(X, y):
  # calculate distances between all points: O(n^2)
  distances = squareform(pdist(X))

  # find closest point from opposite class for each point: O(n^2)
  closest_enemies = []
  for i in range(len(X)):
    label = y[i]
    closest_enemy_dist = math.inf
    closest_enemy = -1
    # find closest enemy for each point: O(n)
    for j in range(len(X)):
      if i == j:
        continue
      other_label = y[j]
      if label == other_label:
        continue
      if distances[i][j] < closest_enemy_dist:
        closest_enemy_dist = distances[i][j]
        closest_enemy = j
    closest_enemies.append(closest_enemy)
  closest_enemies = np.array(closest_enemies)

  # find points from same class within epsilon ball: O(n^2)
  epsilon_neighbours_mat = np.zeros(distances.shape)
  for i in range(len(y)):
    label = y[i]
    closest_enemy_dist = distances[i][closest_enemies[i]]
    for j in range(len(y)):
      other_label = y[j]
      if label != other_label:
        continue
      if distances[i][j] < closest_enemy_dist:
        epsilon_neighbours_mat[i][j] = 1
  
  # get count of epsilon neighbours: O(n)
  epsilon_neighbours_counts = np.sum(epsilon_neighbours_mat, axis=1)

  # find closest point from same class for each point: O(n^2)
  closest_neighbours = []
  for i in range(len(X)):
    label = y[i]
    closest_neighbour_dist = math.inf
    closest_neighbour = -1
    # find closest neighbour for each point: O(n)
    for j in range(len(X)):
      if i == j:
        continue
      other_label = y[j]
      if label != other_label:
        continue
      if distances[i][j] < closest_neighbour_dist:
        closest_neighbour_dist = distances[i][j]
        closest_neighbour = j
    closest_neighbours.append(closest_neighbour)
  closest_neighbours = np.array(closest_neighbours)

  # find points from opposite class within epsilon ball: O(n^2)
  epsilon_enemies_mat = np.zeros(distances.shape)
  for i in range(len(y)):
    label = y[i]
    closest_neighbour_dist = distances[i][closest_neighbours[i]]
    for j in range(len(y)):
      other_label = y[j]
      if label == other_label:
        continue
      if distances[i][j] < closest_neighbour_dist:
        epsilon_enemies_mat[i][j] = 1
  
  # get count of epsilon neighbours: O(n)
  epsilon_enemies_counts = np.sum(epsilon_enemies_mat, axis=1)

  # get size of epsilon ball: O(n)
  epsilon_sizes = np.array([distances[i][closest_enemies[i]] for i in range(len(y))])

  return epsilon_neighbours_counts, epsilon_enemies_counts, epsilon_neighbours_mat, epsilon_enemies_mat, epsilon_sizes

## FITTING CODE

def fit_local_filter_models(id, gamma, random_state=4101):
  print(f"[{id}] Begin fitting local set filter models")

  info = {}

  for split in range(100):

    print(f"[{id}:{split}] Begin fitting split")
    # load dataset split
    X_train, X_test, y_train, y_test, train, test = load_dataset_split(id, split)

    # get local sets
    print(f"[{id}:{split}] Calculating local sets")
    n_c, e_c, n_mat, e_mat, sizes = local_sets(X_train, y_train)
    idx = np.lexsort((sizes, n_c))
    print(f"[{id}:{split}] Completed local sets")

    # Construct reduced set
    print(f"[{id}:{split}] Constructing subset")
    S = []
    for i in range(len(idx)):
      flag = False
      for j in range(len(S)):
        if n_mat[idx[i]][S[j]] == 1:
          flag = True
          break
      if flag:
        continue
      S.append(idx[i])
    X_s = X_train[S]
    y_s = y_train[S]

    info[f"split{split}"] = [len(S), len(y_train)]
    print(f"[{id}:{split}] Completed subset construction")

    clf = SVC(C=1, gamma=gamma)

    print(f"[{id}:{split}] Fit model")
    clf.fit(X_s, y_s)
    print(f"[{id}:{split}] Model fitting completed")

    print(f"[{id}:{split}] Save fitted model")
    save_local_model(id, clf, f"model_{id}_split{split:02d}")

    print(f"[{id}:{split}] Completed split")

  info_df = pd.DataFrame(info)
  info_df["mean"] = info_df.mean(numeric_only=True, axis=1)
  save_dataframe(id, "baseline_local_filter", "info", "", info_df)
  
  print(f"[{id}] End fitting local set filter models")

def main(args):
  if args.id:
    id = args.id
    print(f"LOCAL SET FILTER STRATEGY {id}")
    gamma = fetch_best_gamma(id)
    fit_local_filter_models(id, gamma)
    print(f"COMPLETED {id}")
  
def get_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('id', type=int, help='OpenML dataset ID to process')
  return parser.parse_args()

if __name__ == "__main__":
  args = get_arguments()
  main(args)