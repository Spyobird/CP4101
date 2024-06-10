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
def load_data():
  matrices = np.load("templates/4x4_bal.npy")
  data = convert_Xy(matrices, scale=False)
  return data

def convert_Xy(matrices, scale=True):
  data = []
  n = matrices.shape[1]
  offset = -(n-1)/2
  factor = 1
  if scale:
    offset = -1
    factor = (n-1)/2
  for m in matrices:
    X = []
    y = []
    for i in range(n):
      for j in range(n):
        if m[i, j] == -1:
          continue
        X.append([i/factor + offset, j/factor + offset])
        y.append([m[i, j]])
    data.append((np.array(X), np.array(y).ravel()))
  return data

def generate_test_grid(X, Y, side=1, resolution=100):
  n = resolution - 1
  separation = side/resolution
  offset = -side/2 + separation
  X_test = []
  y_test = []
  for i in range(len(Y)):
    x = X[i]
    y = Y[i]
    for j in range(n):
      for k in range(n):
        coords = x + np.array([j*separation + offset, k*separation + offset])
        X_test.append(coords)
        y_test.append(y)
  X_test = np.array(X_test)
  y_test = np.array(y_test)
  return X_test, y_test

def add_error(X, epsilon=1e-10, random_state=9):
  np.random.seed(random_state)
  X_e = X + (np.random.rand(*X.shape)*2-1)*epsilon
  return X_e
  
# save a dataframe
def save_dataframe(path, name, df):
  if not os.path.exists(f"results/{path}"):
    print(f"Creating directory for results")
    os.makedirs(f"results/{path}", exist_ok=True)

  print(f"Saving dataframe to csv")
  df.to_csv(f"results/{path}/{name}.csv")
  print(f"Dataframe saved")


## EVALUATION CODE

def get_distances(X, y):
  # calculate distances between all points: O(n^2)
  distances = pairwise_distances(X)

  # find nearest enemies: O(n^2)
  ne = np.zeros(len(y), dtype=int)
  ne_dist = np.zeros(len(y))
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
  return distances, ne, ne_dist

def bitmask_to_Xy(X, y, bitmask):
  S_idx = np.array(list(bitmask))
  X_S = X[S_idx]
  y_S = y[S_idx]
  return X_S, y_S

# A: local sets (Levya et al. 2015)
def ls_A(X, y, distances, ne, ne_dist):
  cores = np.array(range(len(y)), dtype=int)
  ls = [set() for _ in range(len(y))]
  # ls = np.array([0 for _ in range(len(y))], dtype=object)
  radii = np.array(ne_dist)

  # check ls membership
  for i in range(len(cores)):
    x = cores[i]
    for j in range(len(y)):
      if distances[x, j] < radii[i]:
        ls[i].add(j)
        # ls[i] = int(ls[i]) | (1 << j)

  return cores, ls, radii

def lsbo(X, y, distances, ne, ne_dist, cores, ls, radii):
  lsc = np.zeros(len(cores))
  S = set()
  # S = 0

  # calculate LSC
  for i in range(len(cores)):
    lsc[i] = len(ls[i])
    # lsc[i] = ls[i].bit_count()

  # sort by LSC
  lsc_sort = np.argsort(lsc)

  # check intersection and add to set
  for i in lsc_sort:
    if len(S.intersection(ls[i])) == 0:
      S.add(i)
    # inter = int(S) & int(ls[i])
    # if inter == 0:
    #   S = int(S) | (1 << i)

  # return int(S)
  return S

# return cluster centroids
def rep_B(X, cores, ls):
  reps = np.zeros((len(cores), X.shape[1]))
  for i in range(len(cores)):
    instances = X[np.array(list(ls[i]))]
    centroid = np.mean(instances, axis=0)
    reps[i] = centroid
  return reps

# return nearest enemy of cores
def ref_A(X, cores, ne):
  refs = np.full(len(cores), -1, dtype=int)
  for i in range(len(cores)):
    x = cores[i]
    refs[i] = ne[x]
  return X[refs]

# cosine similarity between two vectors
def cosine_similarity(a, b, base=0):
  den = np.linalg.norm(a)*np.linalg.norm(b)
  if den == 0:
    return base
  sim = np.dot(a, b)/(den)
  return sim

def cosine_selection_2(X, ls, reps, refs):
  S = set()
  for i in range(len(ls)):
    members = list(ls[i])
    members.sort() # guarantee on order for O(k log k) (can remove when not needed)
    e = refs[i]
    x_tilde = reps[i]
    a = e - x_tilde
    vals = np.fromiter((cosine_similarity(a, X[j] - x_tilde) for j in members), float)
    idx = np.argmax(vals)
    x_hat = members[idx]
    S.add(x_hat)
  return S

# new selection process
def min_a_selection(X, ls, reps, refs):
  S = set()
  for i in range(len(ls)):
    members = list(ls[i])
    e = refs[i]
    vals = np.fromiter((np.linalg.norm(e - X[j]) for j in members), float)
    idx = np.argmin(vals)
    x_hat = members[idx]
    S.add(x_hat)
  return S

# scalar project of a on b
def scalar_projection(a, b, epsilon=1e-10):
  return (np.dot(a, b) + epsilon)/np.linalg.norm(b)

def min_ratio_ad_selection(X, ls, reps, refs):
  S = set()
  for i in range(len(ls)):
    members = list(ls[i])
    e = refs[i]
    x_tilde = reps[i]
    v = e - x_tilde
    vals = np.fromiter((np.linalg.norm(e - X[j])/scalar_projection(X[j] - x_tilde, v) for j in members), float)
    idx = np.argmin(vals)
    x_hat = members[idx]
    S.add(x_hat)
  return S


def process():
  data = load_data()

  results = {
      "index": [],
      "rand_state": [],
      "base_score": [],
      "lsbo_error": [],
      "lsbo_reduction": [],
      "algo_error": [],
      "algo_reduction": [],
  }

  for i in range(len(data[400:])):
    X, y = data[i]
    X_test, y_test = generate_test_grid(X, y)

    random_state = (8+71*i) % (2**16)
    X = add_error(X, random_state=random_state)
    base_nn = KNeighborsClassifier(1)
    base_nn.fit(X, y)
    y_pred = base_nn.predict(X_test)
    base_score = accuracy_score(y_test, y_pred)

    distances, ne, ne_dist = get_distances(X, y)
    cores, ls, radii = ls_A(X, y, distances, ne, ne_dist)

    S = lsbo(X, y, distances, ne, ne_dist, cores, ls, radii)
    X_S, y_S = bitmask_to_Xy(X, y, S)
    lsbo_nn = KNeighborsClassifier(1)
    lsbo_nn.fit(X_S, y_S)
    y_pred = lsbo_nn.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    lsbo_error = base_score - score
    lsbo_red = 1 - len(y_S)/len(y)

    reps = rep_B(X, cores, ls)
    refs = ref_A(X, cores, ne)
    S = cosine_selection_2(X, ls, reps, refs)
    X_S, y_S = bitmask_to_Xy(X, y, S)
    algo_nn = KNeighborsClassifier(1)
    algo_nn.fit(X_S, y_S)
    y_pred = algo_nn.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    algo_error = base_score - score
    algo_red = 1 - len(y_S)/len(y)

    results["index"].append(i)
    results["rand_state"].append(random_state)
    results["base_score"].append(base_score)
    results["lsbo_error"].append(lsbo_error)
    results["lsbo_reduction"].append(lsbo_red)
    results["algo_error"].append(algo_error)
    results["algo_reduction"].append(algo_red)

  df = pd.DataFrame(results)
  save_dataframe("voronoi", "vor_4x4_400_887", df)

def process_2():
  data = load_data()

  results = {
      "index": [],
      "rand_state": [],
      "base_score": [],
      "a_error": [],
      "a_reduction": [],
      "b_error": [],
      "b_reduction": [],
  }

  for i in range(len(data)):
    X, y = data[i]
    X_test, y_test = generate_test_grid(X, y, 16) # to change

    random_state = (8+71*i) % (2**16)
    X = add_error(X, random_state=random_state)
    base_nn = KNeighborsClassifier(1)
    base_nn.fit(X, y)
    y_pred = base_nn.predict(X_test)
    base_score = accuracy_score(y_test, y_pred)

    distances, ne, ne_dist = get_distances(X, y)
    cores, ls, radii = ls_A(X, y, distances, ne, ne_dist)

    reps = rep_B(X, cores, ls)
    refs = ref_A(X, cores, ne)
    S = min_a_selection(X, ls, reps, refs)
    X_S, y_S = bitmask_to_Xy(X, y, S)
    a_nn = KNeighborsClassifier(1)
    a_nn.fit(X_S, y_S)
    y_pred = a_nn.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    a_error = base_score - score
    a_red = 1 - len(y_S)/len(y)

    S = min_ratio_ad_selection(X, ls, reps, refs)
    X_S, y_S = bitmask_to_Xy(X, y, S)
    b_nn = KNeighborsClassifier(1)
    b_nn.fit(X_S, y_S)
    y_pred = b_nn.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    b_error = base_score - score
    b_red = 1 - len(y_S)/len(y)

    results["index"].append(i)
    results["rand_state"].append(random_state)
    results["base_score"].append(base_score)
    results["a_error"].append(a_error)
    results["a_reduction"].append(a_red)
    results["b_error"].append(b_error)
    results["b_reduction"].append(b_red)

    if i % 10 == 0:
      print(i)

  df = pd.DataFrame(results)
  save_dataframe("voronoi", "vor_4x4_0_887_mina_minratioad_lowres", df)

def main(args):
  print(f"VORONOI {id}")
  process_2()
  print(f"COMPLETED {id}")
  
def get_arguments():
  parser = argparse.ArgumentParser()
  return parser.parse_args()

if __name__ == "__main__":
  args = get_arguments()
  main(args)