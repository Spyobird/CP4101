# import relevant libraries
# data management
import numpy as np
import pandas as pd

# helper
import argparse
from collections import Counter
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

# save a ls sel model
def save_ls_sel_model(id, model, ls, sel, name):
  if not os.path.exists(f"models/{ls}/{sel}/{id}"):
    print(f"[{id}] Creating directory for models")
    os.makedirs(f"models/{ls}/{sel}/{id}", exist_ok=True)

  print(f"[{id}] Saving model")
  dump(model, f"models/{ls}/{sel}/{id}/{name}.joblib")
  print(f"[{id}] Model saved")

# save a ls cosine model
def save_ls_cosine_model(id, model, ls, rep, ref, name):
  if not os.path.exists(f"models/{ls}/cos_{rep}_{ref}/{id}"):
    print(f"[{id}] Creating directory for models")
    os.makedirs(f"models/{ls}/cos_{rep}_{ref}/{id}", exist_ok=True)

  print(f"[{id}] Saving model")
  dump(model, f"models/{ls}/cos_{rep}_{ref}/{id}/{name}.joblib")
  print(f"[{id}] Model saved")

def save_ls_cosine_model_2(id, model, ls, rep, ref, name):
  if not os.path.exists(f"models/{ls}/cos2_{rep}_{ref}/{id}"):
    print(f"[{id}] Creating directory for models")
    os.makedirs(f"models/{ls}/cos2_{rep}_{ref}/{id}", exist_ok=True)

  print(f"[{id}] Saving model")
  dump(model, f"models/{ls}/cos2_{rep}_{ref}/{id}/{name}.joblib")
  print(f"[{id}] Model saved")

# save a ls projection model
def save_ls_pro_model(id, model, ls, rep, ref, name):
  if not os.path.exists(f"models/{ls}/pro_{rep}_{ref}/{id}"):
    print(f"[{id}] Creating directory for models")
    os.makedirs(f"models/{ls}/pro_{rep}_{ref}/{id}", exist_ok=True)

  print(f"[{id}] Saving model")
  dump(model, f"models/{ls}/pro_{rep}_{ref}/{id}/{name}.joblib")
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
  # S_idx = []
  # for i in range(len(y)):
  #   if int(bitmask) & (1 << i) != 0:
  #     S_idx.append(i)
  # S_idx = np.array(S_idx, dtype=int)
  S_idx = np.array(list(bitmask))
  X_S = X[S_idx]
  y_S = y[S_idx]
  return X_S, y_S

# LOCAL SET ALGORITHMS

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

# B: hyperspheres (Lorena et al. 2019)
def ls_B(X, y, distances, ne, ne_dist):
  cores = np.array(range(len(y)), dtype=int)
  ls = [set() for _ in range(len(y))]
  # ls = np.array([0 for _ in range(len(y))], dtype=object)
  radii = np.full(len(y), -1, dtype=float)

  def radius(i):
    j = ne[i]
    d = ne_dist[i]
    k = ne[j]
    if i == k:
      radii[i] = d/2
    else:
      if radii[j] == -1:
        radius(j)
      radii[i] = d - radii[j]

  # compute radii of hyperspheres
  for i in range(len(y)):
    if radii[i] != -1:
      continue
    radius(i)

  # check hypersphere membership
  for i in range(len(cores)):
    x = cores[i]
    for j in range(len(y)):
      if distances[x, j] < radii[i]:
        ls[i].add(j)
        # ls[i] = int(ls[i]) | (1 << j)

  return cores, ls, radii

# C: ls_B + subset reduction
def ls_C(X, y, distances, ne, ne_dist):
  c1, l1, r1 = ls_B(X, y, distances, ne, ne_dist)
  c2, l2, r2 = subset_reduction(c1, l1, r1, ne)

  return c2, l2, r2

def subset_reduction(cores, ls, radii, ne):
  def partition_ne(ne):
    ne_set = set(ne)
    ne_idx = list(ne_set)
    partitions = [set() for _ in ne_idx]
    for i in range(len(ne)):
      index = ne_idx.index(ne[i])
      partitions[index].add(i)
    return partitions, ne_idx

  def maximal_sets(ls, partition):
    partition_list = np.array(list(partition))
    lsc = np.array([len(ls[i]) for i in partition_list])
    excluded = set()

    # sort by LSC (descending)
    lsc_sort = np.argsort(-lsc)
    partition_list_sort = partition_list[lsc_sort]

    maximal = []

    # total O(k^3)?
    for i in partition_list_sort: # O(k)
      if i in excluded: # O(1)
        continue
      maximal.append(i)
      excluded.add(i)
      for j in ls[i]: # O(k)
        if j in excluded: # O(1)
          continue
        if ls[j].issubset(ls[i]): # O(k)
          excluded.add(j)
          
    maximal = np.array(maximal)

    return maximal
  
  partitions, nes = partition_ne(ne)
  maximals = [maximal_sets(ls, p) for p in partitions]
  idxs = np.concatenate(maximals)

  return cores[idxs], [ls[i] for i in idxs], radii[idxs]

# D: ls_A + remove LSC=1
def ls_D(X, y, distances, ne, ne_dist):
  c1, l1, r1 = ls_A(X, y, distances, ne, ne_dist)
  c2, l2, r2 = filter_singles(c1, l1, r1)
  return c2, l2, r2

def filter_singles(cores, ls, radii):
  idxs = np.fromiter((i for i in range(len(cores)) if len(ls[i]) > 1), int)
  return cores[idxs], [ls[i] for i in idxs], radii[idxs]

# SELECTION STRATEGIES
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

# COSINE SELECTION STRATEGIES

# cosine similarity between two vectors
def cosine_similarity(a, b, base=0):
  den = np.linalg.norm(a)*np.linalg.norm(b)
  if den == 0:
    return base
  sim = np.dot(a, b)/(den)
  return sim

# get cosine similarities for all members of a local set
def ls_cosine_similarity(X, rep, ref, members):
  idxs = []
  sims = []
  x_tilde = rep
  e = ref
  for i in range(len(X)):
    # if int(members) & (1 << i) == 0:
    #   continue
    if i in members:
      x = X[i]
      a = e - x
      b = x_tilde - x
      sim = cosine_similarity(a, b)
      idxs.append(i)
      sims.append(sim)
  idxs = np.array(idxs)
  sims = np.array(sims)
  return idxs, sims

def sort_cosine_similarities(idxs, sims):
  # can use quickselect to improve time
  sort = np.argsort(sims)
  idxs_sort = idxs[sort]
  return idxs_sort

def choose_best_instances(idxs_sort, k=1):
  if len(idxs_sort) < k:
    return idxs_sort.copy()
  else:
    return idxs_sort[:k].copy()

def cosine_selection(X, ls, reps, refs, k=1):
  S = set()
  # S = 0
  for i in range(len(ls)):
    members = ls[i]
    ref = refs[i]
    rep = reps[i]
    idxs, sims = ls_cosine_similarity(X, rep, ref, members)
    idxs_sort = sort_cosine_similarities(idxs, sims)
    x_hats = choose_best_instances(idxs_sort, k)
    for x_hat in x_hats:
      S.add(x_hat)
      # S |= (1 << x_hat)
  # return int(S)
  return S

def cosine_selection_1(X, ls, reps, refs):
  S = set()
  for i in range(len(ls)):
    members = list(ls[i])
    members.sort() # guarantee on order for O(k log k) (can remove when not needed)
    e = refs[i]
    x_tilde = reps[i]
    vals = np.fromiter((cosine_similarity(e - X[j], x_tilde - X[j]) for j in members), float)
    idx = np.argmin(vals)
    x_hat = members[idx]
    S.add(x_hat)
  return S

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

# PROJECTION IDEA
def projection_value(a, b):
  return np.dot(a, b)

def choose_best_instance(X, members, x_tilde, e, v):
  vals = np.fromiter((projection_value(X[i] - x_tilde, v) for i in members), float)
  max_idx = np.argmax(vals)
  return max_idx

def projection_selection(X, ls, reps, refs):
  S = set()
  for i in range(len(ls)):
    members = list(ls[i])
    members.sort() # guarantee on order for O(k log k) (can remove when not needed)
    e = refs[i]
    x_tilde = reps[i]
    v = e - x_tilde
    idx = choose_best_instance(X, members, x_tilde, e, v)
    x_hat = members[idx]
    S.add(x_hat)
  return S

# REPRESENTATIVE SELECTION

# A: LS cores (naive)
def rep_A(X, cores, ls):
  return X[cores]

# B: cluster centroids
def rep_B(X, cores, ls):
  reps = np.zeros((len(cores), X.shape[1]))
  for i in range(len(cores)):
    instances = X[np.array(list(ls[i]), dtype=int)]
    centroid = np.mean(instances, axis=0)
    reps[i] = centroid
  return reps

# REFERENCE SELECTION

# A: nearest enemies (naive)
def ref_A(X, cores, ne):
  refs = np.full(len(cores), -1, dtype=int)
  for i in range(len(cores)):
    x = cores[i]
    refs[i] = ne[x]
  return X[refs]

## main code
def process(id, gamma, ls_option, sel_option, rep_option, ref_option, rand_option):
  print(f"[{id}] Begin model training")
  print(f"[{id}] Options: {ls_option}, {sel_option}, {rep_option}, {ref_option}")

  info = {}

  for split in range(100):

    print(f"[{id}:{split}] Begin fitting split")
    # load dataset split
    X_train, X_test, y_train, y_test = load_lssm_dataset_split(id, split)

    # preprocess
    distances, ne, ne_dist = get_distances(X_train, y_train)

    # get local sets
    print(f"[{id}:{split}] Calculating local sets")
    ls_algo = eval(f"ls_{ls_option}")
    cores, ls, radii = ls_algo(X_train, y_train, distances, ne, ne_dist)
    print(f"[{id}:{split}] Completed local sets")

    # cosine selection
    if sel_option == "cos":
      # Construct reduced set
      print(f"[{id}:{split}] Constructing subset")
      rep_algo = eval(f"rep_{rep_option}")
      reps = rep_algo(X_train, cores, ls)
      ref_algo = eval(f"ref_{ref_option}")
      refs = ref_algo(X_train, cores, ne)
      S = cosine_selection(X_train, ls, reps, refs, 1)
      X_s, y_s = bitmask_to_Xy(X_train, y_train, S)
      print(f"[{id}:{split}] Completed subset construction")
    
    else: 
      # Construct reduced set
      print(f"[{id}:{split}] Constructing subset")
      sel_algo = eval(f"{sel_option}")
      S = sel_algo(X_train, y_train, distances, ne, ne_dist, cores, ls, radii)
      X_s, y_s = bitmask_to_Xy(X_train, y_train, S)
      print(f"[{id}:{split}] Completed subset construction")

    info[f"split{split}"] = [len(y_s), len(y_train), 1 - len(y_s)/len(y_train)]

    clf = SVC(C=1, gamma=gamma)

    print(f"[{id}:{split}] Fit model")
    clf.fit(X_s, y_s)
    print(f"[{id}:{split}] Model fitting completed")

    print(f"[{id}:{split}] Save fitted model")

    if sel_option == "cos":
      save_ls_cosine_model(id, clf, ls_option, rep_option, ref_option, f"model_{id}_split{split:02d}")
    else:
      save_ls_sel_model(id, clf, ls_option, sel_option, f"model_{id}_split{split:02d}")

    if rand_option:
      print(f"[{id}:{split}] Fitting random sampling model")
      sample_size = len(y_s)
      current_seed = (1287 + hash(split * 27)) % 2**16
      np.random.seed(current_seed)
      random_idx = np.arange(len(y_train))
      while True:
        np.random.shuffle(random_idx)

        X_samp = X_train[random_idx[:sample_size]]
        y_samp = y_train[random_idx[:sample_size]]
        if not np.all(y_samp == y_samp[0]):
          break
        
        current_seed = (hash(current_seed + 1)) % 2**16
        np.random.seed(current_seed)

      clf = SVC(C=1, gamma=gamma)
      clf.fit(X_samp, y_samp)

      print(f"[{id}:{split}] Save fitted model")
      if sel_option == "cos":
        save_ls_cosine_model(id, clf, ls_option, rep_option, ref_option, f"model_rand_{id}_split{split:02d}")
      else:
        save_ls_sel_model(id, clf, ls_option, sel_option, f"model_rand_{id}_split{split:02d}")

    print(f"[{id}:{split}] Completed split")

  info_df = pd.DataFrame(info)
  info_df["mean"] = info_df.mean(numeric_only=True, axis=1)
  
  if sel_option == "cos":
    save_dataframe(id, f"{ls_option}/cos_{rep_option}_{ref_option}", "info", "", info_df)
  else:
    save_dataframe(id, f"{ls_option}/{sel_option}", "info", "", info_df)
  
  print(f"[{id}] End model training")

def process_2(id, gamma, ls_option, sel_option, rep_option, ref_option, rand_option=False, knn=False):
  print(f"[{id}] Begin model training")
  print(f"[{id}] Options: {ls_option}, {sel_option}, {rep_option}, {ref_option}")

  info = {}

  timings = []
  for split in range(100):
    sub_t = []
    print(f"[{id}:{split}] Begin fitting split")
    # load dataset split
    X_train, X_test, y_train, y_test = load_lssm_dataset_split(id, split)

    # preprocess
    start = time.process_time()
    distances, ne, ne_dist = get_distances(X_train, y_train)
    end = time.process_time()
    exec_time = end - start
    sub_t.append(exec_time)

    # get local sets
    print(f"[{id}:{split}] Calculating local sets")
    ls_algo = eval(f"ls_{ls_option}")
    start = time.process_time()
    cores, ls, radii = ls_algo(X_train, y_train, distances, ne, ne_dist)
    end = time.process_time()
    exec_time = end - start
    sub_t.append(exec_time)
    print(f"[{id}:{split}] Completed local sets")

    # projection selection
    if sel_option == "pro":
      # Construct reduced set
      print(f"[{id}:{split}] Constructing subset")
      rep_algo = eval(f"rep_{rep_option}")
      reps = rep_algo(X_train, cores, ls)
      ref_algo = eval(f"ref_{ref_option}")
      refs = ref_algo(X_train, cores, ne)
      S = projection_selection(X_train, ls, reps, refs)
      X_s, y_s = bitmask_to_Xy(X_train, y_train, S)
      print(f"[{id}:{split}] Completed subset construction")
        # cosine selection
    elif sel_option == "cos":
      # Construct reduced set
      print(f"[{id}:{split}] Constructing subset")
      rep_algo = eval(f"rep_{rep_option}")
      reps = rep_algo(X_train, cores, ls)
      ref_algo = eval(f"ref_{ref_option}")
      refs = ref_algo(X_train, cores, ne)
      S = cosine_selection_1(X_train, ls, reps, refs)
      X_s, y_s = bitmask_to_Xy(X_train, y_train, S)
      print(f"[{id}:{split}] Completed subset construction")
    elif sel_option == "cos2":
      # Construct reduced set
      print(f"[{id}:{split}] Constructing subset")
      rep_algo = eval(f"rep_{rep_option}")
      start = time.process_time()
      reps = rep_algo(X_train, cores, ls)
      end = time.process_time()
      exec_time = end - start
      sub_t.append(exec_time)
      ref_algo = eval(f"ref_{ref_option}")
      start = time.process_time()
      refs = ref_algo(X_train, cores, ne)
      end = time.process_time()
      exec_time = end - start
      sub_t.append(exec_time)
      start = time.process_time()
      S = cosine_selection_2(X_train, ls, reps, refs)
      X_s, y_s = bitmask_to_Xy(X_train, y_train, S)
      end = time.process_time()
      exec_time = end - start
      sub_t.append(exec_time)
      print(f"[{id}:{split}] Completed subset construction")
    else: 
      # Construct reduced set
      print(f"[{id}:{split}] Constructing subset")
      sel_algo = eval(f"{sel_option}")
      start = time.process_time()
      S = sel_algo(X_train, y_train, distances, ne, ne_dist, cores, ls, radii)
      X_s, y_s = bitmask_to_Xy(X_train, y_train, S)
      end = time.process_time()
      exec_time = end - start
      sub_t.append(exec_time)
      print(f"[{id}:{split}] Completed subset construction")

    if not knn:
      info[f"split{split}"] = [len(y_s), len(y_train), 1 - len(y_s)/len(y_train)]

    if knn:
      clf = KNeighborsClassifier(n_neighbors=5)
    else:
      clf = SVC(C=1, gamma=gamma, cache_size=1000)

    print(f"[{id}:{split}] Fit model")
    start = time.process_time()
    clf.fit(X_s, y_s)
    end = time.process_time()
    exec_time = end - start
    sub_t.append(exec_time)
    print(f"[{id}:{split}] Model fitting completed")

    print(f"[{id}:{split}] Save fitted model")

    if knn:
      if sel_option == "pro":
        save_ls_pro_model(id, clf, ls_option, rep_option, ref_option, f"model_knn_{id}_split{split:02d}")
      elif sel_option == "cos":
        save_ls_cosine_model(id, clf, ls_option, rep_option, ref_option, f"model_knn_{id}_split{split:02d}")
      elif sel_option == "cos2":
        save_ls_cosine_model_2(id, clf, ls_option, rep_option, ref_option, f"model_knn_{id}_split{split:02d}")
      else:
        save_ls_sel_model(id, clf, ls_option, sel_option, f"model_knn_{id}_split{split:02d}")
    else:
      if sel_option == "pro":
        save_ls_pro_model(id, clf, ls_option, rep_option, ref_option, f"model_{id}_split{split:02d}")
      elif sel_option == "cos":
        save_ls_cosine_model(id, clf, ls_option, rep_option, ref_option, f"model_{id}_split{split:02d}")
      elif sel_option == "cos2":
        save_ls_cosine_model_2(id, clf, ls_option, rep_option, ref_option, f"model_{id}_split{split:02d}")
      else:
        save_ls_sel_model(id, clf, ls_option, sel_option, f"model_{id}_split{split:02d}")

    if rand_option:
      print(f"[{id}:{split}] Fitting random sampling model")
      sample_size = len(y_s)
      current_seed = (961 + hash(split * 91)) % 2**16
      np.random.seed(current_seed)
      random_idx = np.arange(len(y_train))
      while True:
        np.random.shuffle(random_idx)

        X_samp = X_train[random_idx[:sample_size]]
        y_samp = y_train[random_idx[:sample_size]]
        if not np.all(y_samp == y_samp[0]):
          break
      clf = SVC(C=1, gamma=gamma, cache_size=1000)

      clf.fit(X_samp, y_samp)

      print(f"[{id}:{split}] Save fitted model")
      if sel_option == "pro":
        save_ls_pro_model(id, clf, ls_option, rep_option, ref_option, f"model_rand_{id}_split{split:02d}")
      elif sel_option == "cos":
        save_ls_cosine_model(id, clf, ls_option, rep_option, ref_option, f"model_rand_{id}_split{split:02d}")
      elif sel_option == "cos2":
        save_ls_cosine_model_2(id, clf, ls_option, rep_option, ref_option, f"model_rand_{id}_split{split:02d}")
      else:
        save_ls_sel_model(id, clf, ls_option, sel_option, f"model_rand_{id}_split{split:02d}")

    timings.append(sub_t)

    print(f"[{id}:{split}] Completed split")

  if not knn:
    info_df = pd.DataFrame(info)
    info_df["mean"] = info_df.mean(numeric_only=True, axis=1)
    
    if sel_option == "pro":
      save_dataframe(id, f"{ls_option}/pro_{rep_option}_{ref_option}", "info", "", info_df)
    elif sel_option == "cos":
      save_dataframe(id, f"{ls_option}/cos_{rep_option}_{ref_option}", "info", "", info_df)
    elif sel_option == "cos2":
      save_dataframe(id, f"{ls_option}/cos2_{rep_option}_{ref_option}", "info", "", info_df)
    else:
      save_dataframe(id, f"{ls_option}/{sel_option}", "info", "", info_df)
  
  timings = np.array(timings)
  if knn:
    if sel_option == "pro":
      save_numpy(f"results/{ls_option}/pro_{rep_option}_{ref_option}/{id}", "timings_knn", timings)
    elif sel_option == "cos":
      save_numpy(f"results/{ls_option}/cos_{rep_option}_{ref_option}/{id}", "timings_knn", timings)
    elif sel_option == "cos2":
      save_numpy(f"results/{ls_option}/cos2_{rep_option}_{ref_option}/{id}", "timings_knn", timings)
    else:
      save_numpy(f"results/{ls_option}/{sel_option}/{id}", "timings_knn", timings)
  else:
    if sel_option == "pro":
      save_numpy(f"results/{ls_option}/pro_{rep_option}_{ref_option}/{id}", "timings", timings)
    elif sel_option == "cos":
      save_numpy(f"results/{ls_option}/cos_{rep_option}_{ref_option}/{id}", "timings", timings)
    elif sel_option == "cos2":
      save_numpy(f"results/{ls_option}/cos2_{rep_option}_{ref_option}/{id}", "timings", timings)
    else:
      save_numpy(f"results/{ls_option}/{sel_option}/{id}", "timings", timings)
  print(f"[{id}] End model training")

def main(args):
  if args.id:
    id = args.id
    print(f"LOCAL SET MODEL TRAINING {id}")
    gamma = fetch_best_gamma(id)
    process_2(id, gamma, args.ls, args.sel, args.rep, args.ref, args.rand, args.knn)
    print(f"COMPLETED {id}")
  
def get_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('id', type=int, help='OpenML dataset ID to process')
  parser.add_argument('--ls', default='A')
  parser.add_argument('--sel', default='pro')
  parser.add_argument('--rep', default='A')
  parser.add_argument('--ref', default='A')
  parser.add_argument('--rand', action='store_true')
  parser.add_argument('--knn', action='store_true')
  return parser.parse_args()

if __name__ == "__main__":
  args = get_arguments()
  main(args)