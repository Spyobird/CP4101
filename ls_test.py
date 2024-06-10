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

# load a dataframe
def load_dataframe(id, path, name, opt):
  print(f"[{id}] Loading dataframe")
  df = pd.read_csv(f"results/{path}/{id}{opt}/{name}.csv", index_col=0)
  print(f"[{id}] Dataframe loaded")
  return df

# save a model
def save_sampling_model(id, nu, model, name):
  if not os.path.exists(f"models/sampling/{id}/{nu}"):
    print(f"[{id}] Creating directory for models")
    os.makedirs(f"models/sampling/{id}/{nu}", exist_ok=True)

  print(f"[{id}] Saving model")
  dump(model, f"models/sampling/{id}/{nu}/{name}.joblib")
  print(f"[{id}] Model saved")

# load a model
def load_model(root, id, split, paths, name):
  print(f"[{id}] Loading model")
  model = load(f"models/{root}/{id}{paths}/{name}.joblib")
  print(f"[{id}] Model loaded")
  return model

# load baseline model
def load_baseline_model(id, split, extra=""):
  return load_model("baseline", id, split, "", f"model_{extra}{id}_split{split:02d}")

def load_lssm_model(id, split, extra=""):
  return load_model("local_lssm", id, split, "", f"model_{extra}{id}_split{split:02d}")

def load_sel_model(ls, sel, id, split, extra=""):
  model = load_model(f"{ls}/{sel}", id, split, "", f"model_{extra}{id}_split{split:02d}")
  return model

def load_sel_rand_model(ls, sel, id, split):
  model = load_model(f"{ls}/{sel}", id, split, "", f"model_rand_{id}_split{split:02d}")
  return model

def load_cosine_model(ls, rep, ref, id, split, extra=""):
  model = load_model(f"{ls}/cos_{rep}_{ref}", id, split, "", f"model_{extra}{id}_split{split:02d}")
  return model

def load_cosine_rand_model(ls, rep, ref, id, split):
  model = load_model(f"{ls}/cos_{rep}_{ref}", id, split, "", f"model_rand_{id}_split{split:02d}")
  return model

def load_cosine_model_2(ls, rep, ref, id, split, extra=""):
  model = load_model(f"{ls}/cos2_{rep}_{ref}", id, split, "", f"model_{extra}{id}_split{split:02d}")
  return model

def load_cosine_rand_model_2(ls, rep, ref, id, split):
  model = load_model(f"{ls}/cos2_{rep}_{ref}", id, split, "", f"model_rand_{id}_split{split:02d}")
  return model

def load_pro_model(ls, rep, ref, id, split, extra=""):
  model = load_model(f"{ls}/pro_{rep}_{ref}", id, split, "", f"model_{extra}{id}_split{split:02d}")
  return model

def load_pro_rand_model(ls, rep, ref, id, split):
  model = load_model(f"{ls}/pro_{rep}_{ref}", id, split, "", f"model_rand_{id}_split{split:02d}")
  return model

## EVALUATION CODE

# fetch splits
def fetch_data_splits(id, split):
  print(f"[{id}:{split}] Fetching models and data")

  # load dataset split
  X_train, X_test, y_train, y_test, train, test = load_dataset_split(id, split)
  print(f"[{id}:{split}] Data fetched")

  return X_train, X_test, y_train, y_test

# run evaluation metric
def evaluate_models(id, split, X_test, y_test, models, eval_fn):
  print(f"[{id}:{split}] Evaluating models")
  scores = np.zeros(len(models))

  # get predicted labels
  print(f"[{id}:{split}] Generating predicted labels")
  for i in range(len(models)):
    model = models[i]
    y_pred = model.predict(X_test)

     # evaluate score
    print(f"[{id}:{split}] Running evaluation function")
    y_score = eval_fn(y_test, y_pred)

    scores[i] = y_score

  return scores

# run hypothesis test
def hypothesis_test(id, scores_df):
  print(f"[{id}] Conducting hypothesis tests")

  # get split scores
  split_scores = scores_df.filter(regex=r"split\d*")

  # testing
  print(f"[{id}] Testing accuracy")
  acc_wil = wilcoxon(split_scores.iloc[0], split_scores.iloc[1], zero_method="zsplit")
  acc_p = acc_wil.pvalue
  # print(f"[{id}] Testing f1-score")
  # f1_wil = wilcoxon(split_scores.iloc[2], split_scores.iloc[3], zero_method="zsplit")
  # f1_p = f1_wil.pvalue
  # print(f"[{id}] Testing AUC")
  # auc_wil = wilcoxon(split_scores.iloc[4], split_scores.iloc[5], zero_method="zsplit")
  # auc_p = auc_wil.pvalue

  return acc_p

def test_ls(id):
  print(f"[{id}] Running evaluation over ls")

  # initialise dicts for dataframes
  scores = {
    "metric": []
  }

  names = [
    "base",
    "lssm",
    "A_lsbo",
    "B_lsbo",
    "A_cos_A_A",
    "B_cos_A_A",
    "B_cos_B_A",
    "C_cos_B_A",
  ]

  for name in names:
    scores["metric"].append(f"accuracy_{name}")
  for name in names:
    scores["metric"].append(f"f1_{name}")
  for name in names:
    scores["metric"].append(f"auc_{name}")
  for name in names:
    scores["metric"].append(f"sv_len_{name}")
  scores["metric"].append("train_size_base")

  # loop through all splits
  for split in range(100):
    print(f"[{id}:{split}] Fetching models and data")
    # load dataset split
    X_train, X_test, y_train, y_test, train, test = load_dataset_split(id, split)
    print(f"[{id}:{split}] Data fetched")

    models = [
      load_baseline_model(id, split),
      load_lssm_model(id, split),
      load_sel_model("A", "lsbo", id, split),
      load_sel_model("B", "lsbo", id, split),
      load_cosine_model("A", "A", "A", id, split),
      load_cosine_model("B", "A", "A", id, split),
      load_cosine_model("B", "B", "A", id, split),
      load_cosine_model("C", "B", "A", id, split)
    ]
    print(f"[{id}:{split}] Fetching complete")

    # evaluate score
    acc_scores = evaluate_models(id, split, X_test, y_test, models, accuracy_score)
    f1_scores = evaluate_models(id, split, X_test, y_test, models, f1_score)
    auc_scores = evaluate_models(id, split, X_test, y_test, models, roc_auc_score)

    print(f"[{id}:{split}] Calculating SV set size")
    sv_counts = []
    for model in models:
      sv_counts.append(len(model.support_))
    
    base_size = len(y_train)

    scores[f"split{split}"] = [*acc_scores, *f1_scores, *auc_scores, *sv_counts, base_size]
    print(f"[{id}:{split}] Split completed")

  # convert to dataframe
  scores_df = pd.DataFrame(scores)
  scores_df["mean_score"] = scores_df.mean(numeric_only=True, axis=1)

  A_lsbo_df = load_dataframe(id, "A/lsbo", "info", "")
  B_lsbo_df = load_dataframe(id, "B/lsbo", "info", "")
  A_cos_A_A_df = load_dataframe(id, "A/cos_A_A", "info", "")
  B_cos_A_A_df = load_dataframe(id, "B/cos_A_A", "info", "")
  B_cos_B_A_df = load_dataframe(id, "B/cos_B_A", "info", "")
  C_cos_B_A_df = load_dataframe(id, "C/cos_B_A", "info", "")

  # noise reduced size
  size = A_lsbo_df.iloc[1].to_list()
  size.insert(0, "train_size_lssm")
  scores_df.loc[len(scores_df)] = size

  size = A_lsbo_df.iloc[0].to_list()
  size.insert(0, "train_size_A_lsbo")
  scores_df.loc[len(scores_df)] = size
  size = B_lsbo_df.iloc[0].to_list()
  size.insert(0, "train_size_B_lsbo")
  scores_df.loc[len(scores_df)] = size
  size = A_cos_A_A_df.iloc[0].to_list()
  size.insert(0, "train_size_A_cos_A_A")
  scores_df.loc[len(scores_df)] = size
  size = B_cos_A_A_df.iloc[0].to_list()
  size.insert(0, "train_size_B_cos_A_A")
  scores_df.loc[len(scores_df)] = size
  size = B_cos_B_A_df.iloc[0].to_list()
  size.insert(0, "train_size_B_cos_B_A")
  scores_df.loc[len(scores_df)] = size
  size = C_cos_B_A_df.iloc[0].to_list()
  size.insert(0, "train_size_C_cos_B_A")
  scores_df.loc[len(scores_df)] = size
  
  # reduction to base
  base_size = scores_df[scores_df.metric == "train_size_base"].to_numpy()[0][1:]

  size = np.array(A_lsbo_df.iloc[0].to_list())
  reduction = 1 - size/base_size
  reduction = np.insert(reduction, 0, "reduction_A_lsbo")
  scores_df.loc[len(scores_df)] = reduction
  size = np.array(B_lsbo_df.iloc[0].to_list())
  reduction = 1 - size/base_size
  reduction = np.insert(reduction, 0, "reduction_B_lsbo")
  scores_df.loc[len(scores_df)] = reduction
  size = np.array(A_cos_A_A_df.iloc[0].to_list())
  reduction = 1 - size/base_size
  reduction = np.insert(reduction, 0, "reduction_A_cos_A_A")
  scores_df.loc[len(scores_df)] = reduction
  size = np.array(B_cos_A_A_df.iloc[0].to_list())
  reduction = 1 - size/base_size
  reduction = np.insert(reduction, 0, "reduction_B_cos_A_A")
  scores_df.loc[len(scores_df)] = reduction
  size = np.array(B_cos_B_A_df.iloc[0].to_list())
  reduction = 1 - size/base_size
  reduction = np.insert(reduction, 0, "reduction_B_cos_B_A")
  scores_df.loc[len(scores_df)] = reduction
  size = np.array(C_cos_B_A_df.iloc[0].to_list())
  reduction = 1 - size/base_size
  reduction = np.insert(reduction, 0, "reduction_C_cos_B_A")
  scores_df.loc[len(scores_df)] = reduction

  # save scores
  print(f"[{id}] Saving split scores")
  save_dataframe(id, "local_set_1", f"scores", "", scores_df)

  print(f"[{id}] Completed evaluation")

def test_ls_2(id):
  print(f"[{id}] Running evaluation over ls")

  # initialise dicts for dataframes
  scores = {
    "metric": []
  }

  names = [
    "base",
    "lssm",
    "A_lsbo",
    "A_lsbo_rand",
    "A_cos_A_A",
    "A_cos_A_A_rand",
    "A_cos_B_A",
    "A_cos_B_A_rand",
    "B_cos_B_A",
    "B_cos_B_A_rand",
    "C_cos_B_A",
    "C_cos_B_A_rand",
  ]

  for name in names:
    scores["metric"].append(f"accuracy_{name}")
  for name in names:
    scores["metric"].append(f"f1_{name}")
  for name in names:
    scores["metric"].append(f"auc_{name}")
  for name in names:
    scores["metric"].append(f"sv_len_{name}")
  scores["metric"].append("train_size_base")

  # loop through all splits
  for split in range(100):
    print(f"[{id}:{split}] Fetching models and data")
    # load dataset split
    X_train, X_test, y_train, y_test, train, test = load_dataset_split(id, split)
    print(f"[{id}:{split}] Data fetched")

    models = [
      load_baseline_model(id, split),
      load_lssm_model(id, split),
      load_sel_model("A", "lsbo", id, split),
      load_sel_rand_model("A", "lsbo", id, split),
      load_cosine_model("A", "A", "A", id, split),
      load_cosine_rand_model("A", "A", "A", id, split),
      load_cosine_model("A", "B", "A", id, split),
      load_cosine_rand_model("A", "B", "A", id, split),
      load_cosine_model("B", "B", "A", id, split),
      load_cosine_rand_model("B", "B", "A", id, split),
      load_cosine_model("C", "B", "A", id, split),
      load_cosine_rand_model("C", "B", "A", id, split)
    ]
    print(f"[{id}:{split}] Fetching complete")

    # evaluate score
    acc_scores = evaluate_models(id, split, X_test, y_test, models, accuracy_score)
    f1_scores = evaluate_models(id, split, X_test, y_test, models, f1_score)
    auc_scores = evaluate_models(id, split, X_test, y_test, models, roc_auc_score)

    print(f"[{id}:{split}] Calculating SV set size")
    sv_counts = []
    for model in models:
      sv_counts.append(len(model.support_))
    
    base_size = len(y_train)

    scores[f"split{split}"] = [*acc_scores, *f1_scores, *auc_scores, *sv_counts, base_size]
    print(f"[{id}:{split}] Split completed")

  # convert to dataframe
  scores_df = pd.DataFrame(scores)
  scores_df["mean_score"] = scores_df.mean(numeric_only=True, axis=1)

  A_lsbo_df = load_dataframe(id, "A/lsbo", "info", "")
  A_cos_A_A_df = load_dataframe(id, "A/cos_A_A", "info", "")
  A_cos_B_A_df = load_dataframe(id, "A/cos_B_A", "info", "")
  B_cos_B_A_df = load_dataframe(id, "B/cos_B_A", "info", "")
  C_cos_B_A_df = load_dataframe(id, "C/cos_B_A", "info", "")

  # noise reduced size
  size = A_lsbo_df.iloc[1].to_list()
  size.insert(0, "train_size_lssm")
  scores_df.loc[len(scores_df)] = size

  size = A_lsbo_df.iloc[0].to_list()
  size.insert(0, "train_size_A_lsbo")
  scores_df.loc[len(scores_df)] = size
  size = A_cos_A_A_df.iloc[0].to_list()
  size.insert(0, "train_size_A_cos_A_A")
  scores_df.loc[len(scores_df)] = size
  size = A_cos_B_A_df.iloc[0].to_list()
  size.insert(0, "train_size_A_cos_B_A")
  scores_df.loc[len(scores_df)] = size
  size = B_cos_B_A_df.iloc[0].to_list()
  size.insert(0, "train_size_B_cos_B_A")
  scores_df.loc[len(scores_df)] = size
  size = C_cos_B_A_df.iloc[0].to_list()
  size.insert(0, "train_size_C_cos_B_A")
  scores_df.loc[len(scores_df)] = size
  
  # reduction to base
  base_size = scores_df[scores_df.metric == "train_size_base"].to_numpy()[0][1:]

  size = np.array(A_lsbo_df.iloc[0].to_list())
  reduction = 1 - size/base_size
  reduction = np.insert(reduction, 0, "reduction_A_lsbo")
  scores_df.loc[len(scores_df)] = reduction
  size = np.array(A_cos_A_A_df.iloc[0].to_list())
  reduction = 1 - size/base_size
  reduction = np.insert(reduction, 0, "reduction_A_cos_A_A")
  scores_df.loc[len(scores_df)] = reduction
  size = np.array(A_cos_B_A_df.iloc[0].to_list())
  reduction = 1 - size/base_size
  reduction = np.insert(reduction, 0, "reduction_A_cos_B_A")
  scores_df.loc[len(scores_df)] = reduction
  size = np.array(B_cos_B_A_df.iloc[0].to_list())
  reduction = 1 - size/base_size
  reduction = np.insert(reduction, 0, "reduction_B_cos_B_A")
  scores_df.loc[len(scores_df)] = reduction
  size = np.array(C_cos_B_A_df.iloc[0].to_list())
  reduction = 1 - size/base_size
  reduction = np.insert(reduction, 0, "reduction_C_cos_B_A")
  scores_df.loc[len(scores_df)] = reduction

  # save scores
  print(f"[{id}] Saving split scores")
  save_dataframe(id, "local_set_2", f"scores", "", scores_df)

  print(f"[{id}] Completed evaluation")

def test_ls_3(id):
  print(f"[{id}] Running evaluation over ls")

  # initialise dicts for dataframes
  scores = {
    "metric": []
  }

  names = [
    "base",
    "lssm",
    "A_lsbo",
    "A_lsbo_rand",
    "A_cos_B_A",
    "A_cos_B_A_rand",
    "A_cos2_B_A",
    "A_cos2_B_A_rand",
    "A_pro_B_A",
    "A_pro_B_A_rand"
  ]

  for name in names:
    scores["metric"].append(f"accuracy_{name}")
  for name in names:
    scores["metric"].append(f"f1_{name}")
  for name in names:
    scores["metric"].append(f"auc_{name}")
  for name in names:
    scores["metric"].append(f"sv_len_{name}")
  scores["metric"].append("train_size_base")

  # loop through all splits
  for split in range(100):
    print(f"[{id}:{split}] Fetching models and data")
    # load dataset split
    X_train, X_test, y_train, y_test, train, test = load_dataset_split(id, split)
    print(f"[{id}:{split}] Data fetched")

    models = [
      load_baseline_model(id, split),
      load_lssm_model(id, split),
      load_sel_model("A", "lsbo", id, split),
      load_sel_rand_model("A", "lsbo", id, split),
      load_cosine_model("A", "B", "A", id, split),
      load_cosine_rand_model("A", "B", "A", id, split),
      load_cosine_model_2("A", "B", "A", id, split),
      load_cosine_rand_model_2("A", "B", "A", id, split),
      load_pro_model("A", "B", "A", id, split),
      load_pro_rand_model("A", "B", "A", id, split),
    ]
    print(f"[{id}:{split}] Fetching complete")

    # evaluate score
    acc_scores = evaluate_models(id, split, X_test, y_test, models, accuracy_score)
    f1_scores = evaluate_models(id, split, X_test, y_test, models, f1_score)
    auc_scores = evaluate_models(id, split, X_test, y_test, models, roc_auc_score)

    print(f"[{id}:{split}] Calculating SV set size")
    sv_counts = []
    for model in models:
      sv_counts.append(len(model.support_))
    
    base_size = len(y_train)

    scores[f"split{split}"] = [*acc_scores, *f1_scores, *auc_scores, *sv_counts, base_size]
    print(f"[{id}:{split}] Split completed")

  # convert to dataframe
  scores_df = pd.DataFrame(scores)
  scores_df["mean_score"] = scores_df.mean(numeric_only=True, axis=1)

  A_lsbo_df = load_dataframe(id, "A/lsbo", "info", "")
  A_cos_B_A_df = load_dataframe(id, "A/cos_B_A", "info", "")
  A_cos2_B_A_df = load_dataframe(id, "A/cos2_B_A", "info", "")
  A_pro_B_A_df = load_dataframe(id, "A/pro_B_A", "info", "")

  # noise reduced size
  size = A_lsbo_df.iloc[1].to_list()
  size.insert(0, "train_size_lssm")
  scores_df.loc[len(scores_df)] = size

  size = A_lsbo_df.iloc[0].to_list()
  size.insert(0, "train_size_A_lsbo")
  scores_df.loc[len(scores_df)] = size
  size = A_cos_B_A_df.iloc[0].to_list()
  size.insert(0, "train_size_A_cos_B_A")
  scores_df.loc[len(scores_df)] = size
  size = A_cos2_B_A_df.iloc[0].to_list()
  size.insert(0, "train_size_A_cos2_B_A")
  scores_df.loc[len(scores_df)] = size
  size = A_pro_B_A_df.iloc[0].to_list()
  size.insert(0, "train_size_A_pro_B_A")
  scores_df.loc[len(scores_df)] = size
  
  # reduction to base
  base_size = scores_df[scores_df.metric == "train_size_base"].to_numpy()[0][1:]

  size = np.array(A_lsbo_df.iloc[0].to_list())
  reduction = 1 - size/base_size
  reduction = np.insert(reduction, 0, "reduction_A_lsbo")
  scores_df.loc[len(scores_df)] = reduction
  size = np.array(A_cos_B_A_df.iloc[0].to_list())
  reduction = 1 - size/base_size
  reduction = np.insert(reduction, 0, "reduction_A_cos_B_A")
  scores_df.loc[len(scores_df)] = reduction
  size = np.array(A_cos2_B_A_df.iloc[0].to_list())
  reduction = 1 - size/base_size
  reduction = np.insert(reduction, 0, "reduction_A_cos2_B_A")
  scores_df.loc[len(scores_df)] = reduction
  size = np.array(A_pro_B_A_df.iloc[0].to_list())
  reduction = 1 - size/base_size
  reduction = np.insert(reduction, 0, "reduction_A_pro_A_A")
  scores_df.loc[len(scores_df)] = reduction

  # save scores
  print(f"[{id}] Saving split scores")
  save_dataframe(id, "local_set_3", f"scores", "", scores_df)

  print(f"[{id}] Completed evaluation")

def test_ls_4(id):
  print(f"[{id}] Running evaluation over ls")

  # initialise dicts for dataframes
  scores = {
    "metric": []
  }

  names = [
    "base",
    "lssm",
    "A_lsbo",
    "A_lsbo_rand",
    "A_cos2_B_A",
    "A_cos2_B_A_rand",
  ]

  for name in names:
    scores["metric"].append(f"accuracy_{name}")
  for name in names:
    scores["metric"].append(f"f1_{name}")
  for name in names:
    scores["metric"].append(f"auc_{name}")
  for name in names:
    scores["metric"].append(f"sv_len_{name}")
  scores["metric"].append("train_size_base")

  # loop through all splits
  for split in range(100):
    print(f"[{id}:{split}] Fetching models and data")
    # load dataset split
    X_train, X_test, y_train, y_test, train, test = load_dataset_split(id, split)
    print(f"[{id}:{split}] Data fetched")

    models = [
      load_baseline_model(id, split),
      load_lssm_model(id, split),
      load_sel_model("A", "lsbo", id, split),
      load_sel_rand_model("A", "lsbo", id, split),
      load_cosine_model_2("A", "B", "A", id, split),
      load_cosine_rand_model_2("A", "B", "A", id, split),
    ]
    print(f"[{id}:{split}] Fetching complete")

    # evaluate score
    acc_scores = evaluate_models(id, split, X_test, y_test, models, accuracy_score)
    f1_scores = evaluate_models(id, split, X_test, y_test, models, f1_score)
    auc_scores = evaluate_models(id, split, X_test, y_test, models, roc_auc_score)

    print(f"[{id}:{split}] Calculating SV set size")
    sv_counts = []
    for model in models:
      sv_counts.append(len(model.support_))
    
    base_size = len(y_train)

    scores[f"split{split}"] = [*acc_scores, *f1_scores, *auc_scores, *sv_counts, base_size]
    print(f"[{id}:{split}] Split completed")

  # convert to dataframe
  scores_df = pd.DataFrame(scores)
  scores_df["mean_score"] = scores_df.mean(numeric_only=True, axis=1)

  A_lsbo_df = load_dataframe(id, "A/lsbo", "info", "")
  A_cos2_B_A_df = load_dataframe(id, "A/cos2_B_A", "info", "")

  # noise reduced size
  size = A_lsbo_df.iloc[1].to_list()
  size.insert(0, "train_size_lssm")
  scores_df.loc[len(scores_df)] = size

  size = A_lsbo_df.iloc[0].to_list()
  size.insert(0, "train_size_A_lsbo")
  scores_df.loc[len(scores_df)] = size
  size = A_cos2_B_A_df.iloc[0].to_list()
  size.insert(0, "train_size_A_cos2_B_A")
  scores_df.loc[len(scores_df)] = size
  
  # reduction to base
  base_size = scores_df[scores_df.metric == "train_size_base"].to_numpy()[0][1:]

  size = np.array(A_lsbo_df.iloc[0].to_list())
  reduction = 1 - size/base_size
  reduction = np.insert(reduction, 0, "reduction_A_lsbo")
  scores_df.loc[len(scores_df)] = reduction
  size = np.array(A_cos2_B_A_df.iloc[0].to_list())
  reduction = 1 - size/base_size
  reduction = np.insert(reduction, 0, "reduction_A_cos2_B_A")
  scores_df.loc[len(scores_df)] = reduction

  # save scores
  print(f"[{id}] Saving split scores")
  save_dataframe(id, "local_set_4", f"scores", "", scores_df)

  print(f"[{id}] Completed evaluation")

def test_ls_5(id):
  print(f"[{id}] Running evaluation over ls")

  # initialise dicts for dataframes
  scores = {
    "metric": []
  }

  names = [
    "base_knn",
    "lssm_knn",
    "A_lsbo_knn",
    "A_cos2_B_A_knn",
  ]

  for name in names:
    scores["metric"].append(f"accuracy_{name}")
  for name in names:
    scores["metric"].append(f"f1_{name}")
  for name in names:
    scores["metric"].append(f"auc_{name}")
  scores["metric"].append("train_size_base")

  # loop through all splits
  for split in range(100):
    print(f"[{id}:{split}] Fetching models and data")
    # load dataset split
    X_train, X_test, y_train, y_test, train, test = load_dataset_split(id, split)
    print(f"[{id}:{split}] Data fetched")

    models = [
      load_baseline_model(id, split, "knn_"),
      load_lssm_model(id, split, "knn_"),
      load_sel_model("A", "lsbo", id, split, "knn_"),
      load_cosine_model_2("A", "B", "A", id, split, "knn_"),
    ]
    print(f"[{id}:{split}] Fetching complete")

    # evaluate score
    acc_scores = evaluate_models(id, split, X_test, y_test, models, accuracy_score)
    f1_scores = evaluate_models(id, split, X_test, y_test, models, f1_score)
    auc_scores = evaluate_models(id, split, X_test, y_test, models, roc_auc_score)
    
    base_size = len(y_train)

    scores[f"split{split}"] = [*acc_scores, *f1_scores, *auc_scores, base_size]
    print(f"[{id}:{split}] Split completed")

  # convert to dataframe
  scores_df = pd.DataFrame(scores)
  scores_df["mean_score"] = scores_df.mean(numeric_only=True, axis=1)

  # save scores
  print(f"[{id}] Saving split scores")
  save_dataframe(id, "local_set_5", f"scores", "", scores_df)

  print(f"[{id}] Completed evaluation")
      
def main(args):
  if args.id:
    id = args.id
    print(f"LOCAL SET EVALUATION {id}")
    test_ls_5(id)
    print(f"COMPLETED {id}")
  
def get_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('id', type=int, help='OpenML dataset ID to process')
  return parser.parse_args()

if __name__ == "__main__":
  args = get_arguments()
  main(args)