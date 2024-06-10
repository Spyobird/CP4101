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

# save a model
def save_sampling_model(id, nu, model, name):
  if not os.path.exists(f"models/sampling/{id}/{nu}"):
    print(f"[{id}] Creating directory for models")
    os.makedirs(f"models/sampling/{id}/{nu}", exist_ok=True)

  print(f"[{id}] Saving model")
  dump(model, f"models/sampling/{id}/{nu}/{name}.joblib")
  print(f"[{id}] Model saved")

# load a model
def load_model(root, id, split, paths):
  print(f"[{id}] Loading model")
  model = load(f"models/{root}/{id}{paths}/model_{id}_split{split:02d}.joblib")
  print(f"[{id}] Model loaded")
  return model

# load baseline model
def load_baseline_model(id, split):
  return load_model("baseline", id, split, "")

# load sampling model
def load_sampling_model(id, split, nu):
  return load_model("sampling", id, split, f"/{nu}")

# load beta model
def load_beta_model(id, split, nu):
  return load_model("beta", id, split, f"/{nu}")

# load local filter model
def load_local_filter_model(id, split):
  return load_model("local_filter", id, split, "")

# load lsbo model
def load_local_lsbo_model(id, split):
  return load_model("local_lssm_lsbo", id, split, "")

# load lsrb model
def load_local_lsrb_model(id, split):
  return load_model("local_lssm_lsrb", id, split, "")

## EVALUATION CODE

# fetch models and splits
def fetch_models_and_splits(id, split, base_fn, cmp_fn):
  print(f"[{id}:{split}] Fetching models and data")

  # load dataset split
  X_train, X_test, y_train, y_test, train, test = load_dataset_split(id, split)
  print(f"[{id}:{split}] Data fetched")

  # fetch base model
  print(f"[{id}:{split}] Fetching first model")
  base = base_fn(id, split)
  print(f"[{id}:{split}] First model fetched")

  # fetch cmp model
  print(f"[{id}:{split}] Fetching second model")
  cmp = cmp_fn(id, split)
  print(f"[{id}:{split}] Second model fetched")

  print(f"[{id}:{split}] Fetching complete")

  return X_train, X_test, y_train, y_test, base, cmp

# run evaluation metric
def evaluate_models(id, split, X_test, y_test, base, cmp, eval_fn):
  print(f"[{id}:{split}] Evaluating models")

  # get predicted labels
  print(f"[{id}:{split}] Generating predicted labels")
  y_pred_base = base.predict(X_test)
  y_pred_cmp = cmp.predict(X_test)

  # evaluate score
  print(f"[{id}:{split}] Running evaluation function")
  y_score_base = eval_fn(y_test, y_pred_base)
  print(f"[{id}:{split}] Base score: {y_score_base}")
  y_score_cmp = eval_fn(y_test, y_pred_cmp)
  print(f"[{id}:{split}] Cmp score: {y_score_cmp}")

  return y_score_base, y_score_cmp

# run hypothesis test
def hypothesis_test(id, scores_df):
  print(f"[{id}] Conducting hypothesis tests")

  # get split scores
  split_scores = scores_df.filter(regex=r"split\d*")

  # testing
  print(f"[{id}] Testing accuracy")
  acc_wil = wilcoxon(split_scores.iloc[0], split_scores.iloc[1], zero_method="zsplit")
  acc_p = acc_wil.pvalue
  print(f"[{id}] Testing f1-score")
  f1_wil = wilcoxon(split_scores.iloc[2], split_scores.iloc[3], zero_method="zsplit")
  f1_p = acc_wil.pvalue
  print(f"[{id}] Testing AUC")
  auc_wil = wilcoxon(split_scores.iloc[4], split_scores.iloc[5], zero_method="zsplit")
  auc_p = acc_wil.pvalue

  return acc_p, f1_p, auc_p

def test_baseline_sampling(id):
  print(f"[{id}] Running evaluation over baseline and sampling")

  # initialise dicts for dataframes
  scores = {}
  results = {
    "model": [],
    "acc": [],
    "acc_p": [],
    "f1": [],
    "f1_p": [],
    "auc": [],
    "auc_p": [],
  }

  scores["metric"] = ["accuracy_baseline", "accuracy_sampling", "f1_baseline", "f1_sampling", "auc_baseline", "auc_sampling"]

  # loop through all proportion values of nu
  for i in range(19):
      nu = (i + 1) * 5
      print(f"[{id}] Begin at proportion {nu}")

      # loop through all splits
      for split in range(100):
        # fetch models and data
        X_train, X_test, y_train, y_test, base, cmp = fetch_models_and_splits(id, split, load_baseline_model, lambda id, split: load_sampling_model(id, split, nu))

        # evaluate over acc, f1 and auc
        print(f"[{id}:{split}] Evaluating accuracy")
        acc_base, acc_cmp = evaluate_models(id, split, X_test, y_test, base, cmp, accuracy_score)
        print(f"[{id}:{split}] Evaluating f1-score")
        f1_base, f1_cmp = evaluate_models(id, split, X_test, y_test, base, cmp, f1_score)
        print(f"[{id}:{split}] Evaluating AUC")
        auc_base, auc_cmp = evaluate_models(id, split, X_test, y_test, base, cmp, roc_auc_score)

        scores[f"split{split}"] = [acc_base, acc_cmp, f1_base, f1_cmp, auc_base, auc_cmp]

        print(f"[{id}:{split}] Split completed")

      # convert to dataframe
      scores_df = pd.DataFrame(scores)
      scores_df["mean_score"] = scores_df.mean(numeric_only=True, axis=1)

      # save nu scores
      print(f"[{id}] Saving split scores")
      save_dataframe(id, "baseline_sampling", f"scores_nu{nu:02d}", "/nu", scores_df)

      # hypothesis tests
      acc_p, f1_p, auc_p = hypothesis_test(id, scores_df)

      # fetch mean scores
      acc_cmp_mean = scores_df["mean_score"][1]
      f1_cmp_mean = scores_df["mean_score"][3]
      auc_cmp_mean = scores_df["mean_score"][5]

      # append to list in dict
      results["model"].append(f"nu: {nu}")
      results["acc"].append(acc_cmp_mean)
      results["acc_p"].append(acc_p)
      results["f1"].append(f1_cmp_mean)
      results["f1_p"].append(f1_p)
      results["auc"].append(auc_cmp_mean)
      results["auc_p"].append(auc_p)
  
  # add base scores (maybe edit cause reusing variables)
  acc_base_mean = scores_df["mean_score"][0]
  f1_base_mean = scores_df["mean_score"][2]
  auc_base_mean = scores_df["mean_score"][4]

  results["model"].append("base")
  results["acc"].append(acc_base_mean)
  results["acc_p"].append(0)
  results["f1"].append(f1_base_mean)
  results["f1_p"].append(0)
  results["auc"].append(auc_base_mean)
  results["auc_p"].append(0)

  # convert to dataframe
  print(f"[{id}] Converting results to dataframe")
  results_df = pd.DataFrame(results)

  # save dataframe
  print(f"[{id}] Saving dataframe")
  save_dataframe(id, "baseline_sampling", "results", "", results_df)

  print(f"[{id}] Completed evaluation")

def test_baseline_beta(id):
  print(f"[{id}] Running evaluation over baseline and beta")

  # initialise dicts for dataframes
  scores = {}
  results = {
    "model": [],
    "acc": [],
    "acc_p": [],
    "f1": [],
    "f1_p": [],
    "auc": [],
    "auc_p": [],
  }

  scores["metric"] = ["accuracy_baseline", "accuracy_beta", "f1_baseline", "f1_beta", "auc_baseline", "auc_beta"]

  # loop through all proportion values of nu
  for i in range(19):
      nu = (i + 1) * 5
      print(f"[{id}] Begin at proportion {nu}")

      # loop through all splits
      for split in range(100):
        # fetch models and data
        X_train, X_test, y_train, y_test, base, cmp = fetch_models_and_splits(id, split, load_baseline_model, lambda id, split: load_beta_model(id, split, nu))

        # evaluate over acc, f1 and auc
        print(f"[{id}:{split}] Evaluating accuracy")
        acc_base, acc_cmp = evaluate_models(id, split, X_test, y_test, base, cmp, accuracy_score)
        print(f"[{id}:{split}] Evaluating f1-score")
        f1_base, f1_cmp = evaluate_models(id, split, X_test, y_test, base, cmp, f1_score)
        print(f"[{id}:{split}] Evaluating AUC")
        auc_base, auc_cmp = evaluate_models(id, split, X_test, y_test, base, cmp, roc_auc_score)

        scores[f"split{split}"] = [acc_base, acc_cmp, f1_base, f1_cmp, auc_base, auc_cmp]

        print(f"[{id}:{split}] Split completed")

      # convert to dataframe
      scores_df = pd.DataFrame(scores)
      scores_df["mean_score"] = scores_df.mean(numeric_only=True, axis=1)

      # save nu scores
      print(f"[{id}] Saving split scores")
      save_dataframe(id, "baseline_beta", f"scores_nu{nu:02d}", "/nu", scores_df)

      # hypothesis tests
      acc_p, f1_p, auc_p = hypothesis_test(id, scores_df)

      # fetch mean scores
      acc_cmp_mean = scores_df["mean_score"][1]
      f1_cmp_mean = scores_df["mean_score"][3]
      auc_cmp_mean = scores_df["mean_score"][5]

      # append to list in dict
      results["model"].append(f"nu: {nu}")
      results["acc"].append(acc_cmp_mean)
      results["acc_p"].append(acc_p)
      results["f1"].append(f1_cmp_mean)
      results["f1_p"].append(f1_p)
      results["auc"].append(auc_cmp_mean)
      results["auc_p"].append(auc_p)
  
  # add base scores (maybe edit cause reusing variables)
  acc_base_mean = scores_df["mean_score"][0]
  f1_base_mean = scores_df["mean_score"][2]
  auc_base_mean = scores_df["mean_score"][4]

  results["model"].append("base")
  results["acc"].append(acc_base_mean)
  results["acc_p"].append(0)
  results["f1"].append(f1_base_mean)
  results["f1_p"].append(0)
  results["auc"].append(auc_base_mean)
  results["auc_p"].append(0)

  # convert to dataframe
  print(f"[{id}] Converting results to dataframe")
  results_df = pd.DataFrame(results)

  # save dataframe
  print(f"[{id}] Saving dataframe")
  save_dataframe(id, "baseline_beta", "results", "", results_df)

  print(f"[{id}] Completed evaluation")

def test_baseline_local(id):
  print(f"[{id}] Running evaluation over baseline and local set")

  # initialise dicts for dataframes
  scores = {}
  results = {
    "model": [],
    "acc": [],
    "acc_p": [],
    "f1": [],
    "f1_p": [],
    "auc": [],
    "auc_p": [],
  }

  scores["metric"] = ["accuracy_baseline", "accuracy_local", "f1_baseline", "f1_local", "auc_baseline", "auc_local", "sv_baseline", "sv_local"]

  # loop through all splits
  for split in range(100):
    # fetch models and data
    X_train, X_test, y_train, y_test, base, cmp = fetch_models_and_splits(id, split, load_baseline_model, load_local_filter_model)

    # evaluate over acc, f1 and auc
    print(f"[{id}:{split}] Evaluating accuracy")
    acc_base, acc_cmp = evaluate_models(id, split, X_test, y_test, base, cmp, accuracy_score)
    print(f"[{id}:{split}] Evaluating f1-score")
    f1_base, f1_cmp = evaluate_models(id, split, X_test, y_test, base, cmp, f1_score)
    print(f"[{id}:{split}] Evaluating AUC")
    auc_base, auc_cmp = evaluate_models(id, split, X_test, y_test, base, cmp, roc_auc_score)
    print(f"[{id}:{split}] Calculating SV set size")
    sv_base = len(base.support_)
    sv_cmp = len(cmp.support_)

    scores[f"split{split}"] = [acc_base, acc_cmp, f1_base, f1_cmp, auc_base, auc_cmp, sv_base, sv_cmp]

    print(f"[{id}:{split}] Split completed")

  # convert to dataframe
  scores_df = pd.DataFrame(scores)
  scores_df["mean_score"] = scores_df.mean(numeric_only=True, axis=1)

  # save scores
  print(f"[{id}] Saving split scores")
  save_dataframe(id, "baseline_local_filter", f"scores", "", scores_df)

  # hypothesis tests
  acc_p, f1_p, auc_p = hypothesis_test(id, scores_df)

  # fetch mean scores
  acc_cmp_mean = scores_df["mean_score"][1]
  f1_cmp_mean = scores_df["mean_score"][3]
  auc_cmp_mean = scores_df["mean_score"][5]

  # append to list in dict
  results["model"].append("cmp")
  results["acc"].append(acc_cmp_mean)
  results["acc_p"].append(acc_p)
  results["f1"].append(f1_cmp_mean)
  results["f1_p"].append(f1_p)
  results["auc"].append(auc_cmp_mean)
  results["auc_p"].append(auc_p)
  
  # add base scores (maybe edit cause reusing variables)
  acc_base_mean = scores_df["mean_score"][0]
  f1_base_mean = scores_df["mean_score"][2]
  auc_base_mean = scores_df["mean_score"][4]

  results["model"].append("base")
  results["acc"].append(acc_base_mean)
  results["acc_p"].append(0)
  results["f1"].append(f1_base_mean)
  results["f1_p"].append(0)
  results["auc"].append(auc_base_mean)
  results["auc_p"].append(0)

  # convert to dataframe
  print(f"[{id}] Converting results to dataframe")
  results_df = pd.DataFrame(results)

  # save dataframe
  print(f"[{id}] Saving dataframe")
  save_dataframe(id, "baseline_local_filter", "results", "", results_df)

  print(f"[{id}] Completed evaluation")

def test_lsbo_lsrb(id):
  print(f"[{id}] Running evaluation over lsbo and lsrb")

  # initialise dicts for dataframes
  scores = {}
  results = {
    "model": [],
    "acc": [],
    "acc_p": [],
    "f1": [],
    "f1_p": [],
    "auc": [],
    "auc_p": [],
  }

  scores["metric"] = ["accuracy_baseline", "accuracy_lsbo", "accuracy_lsrb", "f1_baseline", "f1_lsbo", "f1_lsrb", "auc_baseline", "auc_lsbo", "auc_lsrb", "sv_baseline", "sv_lsbo", "sv_lsrb"]

  # loop through all splits
  for split in range(100):
    print(f"[{id}:{split}] Fetching models and data")
    # load dataset split
    X_train, X_test, y_train, y_test, train, test = load_dataset_split(id, split)
    print(f"[{id}:{split}] Data fetched")

    # fetch base model
    print(f"[{id}:{split}] Fetching first model")
    base = load_baseline_model(id, split)
    print(f"[{id}:{split}] First model fetched")

    # fetch lsbo model
    print(f"[{id}:{split}] Fetching LSBo model")
    lsbo = load_local_lsbo_model(id, split)
    print(f"[{id}:{split}] LSBo model fetched")

    # fetch lsrb model
    print(f"[{id}:{split}] Fetching LSRB model")
    lsrb = load_local_lsrb_model(id, split)
    print(f"[{id}:{split}] LSRB model fetched")

    print(f"[{id}:{split}] Fetching complete")

    # evaluate over acc, f1 and auc
    print(f"[{id}:{split}] Evaluating models")
    # get predicted labels
    print(f"[{id}:{split}] Generating predicted labels")
    y_pred_base = base.predict(X_test)
    y_pred_lsbo = lsbo.predict(X_test)
    y_pred_lsrb = lsrb.predict(X_test)

    # evaluate score
    print(f"[{id}:{split}] Evaluating accuracy")
    acc_base = accuracy_score(y_test, y_pred_base)
    acc_lsbo = accuracy_score(y_test, y_pred_lsbo)
    acc_lsrb = accuracy_score(y_test, y_pred_lsrb)

    print(f"[{id}:{split}] Evaluating f1-score")
    f1_base = f1_score(y_test, y_pred_base)
    f1_lsbo = f1_score(y_test, y_pred_lsbo)
    f1_lsrb = f1_score(y_test, y_pred_lsrb)

    print(f"[{id}:{split}] Evaluating AUC")
    auc_base = roc_auc_score(y_test, y_pred_base)
    auc_lsbo = roc_auc_score(y_test, y_pred_lsbo)
    auc_lsrb = roc_auc_score(y_test, y_pred_lsrb)

    print(f"[{id}:{split}] Calculating SV set size")
    sv_base = len(base.support_)
    sv_lsbo = len(lsbo.support_)
    sv_lsrb = len(lsrb.support_)

    scores[f"split{split}"] = [acc_base, acc_lsbo, acc_lsrb, f1_base, f1_lsbo, f1_lsrb, auc_base, auc_lsbo, auc_lsrb, sv_base, sv_lsbo, sv_lsrb]

    print(f"[{id}:{split}] Split completed")

  # convert to dataframe
  scores_df = pd.DataFrame(scores)
  scores_df["mean_score"] = scores_df.mean(numeric_only=True, axis=1)

  # save scores
  print(f"[{id}] Saving split scores")
  save_dataframe(id, "lssm_lsrb", f"scores", "", scores_df)

  # # hypothesis tests
  # acc_p, f1_p, auc_p = hypothesis_test(id, scores_df)

  # # fetch mean scores
  # acc_cmp_mean = scores_df["mean_score"][1]
  # f1_cmp_mean = scores_df["mean_score"][3]
  # auc_cmp_mean = scores_df["mean_score"][5]

  # # append to list in dict
  # results["model"].append("cmp")
  # results["acc"].append(acc_cmp_mean)
  # results["acc_p"].append(acc_p)
  # results["f1"].append(f1_cmp_mean)
  # results["f1_p"].append(f1_p)
  # results["auc"].append(auc_cmp_mean)
  # results["auc_p"].append(auc_p)
  
  # # add base scores (maybe edit cause reusing variables)
  # acc_base_mean = scores_df["mean_score"][0]
  # f1_base_mean = scores_df["mean_score"][2]
  # auc_base_mean = scores_df["mean_score"][4]

  # results["model"].append("base")
  # results["acc"].append(acc_base_mean)
  # results["acc_p"].append(0)
  # results["f1"].append(f1_base_mean)
  # results["f1_p"].append(0)
  # results["auc"].append(auc_base_mean)
  # results["auc_p"].append(0)

  # # convert to dataframe
  # print(f"[{id}] Converting results to dataframe")
  # results_df = pd.DataFrame(results)

  # # save dataframe
  # print(f"[{id}] Saving dataframe")
  # save_dataframe(id, "baseline_local_filter", "results", "", results_df)

  print(f"[{id}] Completed evaluation")
      
def main(args):
  if args.id:
    id = args.id
    print(f"STRATEGY EVALUATION {id}")
    if args.samp:
      test_baseline_sampling(id)
    elif args.beta:
      test_baseline_beta(id)
    elif args.local:
      test_baseline_local(id)
    elif args.ls1:
      test_lsbo_lsrb(id)
    print(f"COMPLETED {id}")
  
def get_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('id', type=int, help='OpenML dataset ID to process')
  parser.add_argument('--samp', action="store_true")
  parser.add_argument('--beta', action="store_true")
  parser.add_argument('--local', action="store_true")
  parser.add_argument('--ls1', action="store_true")
  return parser.parse_args()

if __name__ == "__main__":
  args = get_arguments()
  main(args)