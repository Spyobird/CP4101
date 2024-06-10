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
from scipy.stats import t, wilcoxon

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
  
# save a dataframe
def save_dataframe(path, name, df):
  if not os.path.exists(f"results/{path}/"):
    print(f"Creating directory for results")
    os.makedirs(f"results/{path}/", exist_ok=True)

  print(f"Saving dataframe to csv")
  df.to_csv(f"results/{path}/{name}.csv")
  print(f"Dataframe saved")

# load a dataframe
def load_dataframe(id, path, name, opt):
  print(f"[{id}] Loading dataframe")
  df = pd.read_csv(f"results/{path}/{id}{opt}/{name}.csv", index_col=0)
  print(f"[{id}] Dataframe loaded")
  return df

# ref https://www.jmlr.org/papers/volume18/16-305/16-305.pdf
def corrected_std(differences):
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.
    """
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + 1 / 9) # roughly 1 in 9 in 10-fold CV
    corrected_std = np.sqrt(corrected_var)
    return corrected_std


def compute_corrected_ttest(differences, mu=0):
    """Computes right-tailed paired t-test with corrected variance.

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    mu : int
        Mu to test null hypothesis.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    mean = np.mean(differences)
    std = corrected_std(differences)
    if std == 0:
      return 0, 1
    df = len(differences) - 1
    t_stat = (mean - mu) / std
    p_val = t.sf(t_stat, df)  # right-tailed t-test
    return t_stat, p_val

def process(epsilon=0.05):
  path = "results/ls_new_1"
  dir_list = os.listdir(path)
  id_list = np.sort(np.array([int(s) for s in dir_list if os.path.isdir(os.path.join(path, s))]))
  print("IDs found:", id_list)
  
  results = {}
  columns = [
    "accuracy_base",
    "train_size_base",
    "sv_len_base",
    "accuracy_lssm",
    "train_size_lssm",
    "sv_len_lssm",
    "accuracy_A_lsbo",
    "diff_acc_A_lsbo_base",
    "p_A_lsbo_base",
    "diff_acc_A_lsbo_lssm",
    "p_A_lsbo_lssm",
    "train_size_A_lsbo",
    "sv_len_A_lsbo",
    "reduction_A_lsbo",
    "accuracy_A_full_B_A",
    "diff_acc_A_full_B_A_base",
    "p_A_full_B_A_base",
    "diff_acc_A_full_B_A_lssm",
    "p_A_full_B_A_lssm",
    "train_size_A_full_B_A",
    "sv_len_A_full_B_A",
    "reduction_A_full_B_A",
    "accuracy_A_inliers_B_A",
    "diff_acc_A_inliers_B_A_base",
    "p_A_inliers_B_A_base",
    "diff_acc_A_inliers_B_A_lssm",
    "p_A_inliers_B_A_lssm",
    "train_size_A_inliers_B_A",
    "sv_len_A_inliers_B_A",
    "reduction_A_inliers_B_A",
  ]
  for id in id_list:
    df = load_dataframe(id, "ls_new_1", "scores", "")
    assert len(df) == 28

    baseline = df.iloc[0][1:-1].to_numpy()
    lssm = df.iloc[1][1:-1].to_numpy()
    A_lsbo = df.iloc[2][1:-1].to_numpy()
    A_full_B_A = df.iloc[3][1:-1].to_numpy()
    A_inliers_B_A = df.iloc[4][1:-1].to_numpy()

    baseline_eps = df.iloc[0][-1:]["mean_score"] * epsilon
    lssm_eps = df.iloc[1][-1:]["mean_score"] * epsilon

    row = [
      df.iloc[0][-1:]["mean_score"],
      df.iloc[20][-1:]["mean_score"],
      df.iloc[15][-1:]["mean_score"],
      df.iloc[1][-1:]["mean_score"],
      df.iloc[21][-1:]["mean_score"],
      df.iloc[16][-1:]["mean_score"],
      df.iloc[2][-1:]["mean_score"],
      df.iloc[0][-1:]["mean_score"] - df.iloc[2][-1:]["mean_score"],
      compute_corrected_ttest(baseline - A_lsbo, baseline_eps)[1],
      df.iloc[1][-1:]["mean_score"] - df.iloc[2][-1:]["mean_score"],
      compute_corrected_ttest(lssm - A_lsbo, lssm_eps)[1],
      df.iloc[22][-1:]["mean_score"],
      df.iloc[17][-1:]["mean_score"],
      df.iloc[25][-1:]["mean_score"],
      df.iloc[3][-1:]["mean_score"],
      df.iloc[0][-1:]["mean_score"] - df.iloc[3][-1:]["mean_score"],
      compute_corrected_ttest(baseline - A_full_B_A, baseline_eps)[1],
      df.iloc[1][-1:]["mean_score"] - df.iloc[3][-1:]["mean_score"],
      compute_corrected_ttest(lssm - A_full_B_A, lssm_eps)[1],
      df.iloc[23][-1:]["mean_score"],
      df.iloc[18][-1:]["mean_score"],
      df.iloc[26][-1:]["mean_score"],
      df.iloc[4][-1:]["mean_score"],
      df.iloc[0][-1:]["mean_score"] - df.iloc[4][-1:]["mean_score"],
      compute_corrected_ttest(baseline - A_inliers_B_A, baseline_eps)[1],
      df.iloc[1][-1:]["mean_score"] - df.iloc[4][-1:]["mean_score"],
      compute_corrected_ttest(lssm - A_inliers_B_A, lssm_eps)[1],
      df.iloc[24][-1:]["mean_score"],
      df.iloc[19][-1:]["mean_score"],
      df.iloc[27][-1:]["mean_score"],
    ]

    results[f"{id}"] = row

  results_df = pd.DataFrame.from_dict(results, orient="index", columns=columns)
  save_dataframe("ls_new_1", "results", results_df)
      
def main(args):
  print(f"RESULTS")
  process()
  print(f"COMPLETED")
  
def get_arguments():
  parser = argparse.ArgumentParser()
  return parser.parse_args()

if __name__ == "__main__":
  args = get_arguments()
  main(args)