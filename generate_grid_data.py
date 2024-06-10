# import relevant libraries
# data management
import numpy as np
import pandas as pd

# helper
import argparse
import math
import os
import random
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

# generation code
def generate_grid_data(n=3):
  # initialise matrices
  matrices = []
  base_matrix = np.full((n, n), -1)

  seen = []
  seen.append(base_matrix)
  # generate exhaustively
  for i in range(n):
    for j in range(n):
      for m in seen.copy():
        for c in range(2):
          m_new = m.copy()
          m_new[i, j] = c
          flag = False
          # different symmetries to check
          m_a = np.rot90(m_new)
          m_aa = np.rot90(m_new, 2)
          m_aaa = np.rot90(m_new, 3)
          m_x = np.transpose(m_new)
          m_xa = np.rot90(m_x)
          m_xaa = np.rot90(m_x, 2)
          m_xaaa = np.rot90(m_x, 3)
          # class symmetries
          m_c = np.zeros(m_new.shape) + (m_new == -1)*-1 + (m_new == 0)*1
          m_ca = np.rot90(m_c)
          m_caa = np.rot90(m_c, 2)
          m_caaa = np.rot90(m_c, 3)
          m_cx = np.transpose(m_c)
          m_cxa = np.rot90(m_cx)
          m_cxaa = np.rot90(m_cx, 2)
          m_cxaaa = np.rot90(m_cx, 3)
          for m_s in seen.copy():
            if (m_a == m_s).all():
              flag = True
              break
            if (m_aa == m_s).all():
              flag = True
              break
            if (m_aaa == m_s).all():
              flag = True
              break
            if (m_x == m_s).all():
              flag = True
              break
            if (m_xa == m_s).all():
              flag = True
              break
            if (m_xaa == m_s).all():
              flag = True
              break
            if (m_xaaa == m_s).all():
              flag = True
              break
            if (m_c == m_s).all():
              flag = True
              break
            if (m_ca == m_s).all():
              flag = True
              break
            if (m_caa == m_s).all():
              flag = True
              break
            if (m_caaa == m_s).all():
              flag = True
              break
            if (m_cx == m_s).all():
              flag = True
              break
            if (m_cxa == m_s).all():
              flag = True
              break
            if (m_cxaa == m_s).all():
              flag = True
              break
            if (m_cxaaa == m_s).all():
              flag = True
              break
          if not flag:
            seen.append(m_new)
            if len(seen) % 100 == 0:
              print(len(seen))
  matrices = np.array(seen)

  # remove cases where one class is not present
  mask = np.zeros(len(matrices), dtype=bool)
  for i in range(len(matrices)):
    m = matrices[i]
    if (m == 0).any() and (m == 1).any():
      mask[i] = True
  matrices = matrices[mask]

  return matrices

def main(args):
  if args.n:
    matrices = generate_grid_data(args.n)
    if not os.path.exists(f"datasets/grid"):
      print(f"[{id}] Creating directory for grid")
      os.makedirs(f"datasets/grid", exist_ok=True)
    np.save(f"datasets/grid/{args.n}x{args.n}", matrices)
  
def get_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('n', type=int, help='size of grid')
  return parser.parse_args()

if __name__ == "__main__":
  args = get_arguments()
  main(args)