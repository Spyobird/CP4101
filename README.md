### Newer/important files:
ls_train2.py
- trains and saves SVM models on an OpenML dataset
- need to run baseline.py and local_lssm.py first
- contains the main logic of the instance selection algorithm

ls_test2.py
- tests SVM models on test data and saves to csv
- need to run ls_train2.py first

ls_results2.py
- combines multiple csv files from different folders together

baseline.py
- trains a baseline SVM model

local_lssm.py
- runs the LSSM algorithm (a noise filter) over a dataset

### Link to Google Colab files:
https://drive.google.com/drive/folders/1e6UEHcCtL54cbkbZqEvBLZSZkYMkRI06

svm-4-10-voronoi2.ipynb
- the notebook with the latest code for the algorithm
- should be able to make a copy
- link: https://colab.research.google.com/drive/14BBUak11sRh0Goa0shfSIZTpFCiZv6kn
