'''This module runs a 5-Fold CV for all the algorithms (default parameters) on
the movielens datasets, and reports average RMSE, MAE, and total computation
time.  It is used for making tables in the README.md file'''

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import time
import datetime
import random

import numpy as np
import six
from tabulate import tabulate

from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise import NormalPredictor
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNBaseline
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering
from surprise import Reader, Dataset



def read_ratings_file_to_surprise(filepath):
    reader = Reader(line_format=u'user item rating',rating_scale=(0, 4), \
    sep=',', skip_lines = 1)
    surprise_data = Dataset.load_from_file(filepath, reader=reader)
    return surprise_data

# The algorithms to cross-validate
classes = (SVD, SVDpp, NMF, BaselineOnly)
# set RNG
np.random.seed(0)
random.seed(0)

filepath = 'data/df_ratings_subset.csv'
data = read_ratings_file_to_surprise(filepath)
kf = KFold(random_state=0)  # folds will be the same for all algorithms.


table = []
for klass in classes:
    klass_name = klass.__name__
    print('{} about to cross_validate'.format(klass_name))
    start = time.time()
    out = cross_validate(klass(), data, ['rmse', 'mae'], kf)
    print('{} done cross_validating'.format(klass_name))
    cv_time = str(datetime.timedelta(seconds=int(time.time() - start)))
    mean_rmse = '{:.3f}'.format(np.mean(out['test_rmse']))
    mean_mae = '{:.3f}'.format(np.mean(out['test_mae']))

    new_line = [klass.__name__, mean_rmse, mean_mae, cv_time]
    print(tabulate([new_line], tablefmt="pipe"))  # print current algo perf
    table.append(new_line)

header = ['class',
          'RMSE',
          'MAE',
          'Time'
          ]
print(tabulate(table, header, tablefmt="pipe"))



'''
Results:


[Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  3.5min
[Parallel(n_jobs=-1)]: Done  84 out of  84 | elapsed:  7.0min finished
Best SVD score: {'rmse': 1.3722718945683983, 'mae': 1.1763148852318437}
[Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed: 15.4min
[Parallel(n_jobs=-1)]: Done  84 out of  84 | elapsed: 29.5min finished
Best SVDpp score: {'rmse': 1.3717566221129804, 'mae': 1.1701131657590611}
[Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  2.5min
[Parallel(n_jobs=-1)]: Done  75 out of  75 | elapsed:  4.2min finished
Best NMF score: {'rmse': 1.5618000687601814, 'mae': 1.2889719501381725}


initial results on
- 10 pct of data
- reader = Reader(line_format=u'user item rating',rating_scale=(0, 4), sep=',', skip_lines = 1)
- cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
Evaluating RMSE, MAE of algorithm SVD on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std
RMSE (testset)    1.4776  1.4831  1.4832  1.4806  1.4821  1.4813  0.0021
MAE (testset)     1.2546  1.2589  1.2598  1.2575  1.2601  1.2582  0.0020
Fit time          36.06   36.31   35.97   36.03   36.56   36.19   0.22
Test time         1.95    2.03    1.77    1.77    2.03    1.91    0.12
Ini

'''
