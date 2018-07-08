# coding:utf-8
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import csv

from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import GridSearchCV

#データの読み込み
data = Dataset.load_builtin('ml-100k')

#グリッドサーチの設定
param_grid = {'n_factors': [50, 100, 150],
              'lr_all': [0.001, 0.005, 0.02]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=2)

#グリッドサーチの実行
gs.fit(data)

# RMSEの最良値
print("best RMSE score: ", gs.best_score['rmse'])

# RMSEが最良だったパラメータの組み合わせ
print("parameters that gave the best RMSE score: ", gs.best_params['rmse'])

result_dict = gs.cv_results

#結果のcsvへの書き込み
with open('svd_result.csv', 'w') as f:
    writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく

    line = [] #最初の行にヘッダを書き込み
    for k in result_dict.keys():
        line.append(str(k))
    writer.writerow(line)

    for i in range(len(result_dict['mean_fit_time'])): #グリッドサーチの結果を書き込み
        line = []
        for k in result_dict.keys():
            line.append(str(result_dict[k][i]))
        writer.writerow(line)

#pandasを利用できる場合以下でも結果の出力が可能
"""
import pandas as pd
results_df = pd.DataFrame.from_dict(gs.cv_results)
print(results_df)
"""
