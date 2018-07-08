#### 知識情報学第11回演習サンプルプログラム ex11.py
#### Programmed by Wu Hongle, 監修　福井健一
#### Last updated: 2016/11/18

#### SOM学習によるglassデータの可視化

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from som import som

# ==================================================================================
df_glass = pd.read_csv("glass.data", header=None)
df_glass.columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']

input_data = df_glass[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']].values
n_feature = input_data.shape[1]
n_class = max(df_glass['Type'].values)
sc = preprocessing.StandardScaler()
input_data = sc.fit_transform(input_data)

somsize_x = 4
somsize_y = 5
som = som(somsize_x, somsize_y, n_feature, learning_rate=1.0)
som.random_weights_init(input_data)
som.train_batch(input_data,1000)
'''
課題：SOMで学習した各ニューロンノードの参照ベクトルのグラフ，および各ニューロンに分類されたデータのクラス分布を描画する
ヒント：som.winner(x) で勝者ニューロンのインデックスを取得できる．
'''
y = np.array([])
fig,axes = plt.subplots(nrows=5,ncols=4, figsize=(12,8), sharex=True)

label = np.array(['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'])
x = np.arange(len(label))
for i in range(somsize_x):
    for j in range(somsize_y):
        y = som.weights[i][j]
        axes[j,i].bar(x, y, tick_label=label, align="center")
        axes[j,i].set_ylim(-2, 2)
#plt.savefig("fig1.png",format = 'png', dpi=300)

z = np.zeros((5,4))
plt.figure(figsize=(12,8))
class_n = df_glass['Type']
for i in range(n_class):
    for idx, row in enumerate(input_data):
        win = som.winner(row)
        if class_n[idx] == i + 1:
            z[win[1],win[0]] += 1
    plt.subplot(3,3,i + 1)
    X, Y = np.meshgrid(np.arange(somsize_x + 1),np.arange(somsize_y + 1))
    plt.pcolor(X, Y, z)
    plt.colorbar()
    z = np.zeros((5,4))
#plt.savefig("fig2.png",format = 'png', dpi=300)
plt.show()
