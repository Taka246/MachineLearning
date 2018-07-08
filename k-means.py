# coding=UTF-8
#### 知識情報学第10回演習サンプルプログラム ex10.py
#### Programmed by Nattapong Thammasan, 監修　福井健一
#### Modified by Wasin Kalintha and Ekasit Phermphoonphiphat
#### Last updated: 2017/12/10

#### K-means法によるWineデータのクラスタリング
#### Python機械学習本：11.1 K-means, 11.1.3 Distortion

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df_wine = pd.read_csv("wine.data", header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
'OD280/OD315 of diluted wines', 'Proline']

used_feature1 = 'Flavanoids'
used_feature2 = 'Proline'
X = df_wine[[used_feature1, used_feature2]].values
y = df_wine['Class label'].values
n_class = 3

sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

#==================================================================
print('K-means')
km = KMeans(n_clusters=4,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=1)
y_km = km.fit_predict(X)

#==================================================================
# 割り当てられたクラスタによりクラスタリング結果を描画

# 描画に使用する色とマークのセット
colors = (["lightgreen", "orange", "lightblue", "m", "b", "g", "c", "y", "w", "k"])
markers = (["s", "o", "v", "^", "D", ">", "<", "d", "p", "H"])

plt.figure(figsize=(8,8))
for idx in range(0, km.cluster_centers_.shape[0]):
    plt.scatter(X[y_km == idx, 0],
                X[y_km == idx, 1],
                s=50,
                c=colors[idx],
                marker=markers[idx],
                label="cluster " + str(idx+1))

plt.scatter(km.cluster_centers_[:, 0],
    km.cluster_centers_[:, 1],
    s=250,
    marker='*',
    c='red',
    label='centroids')
plt.legend()
plt.grid()

#==================================================================
# 課題(a) 正解クラスとクラスタ中心を描画
#[YOUR CODE HERE]
plt.figure(figsize=(8,8))
for idx in range(n_class):
    plt.scatter(X[y == idx + 1, 0],
                X[y == idx + 1 , 1],
                s=50,
                c=colors[idx],
                marker=markers[idx],
                label="class " + str(idx+1))
plt.scatter(km.cluster_centers_[:, 0],
    km.cluster_centers_[:, 1],
    s=250,
    marker='*',
    c='red',
    label='centroids')
plt.legend()
plt.grid()
#==================================================================
# 課題(b) クラスタ数を変えて内部基準SSEをグラフにプロット
# SSEは，scikit-learnのK-meansではintertia_で参照可能
#[YOUR CODE HERE]
plt.figure(figsize=(8,8))
distortions = []

for i  in range(1,11):                
    km = KMeans(n_clusters=i,
                init='k-means++',     
                n_init=10,
                max_iter=300,
                tol=1e-04,
                random_state=1)
    km.fit(X)                         
    distortions.append(km.inertia_)   

plt.plot(range(1,11),distortions,marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
#==================================================================
# 課題(c) クラスタ数を変えて外部基準Purityをプロット
# なお，scikit-learnにPurityは実装されていません．
#[YOUR CODE HERE]

plt.figure(figsize=(8,8))
Purities = []

for i  in range(1,11):
    num_max = []              
    km = KMeans(n_clusters=i,
                init='k-means++',     
                n_init=10,
                max_iter=300,
                tol=1e-04,
                random_state=1)
    km.fit(X)
    y_km = km.fit_predict(X)

    for idx in range(i):
    	num = []
    	for a in range(n_class):
    		p = 0
    		for b in y[y_km == idx]:
    			if b == a:
    				p += 1
    			num.append(p)
    	num_max.append(max(num))
    Purities.append(sum(num_max)/len(X[:, 0]))   
plt.plot(range(1,11),Purities,marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Purity')
plt.show()
