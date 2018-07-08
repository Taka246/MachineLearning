#### 知識情報学第3回演習サンプルプログラム ex3.py
#### Programmed by Nattapong Thammasan, 監修　福井健一
#### Last updated: 2017/10/12

#### 決定木学習による識別と決定木の描画
#### Python機械学習本：3.6.2節（pp. 84-86）

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.metrics import precision_recall_fscore_support

# テストデータの割合
test_proportion = 0.3
# Iris データセットをロード  
iris = datasets.load_iris()
# 特徴ベクトルを取得
X = iris.data
# クラスラベルを取得
y = iris.target

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_proportion, random_state = 1)

# Zスコアで正規化
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# エントロピーを指標とする決定木のインスタンスを生成し，決定木のモデルに学習データを適合させる
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3)
tree.fit(X_train_std, y_train)

# =====================================================================
# 課題(a) 学習した決定木を用いて学習データおよびテストデータのクラスを予測し，結果をt_train_predicted, y_test_predictedに格納する
y_train_predicted = tree.predict(X_train_std)
y_test_predicted = tree.predict(X_test_std)

# =====================================================================

# テストデータの正解クラスと決定木による予測クラスを出力
print("Test Data")
print("True Label     ", y_test)
print("Predicted Label", y_test_predicted)

# =====================================================================
# 課題(b) 関数precision_recall_fscore_supportを使用して，学習データおよびテストデータに対する
# precision，recall，F値の算出しfscore_train, fscore_testに格納する
fscore_train = precision_recall_fscore_support(y_train,y_train_predicted,average=None)
fscore_test = precision_recall_fscore_support(y_test,y_test_predicted,average=None)


# =====================================================================
# 課題(c) np.averageを使用して，平均precision, recall, F値を算出する

print('Training data')
print('Class 0 Precision: %.3f, Recall: %.3f, Fscore: %.3f' % (fscore_train[0][0], fscore_train[1][0], fscore_train[2][0]))
print('Class 1 Precision: %.3f, Recall: %.3f, Fscore: %.3f' % (fscore_train[0][1], fscore_train[1][1], fscore_train[2][1]))
print('Class 2 Precision: %.3f, Recall: %.3f, Fscore: %.3f' % (fscore_train[0][2], fscore_train[1][2], fscore_train[2][2]))
print('Average Precision: %.3f, Recall: %.3f, Fscore: %.3f' % (np.average(fscore_train[0]), np.average(fscore_train[1]), np.average(fscore_train[2])))
print('Test data')
print('Class 0 Precision: %.3f, Recall: %.3f, Fscore: %.3f' % (fscore_test[0][0], fscore_test[1][0], fscore_test[2][0]))
print('Class 1 Precision: %.3f, Recall: %.3f, Fscore: %.3f' % (fscore_test[0][1], fscore_test[1][1], fscore_test[2][1]))
print('Class 2 Precision: %.3f, Recall: %.3f, Fscore: %.3f' % (fscore_test[0][2], fscore_test[1][2], fscore_test[2][2]))
print('Average Precision: %.3f, Recall: %.3f, Fscore: %.3f' % (np.average(fscore_test[0]), np.average(fscore_test[1]), np.average(fscore_test[2])))

# 学習した決定木モデルをGraphviz形式で出力
# 出力されたtree.dotファイルは，別途Graphviz(gvedit)から開くことで木構造を描画できる
# コマンドラインの場合は，'dot -T png tree.dot -o tree.png'
export_graphviz(tree, out_file='tree.dot', feature_names=['Sepal length', 'Sepal width', 'Petal length', 'Petal width'])
print("tree.dot file is generated")
