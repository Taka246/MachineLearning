#!/usr/bin/env python
# -*- coding: utf-8 -*-

#出力結果はpdf形式でcurrent directoryに保存されます。

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
from graphviz import Digraph
from sklearn.externals.six import StringIO
from sklearn.metrics import (roc_curve, auc, accuracy_score)

def preprocess(df):
    #欠損値処理
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna('S')

    #カテゴリ変数の変換
    df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    df = df.drop(['Cabin','Name','PassengerId','Ticket'],axis=1)
    train_X = df.drop('Survived', axis=1)
    train_y = df.Survived
    #訓練データとテストデータに分割
    (train_X, test_X ,train_y, test_y) = train_test_split(train_X, train_y, test_size = 0.3, random_state = 666)
    return (train_X, test_X ,train_y, test_y)

def visualize(clf,depth):
    #可視化
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,feature_names=train_X.columns, max_depth=depth)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("graph.pdf")

##################
def classifier(train_X,train_y):
    out = DecisionTreeClassifier(max_depth=3).fit(train_X, train_y)
    return out
    #学習済みモデルをreturnしてください。
##################

#テストデータで評価
def evaluate(clf,test_X,test_y):
    pred = clf.predict(test_X)
    fpr, tpr, thresholds = roc_curve(test_y, pred, pos_label=1)
    print("auc = "+str(auc(fpr, tpr)))
    print("accuracy = "+str(accuracy_score(pred, test_y)))

if __name__ == '__main__':
    df = pd.read_csv('titanic.csv')
    (train_X, test_X ,train_y, test_y) = preprocess(df)

    clf = classifier(train_X, train_y)

    evaluate(clf, test_X, test_y)

##################
#depthはパラメータですので、適宜設定してください。
    depth = 3
##################
    visualize(clf,depth)
