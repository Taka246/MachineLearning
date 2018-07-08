#### 知識情報学第12回演習サンプルプログラム ex12.py
#### Programmed by Nattapong Thammasan, 監修　福井健一
#### Modified by Wasin Kalintha and Ekasit Phermphoonphiphat
#### Last updated: 2018/01/07

#### スーパマーケットの購買履歴データからの頻出項目抽出(Aprioriアルゴリズム)

import numpy as np
from scipy.io import arff

# ===================================================================
# arffデータの読み込み
f = open("supermarket.arff", "r", encoding="utf-8")
data, meta = arff.loadarff(f)
item_name = meta.names()

item_record, class_label = [], []
for x in data:
    w = list(x)
    class_label.append(w[-1])
    w.pop(-1)
    item_record.append(w)

# ===================================================================
# トランザクション ’j’ において、b't' = 1 を持つItemのインデックスの配列をitem_values[j,:]に格納、他は-1の値を持つ
item_values = np.zeros([len(item_record),len(item_record[0])]) - 1
for j in range(len(item_record)):
    b = 0
    for n,i in enumerate(item_record[j]):
    	if i == b't':
            item_values[j,b] += n + 1
            b += 1
    	else:
            continue
# ===================================================================
# 課題(a): 頻出項目集合と支持度を計算する関数
# 引数：候補項目集合list
# 戻り値：候補項目集合の内の最小支持度以上の項目集合list，それに対応する支持度list

#　上記で簡便のため、item_binaryの代わりにitem_valuesも作成した
def scanD(candidates):
    retlist = []
    support_data = []
    listdata = np.zeros(len(candidates))
    minvalue = minsupport * len(item_values)

    # 引数の配列が1次元かそうでないかで場合分け
    if np.array(candidates).ndim == 1:
        for m,candidate in enumerate(candidates):
            for ture_list in  item_values:
                # 任意のトランザクションにおいて、任意のcandidatesの要素を含んでいるか判定。含んでいたらカウント。
                if candidate in ture_list:
                    listdata[m] += 1
        for n,k in enumerate(listdata):
            # 最小支持値以上になりうるかどうかを判定。なりうる場合、リストと支持値をretlist、support_dataに格納
            if k < minvalue: continue
            else:
                retlist.append(candidates[n])
                support_value = k / len(item_values)
                support_data.append(support_value)
        retlist = np.array(retlist).reshape(len(retlist),1)

    # 引数の配列が２次元以上の時
    else:
        for m,candidate in enumerate(candidates):
            candidate_set = set(candidate)
            for ture_list in  item_values:
                # 任意のトランザクションにおいて、任意のcandidatesの要素をすべて含んでいるか判定。含んでいたらカウント。
                if candidate == list(candidate_set & set(ture_list)):
                    listdata[m] += 1
        for n,k in enumerate(listdata):
            # 最小支持値以上になりうるかどうかを判定。なりうる場合、リストと支持値をretlist、support_dataに格納
            if k < minvalue: continue
            else:
                retlist.append(candidates[n])
                support_value = k / len(item_values)
                support_data.append(support_value)

    return retlist, support_data

# サイズkの候補項目集合を生成する関数
# 引数：サイズk-1の頻出項目集合list, 候補項目集合のサイズk
# 戻り値：サイズkの候補項目集合list
# 実行例：
# print (aprioriGen([[0],[1],[2],[4]], 2))
# --> [[0, 1], [0, 2], [0, 4], [1, 2], [1, 4], [2, 4]]
# print (aprioriGen([[0,2] , [1,2] , [1,4] , [2,4]] , 3))
# --> [[1, 2, 4]]
def aprioriGen(freq_sets, k):
    "Generate the joint transactions from candidate sets"
    retList = []
    lenLk = len(freq_sets)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(freq_sets[i])[:k - 2]
            L2 = list(freq_sets[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(list(set(freq_sets[i]) | set(freq_sets[j])))
    return retList

# ===================================================================
# 課題(b):（頻出項目抽出）の実行
# Lに頻出項目集合list，Sにそれに対応する支持度を格納する
# 下記の例の型で格納
# L = [[[0],[1],[2],[4]] , [[0,2] , [1,2] , [1,4] , [2,4]] , [[1,2,4]]]
# S = [[0.5,0.75,0.75,0.75] , [0.5,0.5,0.75,0.5] , [0.5]]

minsupport=0.4
L = ([])
S = ([])
# input_listはscanDに入力するリスト。以下は初回に入力するリスト。
input_list = np.arange(len(item_record[0]))

for i in range(len(item_record[0])):
    # 候補項目集合の内の最小支持度以上の項目集合list，それに対応する支持度listを取得。
    l, s = scanD(input_list)
    # 最小支持度以上の項目集合listがなければ終了。
    if l == []: break
    L.append(list(l))
    S.append(list(s))
    # 次の候補項目集合を作成。
    input_list = aprioriGen(l, i + 2)

# 出力
print('minimum support =', minsupport)
print('Support','\t','Items')
for k,item in enumerate(L):
    if(len(item)==0): break
    print('L=',k+1)
    for i,freqitem in enumerate(item):
        freqitem_name = []
        for val in freqitem:
            freqitem_name.append(item_name[val])
        print(S[k][i], '\t', freqitem_name)
