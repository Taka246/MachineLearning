#### 知識情報学第4回演習サンプルプログラム ex4.py
#### Programmed by Nattapong Thammasan, 監修　福井健一
#### Last updated: 2017/10/19
#### Checked with Python 3.6, scikit-learn 0.19

#### ナイーブベイズ分類器による識別とROC,AUCによる評価
#### 天候とゴルフプレイのラベル特徴データを使用

import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from scipy.io import arff
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_curve,auc,roc_auc_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB

# arffデータの読み込み
f = open("weather.nominal.arff", "r", encoding="utf-8")
data, meta = arff.loadarff(f)

# ラベルエンコーダの設定
le = [LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder(),LabelEncoder()]
for idx,attr in enumerate(meta):
    le[idx].fit(list(meta._attributes[attr][1]))

class_array = np.array([])
feature_array = np.zeros((0,4))

# LabelEncoderを使ってラベル特徴を数値に変換
# 例えば，変数outlookの値{sunny, overcast, rainy}は，{0,1,2}に変換される
for x in data:
    w = list(x)
    class_array = np.append(class_array, le[-1].transform(w[-1].decode("utf-8").split()))
    w.pop(-1)
    for idx in range(0, len(w)):
        w[idx] = le[idx].transform(w[idx].decode("utf-8").split())
    temp = np.array(w)
    feature_array = np.append(feature_array, np.ravel(temp).reshape(1,-1), axis=0)

# OneHotEncoderを使ってLabelEncoderで数値化したラベル特徴をさらに変換
# sunnyは{1,0,0}，overcastは{0,1,0},rainyは{0,0,1}に変換される
# 順序を持たないラベル変数の場合はLabelEncoderだけでは不適切
enc = OneHotEncoder()
feature_encoded = enc.fit_transform(feature_array).toarray()

# =====================================================================
# 1個抜き交差検証（Leave one out cross-validation）
# 全N個のデータから1個を除いた(N-1)個を学習データとしてモデルを学習し，
# 残り1個で学習したモデルのテストを行う．これをN回繰り返す．

print("Leave-one-out Cross-validation")
y_train_post_list,y_train_list,y_test_post_list,y_test_list = [],[],[],[]

loo = LeaveOneOut()
for train_index, test_index in loo.split(feature_encoded):
    X_train, X_test = feature_encoded[train_index], feature_encoded[test_index]
    y_train, y_test = class_array[train_index], class_array[test_index]

    # =====================================================================
    # 課題(a) ナイーブベイズ分類器のインスタンスを生成し，学習データに適合させる．
    # ベルヌーイナイーブベイズ（BernoulliNB）を使用する．
    # alpha(>0)はスムージングのパラメータ．
    # ただし，スライドの等価標本サイズmと定義が逆数(m=1/alpha)のため注意．
    # http://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes
    # fit_prior=Trueに指定すると学習データから事前確率を求める．
    # class_priorは，class_prior=[0.2,0.8]の形で事前確率を指定する．fit_prior=Falseのときに有効．
    bnb = BernoulliNB(alpha=1.0, class_prior=[0.2,0.8], fit_prior=False)
	
	

    # =====================================================================
    # 課題(b) 学習データとテストデータに対する各クラスの事後確率を算出
    posterior_trn = bnb.fit(X_train,y_train).predict_proba(X_train)
    posterior_tst = bnb.fit(X_train,y_train).predict_proba(X_test)

    # テストデータの正解クラスと事後確率を出力
    print("True Label:", y_test)
    print("Posterior Probability:", posterior_tst)

    # 正解クラスと事後確率を保存
    y_train_post_list.extend(posterior_trn[:,[1]])
    y_train_list.extend(y_train)
    y_test_post_list.append(posterior_tst[0][1])
    y_test_list.extend(y_test)

# =====================================================================
# 課題(c) ROC曲線の描画とAUCの算出．scikit-learnにモジュールがあります．
fpr_trn, tpr_trn, thresholds_trn = roc_curve(y_train_list, y_train_post_list)
roc_auc_trn = auc(fpr_trn, tpr_trn)
plt.plot(fpr_trn, tpr_trn, 'k--',label='ROC for training data (AUC = %0.2f)' % roc_auc_trn, lw=2, linestyle="-")

fpr_tst, tpr_tst, thresholds_tst = roc_curve(y_test_list, y_test_post_list)
roc_auc_tst = auc(fpr_tst, tpr_tst)
plt.plot(fpr_tst, tpr_tst, 'k--',label='ROC for test data (AUC = %0.2f)' % roc_auc_tst, lw=2, linestyle="--")

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")

plt.show()
