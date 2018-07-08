#### 知識情報学第15回演習サンプルプログラム ex15.py
#### Programmed by Nattapong Thammasan, 監修　福井健一
#### Last updated: 2017/06/27
#### Checked with Python 3.6, scikit-learn 0.18, chainer 2.00

#### This code id based on https://gist.github.com/matsuken92/3b945f3ea4d07e9dcc0a

#### AutoEncoderにより事前学習を行う3層ニューラルネットワークによるMNISTデータの識別

import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import optimizers, Variable, Chain, using_config
import matplotlib.pyplot as plt
import struct,os

# ======================================================================
# MNISTデータの読み込み関数
def load_mnist(path, kind='train', ndim='n_dim'):

    labels_path = os.path.join(path,'%s-labels-idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte'% kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
        labels_cvrt = labels.astype(np.int32);

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), n_dim)
        images_cvrt = images.astype(np.float32);

    return images_cvrt, labels_cvrt
# ======================================================================

# MNISTデータの次元数（ピクセル数）
n_dim = 784

# MNISTデータの読み込み
current_path = os.path.dirname(os.path.realpath(__file__))
x_trn, y_trn = load_mnist(current_path, kind='train')
x_tst, y_tst = load_mnist(current_path, kind='t10k')

n_training_data = 1000
n_test_data = 1000

x_train = x_trn[:n_training_data][:]
x_test = x_tst[:n_test_data][:]
y_train = x_train
y_test = x_test

N = len(y_train)
N_test = len(y_test)

# 確率的勾配降下法で学習させる際の１回分のバッチサイズ
batchsize = 200

# 学習の繰り返し回数
n_epoch   = 100

# 中間層のノード数
n_units1   = 500

# ======================================================================
# AutoEncoderのモデルの設定
# 入力 784次元、出力 784次元

# AutoEncoderのネットワーク構造
model_ae = Chain(l1=L.Linear(n_dim, n_units1),
                 l2=L.Linear(n_units1, n_units1),
                 l3=L.Linear(n_units1, n_dim))

# 活性化関数とDropOutの設定
# relu は relu(x) = max(0,x)を返す活性化関数
def forward(x_data, y_data, train=True):
    x, t = Variable(x_data), Variable(y_data)
    h1 = F.dropout(F.relu(model_ae.l1(x)), ratio=0.2)
    h2 = F.dropout(F.relu(model_ae.l2(h1)), ratio=0.2)
    x_hat = F.dropout(F.relu(model_ae.l3(h2)), ratio=0.5)

    # 誤差関数として二乗誤差関数を用いる
    return F.mean_squared_error(x_hat, t)

# パラメータの最適化アルゴリズムの選択
optimizer = optimizers.Adam()
optimizer.setup(model_ae)

l1_W,l2_W,l3_W,l1_b,l2_b,l3_b = [],[],[],[],[],[]
train_loss,test_loss,test_mean_loss = [],[],[]

# ======================================================================
# AutoEncoderの学習
print("******* Learning AutoEncoder *******")

for epoch in range(1, n_epoch+1):
    print('epoch', epoch)

    # 学習
    perm = np.random.permutation(N)
    sum_loss = 0
    for i in range(0, N, batchsize):
        x_batch = x_train[perm[i:i+batchsize]]
        y_batch = y_train[perm[i:i+batchsize]]

        model_ae.cleargrads()
        loss = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()

        train_loss.append(loss.data)
        sum_loss += float(loss.data) * batchsize

    print('\ttrain mean loss={0:.3f} '.format(sum_loss / N))

    # 評価（テストデータの評価は学習のループ終了後に1回だけ実行してもよい）
    sum_loss     = 0
    for i in range(0, N_test, batchsize):
        x_batch = x_test[i:i+batchsize]
        y_batch = y_test[i:i+batchsize]
        loss = forward(x_batch, y_batch, train=False)

        test_loss.append(loss.data)
        sum_loss += float(loss.data) * batchsize

    loss_val = sum_loss / N_test
    test_mean_loss.append(loss_val)
    print('\ttest  mean loss={0:.3f}'.format(loss_val))

    l1_W.append(model_ae.l1.W)
    l2_W.append(model_ae.l2.W)
    l3_W.append(model_ae.l3.W)

# 平均損失関数値のグラフ出力
plt.style.use('ggplot')
plt.figure(figsize=(8,6))
plt.title("")
plt.ylabel("mean loss")
plt.xlabel("epoch")
plt.plot(test_mean_loss, lw=1)

# ======================================================================
# 入力と再構成した画像を描画する関数
def draw_digit_ae(data, n, row, col, _type):
    size = 28
    plt.subplot(row, col, n)
    Z = data.reshape(size,size)
    Z = Z[::-1,:]
    plt.xlim(0,28)
    plt.ylim(0,28)
    plt.pcolor(Z)
    plt.title("type=%s"%(_type), size=8)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")
# ======================================================================
# AutoEncoderの再構成画像を描画
plt.figure(figsize=(20,20))
num = 10
cnt = 0
ans_list  = []
pred_list = []
for idx in np.random.permutation(N_test)[:num]:
    with using_config('train', False):
        xxx = x_test[idx].astype(np.float32)
        h1 = F.dropout(F.relu(model_ae.l1(Variable(xxx.reshape(1,n_dim)))))
        h2 = F.dropout(F.relu(model_ae.l2(h1)))
        y  = model_ae.l3(h2)
        cnt+=1
        ans_list.append(x_test[idx])
        pred_list.append(y)

cnt = 0
for i in range(int(num/10)):
    for j in range (10):
        img_no = i*10+j
        pos = (2*i)*10+j
        draw_digit_ae(ans_list[img_no],  pos+1, 20, 10, "ans")

    for j in range (10):
        img_no = i*10+j
        pos = (2*i+1)*10+j
        draw_digit_ae(pred_list[i*10+j].data, pos+1, 20, 10, "reconst.")

# ======================================================================
# 教師情報を用いた識別器の学習モデルを準備
# 入力 784次元、出力 10次元

y_train = y_trn[:n_training_data][:]
y_test = y_tst[:n_test_data][:]

N = len(y_train)
N_test = len(y_test)

# 識別用ニューラルネットの構造（出力層以外はAutoEncoderと構成）
model_mlp = Chain(l1=L.Linear(n_dim,n_units1),
                  l2=L.Linear(n_units1, n_units1),
                  l3=L.Linear(n_units1, 10))

# AutoEncoderの学習結果から重みをコピー
model_mlp.l1.W.data = model_ae.l1.W.data[:]
model_mlp.l1.b.data = model_ae.l1.b.data[:]
model_mlp.l2.W.data = model_ae.l2.W.data[:]
model_mlp.l2.b.data = model_ae.l2.b.data[:]

# 活性化関数とDropOutの設定
def forward_mlp(x_data, y_data, train=True):
    x, t = Variable(x_data), Variable(y_data)
    h1 = F.dropout(F.relu(model_mlp.l1(x)))
    h2 = F.dropout(F.relu(model_mlp.l2(h1)))
    y  = model_mlp.l3(h2)
    # 多クラス分類なので誤差関数としてソフトマックス関数を使用
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

# パラメータ最適化アルゴリズムの選択
optimizer_mlp = optimizers.Adam()
optimizer_mlp.setup(model_mlp)

# ======================================================================
print("******* Learning Classifiler *******")
train_loss, train_acc, test_loss, test_acc = [], [], [], []

# Learning loop
for epoch in range(1, n_epoch+1):
    print('epoch', epoch)

    # 学習データN個の順番をランダムに並び替える
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    # 0〜Nまでのデータをバッチサイズごとに使って学習
    for i in range(0, N, batchsize):
        x_batch = x_train[perm[i:i+batchsize]]
        y_batch = y_train[perm[i:i+batchsize]]

        # 勾配を初期化
        model_mlp.cleargrads()
        # 順伝播させて誤差と精度を算出
        loss, acc = forward_mlp(x_batch, y_batch)
        # 誤差逆伝播で勾配を計算
        loss.backward()
        optimizer_mlp.update()

        sum_accuracy += acc.data * batchsize
        sum_loss     += loss.data * batchsize

    # 損失関数値と分類精度を表示
    print('train mean loss={0:.3f}, accuracy={1:.3f}'.format(sum_loss / N, sum_accuracy / N))
    train_acc.append(sum_accuracy/N)

    # 評価
    sum_accuracy = 0
    sum_loss     = 0
    for i in range(0, N_test, batchsize):
        x_batch = x_test[i:i+batchsize]
        y_batch = y_test[i:i+batchsize]

        # 順伝播させて誤差と精度を算出
        loss, acc = forward_mlp(x_batch, y_batch, train=False)

        sum_accuracy += acc.data * batchsize
        sum_loss     += loss.data * batchsize

    # テストデータに対する損失関数値と分類精度を表示
    print('test  mean loss={0:.3f}, accuracy={1:.3f}'.format(sum_loss / N_test, sum_accuracy / N_test))
    test_acc.append(sum_accuracy/N)

# ======================================================================
# 学習データおよびテストデータに対する分類精度をグラフ出力
plt.figure(figsize=(8,6))
plt.plot(range(len(train_acc)), train_acc)
plt.plot(range(len(test_acc)), test_acc)
plt.legend(["train_acc","test_acc"],loc=4)
plt.title("Accuracy of digit recognition.")
plt.show()
