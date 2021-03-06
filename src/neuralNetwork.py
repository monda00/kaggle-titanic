import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import keras.optimizers
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.preprocessing import StandardScaler
from time import time

# ----------------------
# 前処理
# ----------------------

# Test: パラメータ調整、Train: 学習
Mode = "Test"
# Mode = "Train"

# 学習する特徴量
features = ['Pclass', 'Sex', 'Age', 'FamilySize', 'Fare']
scaler = StandardScaler()

if Mode == "Test":
    ####
    # 学習データ分割あり
    ####
    df = pd.read_csv("../data/train.csv").replace({"male": 0, "female": 1, "S": 0, "C": 1, "Q": 2})

    # 学習データ
    df_train = df.iloc[:712, :]

    # データの調整
    df_train["Age"].fillna(df_train.Age.median(), inplace=True)
    df_train["FamilySize"] = df_train["SibSp"] + df_train["Parch"] + 1

    x_train = scaler.fit_transform(df_train[features].values)
    y_train = df_train["Survived"].values
    y_train_onehot = pd.get_dummies(df_train['Survived']).values

    # 確認データ
    df_test = df.iloc[712:, :]

    # データの調整
    df_test["Age"].fillna(df_test.Age.median(), inplace=True)
    df_test["FamilySize"] = df_test["SibSp"] + df_test["Parch"] + 1

    x_test = scaler.transform(df_test[features].values)
    y_test = df_test['Survived'].values
    y_test_onehot = pd.get_dummies(df_test['Survived']).values

else:
    ####
    # 学習データ分割なし
    ####

    # 学習データ
    df_train = pd.read_csv("../data/train.csv").replace({"male": 0, "female": 1, "S": 0, "C": 1, "Q": 2})

    df_train["Age"].fillna(df_train.Age.median(), inplace=True)
    df_train["FamilySize"] = df_train["SibSp"] + df_train["Parch"] + 1

    x_train = scaler.fit_transform(df_train[features].values)
    y_train = df_train["Survived"].values
    y_train_onehot = pd.get_dummies(df_train['Survived']).values

    # 提出用のテストデータ
    df_test= pd.read_csv("../data/test.csv").replace("male",0).replace("female",1)

    test_data = df_test.values
    df_test["Age"].fillna(df_test.Age.median(), inplace=True)
    df_test["FamilySize"] = df_test["SibSp"] + df_test["Parch"] + 1

    x_test = scaler.fit_transform(df_test[features].values)

'''学習データ可視化
split_data = []
for survived in [0, 1]:
    # 生存したかどうかで配列を分ける
    split_data.append(df[df.Survived==survived])

tmp = [data[feature].dropna() for data in split_data]
plt.hist(tmp, histtype="barstacked", bins=5)
plt.show()
'''

# ----------------------
# 学習
# ----------------------

start = time()

# モデル作成
model = Sequential()
model.add(Dense(input_dim=5, units=16, init='he_uniform'))
model.add(Activation("relu"))
model.add(Dense(units=2))
model.add(Activation("softmax"))

# sgd = keras.optimizers.SGD(lr=0.01, momentum=0.5, decay=0, nesterov=False)
adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)

model.compile(loss='categorical_crossentropy', optimizer=adam,
              metrics=['accuracy'])

# 予測
if Mode == "Test":
    # 学習
    history = model.fit(x_train, y_train_onehot, batch_size=50, epochs=100, verbose=1, validation_data=(x_test, y_test_onehot))
    print('\ntime taken %s seconds' % str(time() - start))

    y_prediction = model.predict_classes(x_test)
    print("\n\naccuracy", np.sum(y_prediction == y_test) / float(len(y_test)))

    # 学習結果の描画
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

else:
    # 学習
    model.fit(x_train, y_train_onehot, batch_size=50, epochs=100)
    print('\ntime taken %s seconds' % str(time() - start))
    output = model.predict_classes(x_test)

# ----------------------
# 予測結果出力
# ----------------------
if Mode == "Train":
    with open("../data/predict_result_data.csv", "w") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(["PassengerId", "Survived"])
        for pid, survived in zip(test_data[:,0].astype(int), output.astype(int)):
            writer.writerow([pid, survived])
