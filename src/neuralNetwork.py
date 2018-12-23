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
# Mode = "Test"
Mode = "Train"

# 学習する特徴量
features = ['Pclass', 'Sex', 'Age', 'FamilySize']
scaler = StandardScaler()

if Mode == "Test":
    ####
    # 学習データ分割あり
    ####
    df = pd.read_csv("../data/train.csv").replace({"male": 0, "female": 1})

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

else:
    ####
    # 学習データ分割なし
    ####

    # 学習データ
    df_train = pd.read_csv("../data/train.csv").replace({"male": 0, "female": 1})

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
model.add(Dense(input_dim=4, units=2))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='sgd',
              metrics=['accuracy'])

# 学習
model.fit(x_train, y_train_onehot, epochs=500)

print('\ntime taken %s seconds' % str(time() - start))

# 予測
if Mode == "Test":
    y_prediction = model.predict_classes(x_test)
    print("\n\naccuracy", np.sum(y_prediction == y_test) / float(len(y_test)))
else:
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
