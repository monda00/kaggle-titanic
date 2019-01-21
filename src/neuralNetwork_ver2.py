'''
ニューラルネットワーク
特徴量を精査版
'''

# data analisis
import pandas as pd
import numpy as np
import random as rnd
import csv
from time import time

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# neural network
import keras.optimizers
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.preprocessing import StandardScaler

# ---------------------------
# 前処理
# ---------------------------

# データの取り込み
train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")
combine = [train_df, test_df]

# TicketとCabinの特徴量を削除
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

# Titleを抽出
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',\
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer',\
                                                 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# Titleを数値に変換
title_mapping = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Rare":5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# NameとPassengerIdを削除
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

# 性別を数値に変換
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female':1, 'male':0}).astype(int)

# 年齢の欠損値を補完
guess_ages = np.zeros((2, 3))

for dataset in combine:
    for i in range(0, 2): # 性別ごと
        for j in range(0, 3): # 客室クラスごと
            guess_df = dataset[(dataset['Sex'] == i) &\
                               (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i,j] = int(age_guess/0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                        'Age'] = guess_ages[i,j]

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
# print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).\
#       mean().sort_values(by='AgeBand', ascending=True))

for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] <= 64, 'Age'] = 4

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]

# 一人かどうかの特徴量を作成
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# ParchとSibSpとFamilySizeを削除
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

# PclassとAgeを合わせた特徴量を作成
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

# Embarked特徴量の欠損値を一般的な値で補完
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

# Embarkedを数値に変換
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

# FareからFareBandを作成
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
# print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).\
#       mean().sort_values(by='FareBand', ascending=True))

for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] <= 31, 'Fare'] = 3

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

# print(test_df["PassengerId"])
# exit()

# ここからパラメータ調整モードと学習モードに分ける------------------------------------------------------

# Test:パラメータ調整、Train:学習
# Mode = "Test"
Mode = "Train"

scaler = StandardScaler()
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'IsAlone', 'Age*Class']

if Mode == "Test":
    df = train_df
    # 学習データ
    df_train = df.iloc[:712, :]
    x_train = scaler.fit_transform(df_train[features].values)
    y_train = df_train["Survived"].values
    y_train_onehot = pd.get_dummies(df_train["Survived"]).values

    # 確認データ
    df_test = df.iloc[712:, :]
    x_test = scaler.fit_transform(df_test[features].values)
    y_test = df_train["Survived"].values
    y_test_onehot = pd.get_dummies(df_test["Survived"]).values
else:
    df_train = train_df
    df_test = test_df

    x_train = scaler.fit_transform(df_train[features].values)
    y_train = df_train["Survived"].values
    y_train_onehot = pd.get_dummies(df_train['Survived']).values

    x_test = scaler.fit_transform(df_test[features].values)


# ---------------------------
# 学習
# ---------------------------

start = time()

# モデル作成
model = Sequential()
model.add(Dense(input_dim=8, units=16, init='he_uniform'))
model.add(Activation("relu"))
model.add(Dense(units=32))
model.add(Activation("relu"))
model.add(Dense(units=16))
model.add(Activation("relu"))
model.add(Dense(units=8))
model.add(Activation("relu"))
model.add(Dense(units=2))
model.add(Activation("softmax"))

adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)

model.compile(loss='categorical_crossentropy', optimizer=adam,\
              metrics=['accuracy'])

if Mode == "Test":
    # 学習
    history = model.fit(x_train, y_train_onehot, batch_size=10, epochs=200,\
                        verbose=1, validation_data=(x_test, y_test_onehot))
    print('\ntime taken %s seconds' % str(time() - start))

    # 予測
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
    model.fit(x_train, y_train_onehot, batch_size=10, epochs=120)
    print('\ntime taken %s seconds' % str(time() - start))
    output = model.predict_classes(x_test)


# ---------------------------
# 評価
# ---------------------------

if Mode == "Train":
    with open("../data/predict_result_data.csv", "w") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(["PassengerId", "Survived"])
        for pid, survived in zip(test_df["PassengerId"].astype(int), output.astype(int)):
            writer.writerow([pid, survived])

