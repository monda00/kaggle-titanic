'''
機械学習の手法を色々試してみる。
sklearnの練習
'''
# data analisis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression # ロジスティック回帰
from sklearn.svm import SVC, LinearSVC # SVMでクラス分類
from sklearn.ensemble import RandomForestClassifier # ランダムフォレスト
from sklearn.neighbors import KNeighborsClassifier # k近傍法
from sklearn.naive_bayes import GaussianNB # ナイーブベイズのガウス分布版
from sklearn.linear_model import Perceptron # パーセプトロン
from sklearn.linear_model import SGDClassifier # 確率的勾配降下法でクラス分類
from sklearn.tree import DecisionTreeClassifier # 決定木


# csvからデータの取り込み
train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")
combine = [train_df, test_df]

'''
まずは、データ分析から
'''

# ---------------------------
# 特徴量の確認
# ---------------------------
# print(train_df.columns.values)
# print(train_df.head())

# ---------------------------
# データの情報確認
# ---------------------------
# print(train_df.info())
# print('_'*40)
# print(test_df.info())
# print('_'*40)
# print(train_df.describe())

# ---------------------------
# 特徴量の解析
# ---------------------------
# print("チケットクラスが生存に関係するか")
# print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# print("性別が生存に関係するか")
# print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# print("チケットクラスと性別を合わせると？")
# print(train_df[['Pclass', 'Sex', 'Survived']].groupby(['Pclass', 'Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# sns.set()
# 年齢（数値特徴量の相関）
# grid = sns.FacetGrid(train_df, col='Survived')
# grid.map(plt.hist, 'Age', bins=20)
# チケットクラスと年齢（順序尺度と数値特徴量の相関）
# grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend()
# 出港地（カテゴリカル特徴量の相関）
# grid = sns.FacetGrid(train_df, row='Embarked', height=2.2, aspect=1.6)
# grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
# grid.add_legend()
# 出港地と性別と運賃（カテゴリカル特徴量と数値の相関）
# grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', height=2.2, aspect=1.6)
# grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
# grid.add_legend()
# plt.show()

# ---------------------------
# データの調整
# ---------------------------

# TicketとCabinの特徴量を削除
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

# Title（肩書）を抽出
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# print("肩書が生存に関わるか")
# print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# NameとPassengerIdを削除
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

# 性別を数値に変換
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# 年齢の欠損値を補完
guess_ages = np.zeros((2,3))

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
# print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))

for dataset in combine:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]

# 既存の特徴量を組み合わせてFamilySizeを作成する
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# print("家族データは生存に関係するか")
# print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# 一人かどうかのデータを作成
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# print("一人できたかどうかは生存に関係するか")
# print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

# PclassとAgeを組み合わせた特徴量を作成
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

# Embarked特徴量の欠損値を最も一般的な値で埋める
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

# print("出向港が生存に関係するか")
# print(train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# Embarked特徴量を数値に変換
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# Fare特徴量からFareBand特徴量を作成
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
# print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

'''
学習
'''

# ---------------------------
# データを読み込む
# ---------------------------
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1).copy()

# ---------------------------
# Logistic Regression
# ---------------------------
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print('Logistic Regression : {}'.format(acc_log))

# 特徴量の有用性を検証
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

print(coeff_df.sort_values(by='Correlation', ascending=False))

# ---------------------------
# Support Vector Machines
# ---------------------------
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_log = round(svc.score(X_train, Y_train) * 100, 2)
print('Support Vector Machines : {}'.format(acc_log))

# ---------------------------
# k-Nearest Neighbors
# ---------------------------
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print('k-Nearest Neighbors : {}'.format(acc_knn))

# ---------------------------
# Naive Bayes classifier
# ---------------------------
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print('Naive Bayes classifier : {}'.format(acc_gaussian))

# ---------------------------
# Perceptron
# ---------------------------
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print('Perceptron : {}'.format(acc_perceptron))

# ---------------------------
# Decision Tree
# ---------------------------
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print('Decision Tree : {}'.format(acc_decision_tree))

# ---------------------------
# Random Forest
# ---------------------------
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print('Random Forest : {}'.format(acc_random_forest))

'''
kaggleに登録するファイルの作成
'''
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('../data/submission.csv', index=False)
