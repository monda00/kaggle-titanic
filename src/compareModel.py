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

# 特徴量の確認
print(train_df.columns.values)
print(train_df.head())

# データの情報確認
print(train_df.info())
print('_'*40)
print(test_df.info())
print('_'*40)
print(train_df.describe())

# 特徴量の解析
print("チケットクラスが生存に関係するか")
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print("性別が生存に関係するか")
print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print("チケットクラスと性別を合わせると？")
print(train_df[['Pclass', 'Sex', 'Survived']].groupby(['Pclass', 'Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

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
