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

# 特徴量の確認
print(train_df.columns.values)
print(train_df.head())

# データの情報確認
print(train_df.info())
print('_'*40)
print(test_df.info())
print('_'*40)
print(train_df.describe())
