import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.ensemble import RandomForestClassifier

# ----------------------
# 前処理
# ----------------------

# 男性を0、女性を1
df = pd.read_csv("../data/train.csv").replace({"male": 0, "female": 1})

df["Age"].fillna(df.Age.median(), inplace=True)
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df = df.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)

train_data = df.values
x = train_data[:, 2:] # Pclass以降の変数
y = train_data[:, 1]  # 正解データ

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
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(x, y)

# ----------------------
# テスト
# ----------------------
test_df= pd.read_csv("../data/test.csv").replace("male",0).replace("female",1)
# 欠損値の補完
test_df["Age"].fillna(df.Age.median(), inplace=True)
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1
test_df = test_df.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)

test_data = test_df.values
x_test = test_data[:, 1:]
output = forest.predict(x_test)

zip_data = zip(test_data[:,0].astype(int), output.astype(int))
predict_data = list(zip_data)

# ----------------------
# 予測結果出力
# ----------------------
with open("../data/predict_result_data.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(test_data[:,0].astype(int), output.astype(int)):
        writer.writerow([pid, survived])

