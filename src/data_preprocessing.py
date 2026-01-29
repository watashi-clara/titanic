import pandas as pd

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

women = train_data.loc[train_data.Sex == "female"]["Survived"]
men = train_data.loc[train_data.Sex == "male"]["Survived"]

rate_women = women.sum() / len(women)
rate_men = men.sum() / len(men)

print("% of women who survived:", rate_women)
print("% of men who survived:", rate_men)

features = ["Pclass", "Sex", "SibSp", "Parch"]

y = train_data["Survived"]

X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)

y.to_csv("data/y_train.csv", index=False)
X.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)

print("Features saved to data/")