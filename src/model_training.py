import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

X = pd.read_csv("data/X_train.csv")
y = pd.read_csv("data/y_train.csv").squeeze("columns")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=1
)

model.fit(X, y)

joblib.dump(model, "models/model.pkl")

print("Model saved to models/model.pkl")