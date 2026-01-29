import pandas as pd
import joblib

model = joblib.load("models/model.pkl")

X_test = pd.read_csv("data/X_test.csv")
test_data = pd.read_csv("data/test.csv")

predictions = model.predict(X_test)

output = pd.DataFrame({
    "PassengerId": test_data.PassengerId,
    "Survived": predictions
})

output.to_csv("submission.csv", index=False)

print("Your submission was successfully saved!")