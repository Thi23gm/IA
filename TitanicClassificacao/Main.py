import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

train_data = pd.read_csv("dados/train.csv")

train_data = train_data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

train_data["Age"].fillna(train_data["Age"].mean(), inplace=True)

train_data = pd.get_dummies(train_data, columns=["Sex", "Embarked"], drop_first=True)

X = train_data.drop(["Survived"], axis=1)
y = train_data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 1. Bagging com Árvores de Decisão
bagging_model = BaggingClassifier(n_estimators=100, random_state=42)
bagging_model.fit(X_train, y_train)
bagging_predictions = bagging_model.predict(X_test)
bagging_accuracy = accuracy_score(y_test, bagging_predictions)
print("Acurácia do modelo Bagging:", bagging_accuracy)

# 2. Random Forest
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)
rf_predictions = random_forest_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Acurácia do modelo Random Forest:", rf_accuracy)

# 3. Boosting com XGBoost
xgb_model = XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
print("Acurácia do modelo XGBoost:", xgb_accuracy)
