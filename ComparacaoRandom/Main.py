import pandas as pd
import io
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

data = """
dia,aparencia,temperatura,umidade,ventando,jogar
d1,sol,Quente,Alta,nao,nao
d2,sol,Quente,Alta,sim,nao
d3,Nublado,Quente,Alta,nao,sim
d4,Chuva,Agradavel,Alta,nao,sim
d5,Chuva,Fria,Normal,nao,sim
d6,Chuva,Fria,Normal,sim,nao
d7,Nublado,Fria,Normal,sim,sim
d8,sol,Agradavel,Alta,nao,nao
d9,sol,Fria,Normal,nao,sim
d10,Chuva,Agradavel,Normal,nao,sim
d11,sol,Agradavel,Normal,sim,sim
d12,Nublado,Agradavel,Alta,sim,sim
d13,Nublado,Quente,Normal,nao,sim
d14,Chuva,Agradavel,Alta,sim,nao
"""

df = pd.read_csv(io.StringIO(data), sep=",")

label_encoders = {}
for column in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

X = df.drop(["dia", "jogar"], axis=1)
y = df["jogar"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

rf = RandomForestClassifier()
nb = CategoricalNB()
dt = DecisionTreeClassifier()

stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

param_grid_rf = {
    "n_estimators": [10, 50, 100, 200],
    "max_features": ["sqrt", "log2"],
    "max_depth": [2, 4, 6, 8, 10, None],
    "bootstrap": [True, False],
}
param_grid_dt = {
    "criterion": ["gini", "entropy"],
    "splitter": ["best", "random"],
    "max_depth": [2, 4, 6, 8, 10, None],
}

rf_grid_search = GridSearchCV(rf, param_grid_rf, cv=stratified_kfold)
rf_random_search = RandomizedSearchCV(rf, param_grid_rf, n_iter=10, cv=stratified_kfold)

dt_grid_search = GridSearchCV(dt, param_grid_dt, cv=stratified_kfold)
dt_random_search = RandomizedSearchCV(dt, param_grid_dt, n_iter=10, cv=stratified_kfold)

rf_grid_search.fit(X_train, y_train)
rf_random_search.fit(X_train, y_train)

dt_grid_search.fit(X_train, y_train)
dt_random_search.fit(X_train, y_train)

nb.fit(X_train, y_train)

models = {
    "Random Forest GridSearch": rf_grid_search,
    "Random Forest RandomSearch": rf_random_search,
    "Decision Tree GridSearch": dt_grid_search,
    "Decision Tree RandomSearch": dt_random_search,
    "Naive Bayes": nb,
}

for name, model in models.items():
    if "Naive" in name:
        y_pred = model.predict(X_test)
    else:
        y_pred = model.best_estimator_.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acerto do {name} : {accuracy:.4f}")
