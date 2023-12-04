import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from io import StringIO

file_path = "./dados/ReutersGrain-train.csv"

# Carregando os dados do CSV
data = pd.read_csv(file_path, sep=";", encoding="utf-8")

# Separando os dados em features (X) e rótulos (y)
X = data["Text"]
y = data["class-att"]

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vetorizando os textos usando a abordagem de Bag of Words
vectorizer = CountVectorizer(analyzer="word")
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Treinando o modelo Naive Bayes Multinomial
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_bow, y_train)

# Fazendo previsões no conjunto de teste
y_pred = nb_classifier.predict(X_test_bow)

# Avaliando o desempenho do modelo
accuracy = metrics.accuracy_score(y_test, y_pred)
classification_report_result = metrics.classification_report(y_test, y_pred)

print(f"Acurácia do modelo: {accuracy}")
print(f"Relatório de Classificação:\n{classification_report_result}")

# Matriz de Confusão
confusion_matrix_result = pd.crosstab(
    y_test, y_pred, rownames=["Real"], colnames=["Predito"], margins=True
)
print(f"Matriz de Confusão:\n{confusion_matrix_result}")
