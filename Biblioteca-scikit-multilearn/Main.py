import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Carregando os dados do CSV
file_path = "./dados/multilabel_dataset.csv"
data = pd.read_csv(file_path, encoding="utf-8")

# Separando os dados em features (X) e rótulos (y)
X = data["Text"]
y = data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vetorizando os textos usando a abordagem TF-IDF
vectorizer = TfidfVectorizer(analyzer="word", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Criando o classificador Binary Relevance com Naive Bayes
classifier = BinaryRelevance(classifier=MultinomialNB(), require_dense=[True, True])
classifier.fit(X_train_tfidf, y_train)

# Fazendo previsões no conjunto de teste
y_pred = classifier.predict(X_test_tfidf)

# Avaliando o desempenho do modelo
accuracy = metrics.accuracy_score(y_test, y_pred)
classification_report_result = metrics.classification_report(y_test, y_pred)

print(f"Acurácia do modelo: {accuracy}")
print(f"Relatório de Classificação:\n{classification_report_result}")
