import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from kneed import DataGenerator, KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

base = pd.read_csv("dados/Iris.csv", delimiter=",")
Entrada = base.iloc[:, 0:4].values
scaler = MinMaxScaler()
Entrada = scaler.fit_transform(Entrada)

limit = int((Entrada.shape[0] // 2) ** 0.5)
for k in range(2, limit + 1):
    model = KMeans(n_clusters=k)
    model.fit(Entrada)
    pred = model.predict(Entrada)
    silhouette = silhouette_score(Entrada, pred)
    db_score = davies_bouldin_score(Entrada, pred)
    print("Silhouette Score k = {}: {:<.3f}".format(k, silhouette))
    print("Davies-Bouldin Score k = {}: {:<.3f}".format(k, db_score))

wcss = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=10)
    kmeans.fit(Entrada)
    wcss.append(kmeans.inertia_)

plt.style.use("fivethirtyeight")
plt.plot(range(2, 11), wcss)
plt.xticks(range(2, 11))
plt.title("The elbow method")
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

kl = KneeLocator(range(2, 11), wcss, curve="convex", direction="decreasing")
kl.elbow

kmeans = KMeans(n_clusters=3, random_state=0)
saida_kmeans = kmeans.fit_predict(Entrada)

plt.scatter(
    Entrada[saida_kmeans == 0, 0],
    Entrada[saida_kmeans == 0, 1],
    s=100,
    c="purple",
    label="Iris-setosa",
)
plt.scatter(
    Entrada[saida_kmeans == 1, 0],
    Entrada[saida_kmeans == 1, 1],
    s=100,
    c="orange",
    label="Iris-versicolour",
)
plt.scatter(
    Entrada[saida_kmeans == 2, 0],
    Entrada[saida_kmeans == 2, 1],
    s=100,
    c="green",
    label="Iris-virginica",
)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=100,
    c="red",
    label="Centroids",
)
plt.legend()
