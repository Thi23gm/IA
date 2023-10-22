# Ainda tem q fazer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Defina as cores para as classes verdadeiras
colors_true = np.array(["purple", "orange", "green"])

# Carregue seus dados
base = pd.read_csv("dados/Iris.csv", delimiter=",")
Entrada = base.iloc[:, 0:4].values
scaler = MinMaxScaler()
Entrada = scaler.fit_transform(Entrada)

# Suponhamos que você já tenha definido as variáveis colors_kmeans e saida_kmeans
colors_kmeans = np.array(
    ["red", "blue", "yellow"]
)  # Defina as cores dos grupos do K-Means

# Defina o número de clusters para o K-Means (no seu exemplo, 3 clusters)
num_clusters = 3

# Ajuste o modelo K-Means
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
saida_kmeans = kmeans.fit_predict(Entrada)

# Mapeie as classes verdadeiras para índices inteiros
class_mapping = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
class_indices = [class_mapping[class_name] for class_name in base["class"]]

# Gráfico 1: Separação de classes verdadeiras
plt.subplot(1, 2, 1)
plt.scatter(Entrada[:, 0], Entrada[:, 1], c=colors_true[class_indices], s=50)
plt.title("Classes Verdadeiras")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")

# Gráfico 2: Separação de grupos pelo K-Means
plt.subplot(1, 2, 2)
plt.scatter(Entrada[:, 0], Entrada[:, 1], c=colors_kmeans[saida_kmeans], s=50)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=100,
    c="black",
    label="Centroids",
)
plt.title("Agrupamento pelo K-Means")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")

# Identifique as instâncias agrupadas incorretamente
incorrectly_clustered = np.array(class_indices) != saida_kmeans
incorrect_points = Entrada[incorrectly_clustered]

# Plote as instâncias agrupadas incorretamente
plt.scatter(
    incorrect_points[:, 0],
    incorrect_points[:, 1],
    s=100,
    c="gray",
    marker="x",
    label="Incorretamente Agrupado",
)
plt.legend()

plt.show()
