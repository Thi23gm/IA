from chefboost import Chefboost as chef
import pandas as pd

# Carregar os dados
df = pd.read_csv("../Dados/Credito.csv")

# Configurações para cada algoritmo
configs = [
    {'algorithm': 'CART'},
    {'algorithm': 'ID3'},
    {'algorithm': 'C4.5'}
]

# Treinar e exibir as árvores para cada algoritmo
for config in configs:
    print(f"Treinando modelo usando algoritmo {config['algorithm']}...")
    model = chef.fit(df.copy(), config=config)
    print(f"Árvore gerada usando {config['algorithm']}:")
    chef.visualize(model, view=True)  # Isso irá gerar e exibir um arquivo .dot que representa a árvore
