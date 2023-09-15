from chefboost import Chefboost as chef
import pandas as pd

df = pd.read_csv("../Dados/Credito.csv")

configs = [{"algorithm": "CART"}, {"algorithm": "ID3"}, {"algorithm": "C4.5"}]

for config in configs:
    print(f"Treinando modelo usando algoritmo {config['algorithm']}...")
    model = chef.fit(df.copy(), config=config)
    print(f"Árvore gerada usando {config['algorithm']}:")
    chef.visualize(model, view=True)
