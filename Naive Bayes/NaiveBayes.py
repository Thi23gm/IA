import pandas as pd
import io
from collections import defaultdict

# Lendo o conjunto de dados
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


def naive_bayes_classifier(df, test_data):
    target = "jogar"
    classes = df[target].unique()
    result_probabilities = defaultdict(float)

    # Calcular a probabilidade total de cada classe
    total_records = len(df)
    for c in classes:
        class_prob = len(df[df[target] == c]) / total_records

        # Calcular a probabilidade condicional para cada atributo no registro de teste
        conditional_prob = 1.0
        for attr, value in test_data.items():
            subset = df[df[target] == c]
            conditional_prob *= len(subset[subset[attr] == value]) / len(subset)

        result_probabilities[c] = class_prob * conditional_prob

    # Normalizando as probabilidades para que somem 1
    sum_probs = sum(result_probabilities.values())
    normalized_probs = {
        k: (v / sum_probs) * 100 for k, v in result_probabilities.items()
    }

    # Retorna a classe com a maior probabilidade e as probabilidades normalizadas
    return max(normalized_probs, key=normalized_probs.get), normalized_probs


# Testar o classificador
test_data = {
    "aparencia": "Chuva",
    "temperatura": "Fria",
    "umidade": "Normal",
    "ventando": "sim",
}

prediction, probabilities = naive_bayes_classifier(df, test_data)

print(f"Previsão: {prediction}")
for class_name, prob in probabilities.items():
    print(f"Probabilidade da classe {class_name}: {prob:.2f}%")
