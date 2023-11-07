import numpy as np


class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1, epochs=100):
        self.weights = np.random.rand(n_inputs)
        self.bias = np.random.rand(1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def step_function(self, x):
        return 1 if x > 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return self.step_function(summation)

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights += self.learning_rate * (label - prediction) * inputs
                self.bias += self.learning_rate * (label - prediction)


def generate_data(n):
    data = np.random.randint(0, 2, (2**n, n))
    labels_and = np.all(data, axis=1)
    labels_or = np.any(data, axis=1)
    labels_xor = np.logical_xor(data[:, 0], data[:, 1])
    return data, labels_and, labels_or, labels_xor


def test_perceptron(n_inputs, logic_function):
    data, label_and, label_or, label_xor = generate_data(n_inputs)
    p = Perceptron(n_inputs)

    if logic_function == "AND":
        p.train(data, label_and)
        print(f"Teste para função AND com {n_inputs} entradas:")
        for i in range(len(data)):
            print(
                f"Entrada: {data[i]} Saída: {p.predict(data[i])} Esperado: {label_and[i]}"
            )
    elif logic_function == "OR":
        p.train(data, label_or)
        print(f"Teste para função OR com {n_inputs} entradas:")
        for i in range(len(data)):
            print(
                f"Entrada: {data[i]} Saída: {p.predict(data[i])} Esperado: {label_or[i]}"
            )
    elif logic_function == "XOR" and n_inputs >= 2:
        p.train(data, label_xor)
        print(f"Teste para função XOR com {n_inputs} entradas:")
        for i in range(len(data)):
            print(
                f"Entrada: {data[i]} Saída: {p.predict(data[i])} Esperado: {label_xor[i]}"
            )
    else:
        print(
            "Função lógica não reconhecida ou número de entradas insuficiente para XOR."
        )


test_perceptron(2, "AND")
test_perceptron(2, "OR")
test_perceptron(2, "XOR")
