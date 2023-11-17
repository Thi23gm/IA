import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        learning_rate,
        activation_function,
        activation_derivative,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative

        # Inicialização dos pesos
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)

        # Inicialização dos bias
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def feedforward(self, inputs):
        # Camada oculta
        hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.activation_function(hidden_input)

        # Camada de saída
        output_input = (
            np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        )
        output = self.activation_function(output_input)

        return hidden_output, output

    def backward(self, inputs, target, hidden_output, output):
        # Cálculo do erro
        output_error = target - output

        # Gradiente na camada de saída
        output_delta = output_error * self.activation_derivative(output)

        # Atualização dos pesos e bias na camada de saída
        self.weights_hidden_output += (
            np.dot(hidden_output.T, output_delta) * self.learning_rate
        )
        self.bias_output += (
            np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        )

        # Propagação do erro para a camada oculta
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)

        # Gradiente na camada oculta
        hidden_delta = hidden_error * self.activation_derivative(hidden_output)

        # Atualização dos pesos e bias na camada oculta
        self.weights_input_hidden += np.dot(inputs.T, hidden_delta) * self.learning_rate
        self.bias_hidden += (
            np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate
        )

    def train(self, inputs, targets, epochs, convergence_threshold=0.01):
        for epoch in range(epochs):
            total_error = 0  # Track total error for convergence check
            for i in range(len(inputs)):
                input_sample = inputs[i].reshape(1, -1)
                target_sample = targets[i].reshape(1, -1)

                hidden_output, output = self.feedforward(input_sample)
                self.backward(input_sample, target_sample, hidden_output, output)

                # Update total_error for convergence check
                total_error += np.sum(np.square(target_sample - output))

            # Calculate average error for this epoch
            average_error = total_error / len(inputs)

            # Print or plot the average error for monitoring
            print(f"Epoch {epoch + 1}, Average Error: {average_error}")

            # Check for convergence based on the average error
            if average_error < convergence_threshold:
                print(f"Converged at epoch {epoch + 1}")
                break  # Exit the training loop if converged

    def predict(self, inputs):
        _, output = self.feedforward(inputs)
        return output


# Funções de ativação
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


# Função para treinar e testar a RNA
def experiment(
    input_size,
    logic_function,
    learning_rate,
    activation_function,
    activation_derivative,
    epochs,
):
    # Geração de dados de treinamento e teste para a função lógica selecionada
    inputs_train = np.random.randint(2, size=(1000, input_size))
    targets_train = logic_function(
        inputs_train, axis=1, keepdims=True
    )  # Specify the axis and keepdims

    inputs_test = np.random.randint(2, size=(100, input_size))
    targets_test = logic_function(
        inputs_test, axis=1, keepdims=True
    )  # Specify the axis and keepdims

    # Criação e treinamento da RNA
    nn = NeuralNetwork(
        input_size, 5, 1, learning_rate, activation_function, activation_derivative
    )
    nn.train(inputs_train, targets_train, epochs)

    # Teste da RNA
    correct_predictions = 0
    predictions = []

    for i in range(len(inputs_test)):
        # Adicionamos uma dimensão extra para garantir que os arrays tenham a forma correta
        input_sample = inputs_test[i].reshape(1, -1)
        hidden_output, output = nn.feedforward(input_sample)
        prediction = np.round(output)
        predictions.append(prediction)
        if np.array_equal(prediction, targets_test[i]):
            correct_predictions += 1

    accuracy = correct_predictions / len(inputs_test)
    print(
        f"Accuracy for {logic_function.__name__} with {input_size} inputs: {accuracy}"
    )

    return accuracy, predictions


# Experimentos
input_sizes = [2, 5, 10]
learning_rates = [0.1, 0.01, 0.001]
activation_functions = [sigmoid, relu]
activation_derivatives = [sigmoid_derivative, relu_derivative]

# Armazenamento dos resultados para plotagem
results = {
    "input_size": [],
    "learning_rate": [],
    "activation_function": [],
    "accuracy": [],
}

for input_size in input_sizes:
    for learning_rate in learning_rates:
        for activation_function, activation_derivative in zip(
            activation_functions, activation_derivatives
        ):
            print(
                f"\nExperiment for {input_size} inputs, learning rate {learning_rate}, and activation function {activation_function.__name__}:"
            )
            accuracy, _ = experiment(
                input_size,
                np.sum,
                learning_rate,
                activation_function,
                activation_derivative,
                epochs=100,
            )

            # Armazenar resultados para plotagem
            results["input_size"].append(input_size)
            results["learning_rate"].append(learning_rate)
            results["activation_function"].append(activation_function.__name__)
            results["accuracy"].append(accuracy)

# Plotagem dos resultados
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
fig.suptitle("Neural Network Performance Analysis")

# Gráfico 1: Precisão em relação à taxa de aprendizado
axes[0].set_title("Accuracy vs. Learning Rate")
for input_size in input_sizes:
    for activation_function in activation_functions:
        subset = [
            (results["input_size"][i], results["accuracy"][i])
            for i in range(len(results["input_size"]))
            if results["activation_function"][i] == activation_function.__name__
        ]
        axes[0].plot(
            [subset[i][0] for i in range(len(subset))],
            [subset[i][1] for i in range(len(subset))],
            label=f"{activation_function.__name__}",
        )

axes[0].set_xlabel("Input Size")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

# Gráfico 2: Precisão em relação à função de ativação
axes[1].set_title("Accuracy vs. Activation Function")
for input_size in input_sizes:
    for learning_rate in learning_rates:
        subset = [
            (results["activation_function"][i], results["accuracy"][i])
            for i in range(len(results["activation_function"]))
            if results["learning_rate"][i] == learning_rate
        ]
        axes[1].plot(
            [subset[i][0] for i in range(len(subset))],
            [subset[i][1] for i in range(len(subset))],
            label=f"LR={learning_rate}",
        )

axes[1].set_xlabel("Activation Function")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.show()
