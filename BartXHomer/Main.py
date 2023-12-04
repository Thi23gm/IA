from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing import image
import numpy as np
from IPython.display import Image
from PIL import Image as PILImage

# Inicializando a Rede Neural Convolucional
classifier = Sequential()

# Passo 1 - Primeira Camada de Convolução
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))

# Passo 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adicionando a Segunda Camada de Convolução
classifier.add(Conv2D(32, (3, 3), activation="relu"))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Passo 3 - Flattening
classifier.add(Flatten())

# Passo 4 - Full connection
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dropout(0.5))  # Adicionando Dropout para evitar overfitting
classifier.add(
    Dense(units=2, activation="softmax")
)  # Mudando para 2 unidades para as duas classes

# Compilando a rede
classifier.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

# Criando os objetos train_datagen e validation_datagen com as regras de pré-processamento das imagens
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Pré-processamento das imagens de treino e validação
training_set = train_datagen.flow_from_directory(
    "./archive/training_set",
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical",
)

validation_set = validation_datagen.flow_from_directory(
    "./archive/test_set", target_size=(64, 64), batch_size=32, class_mode="categorical"
)

# Executando o treinamento
classifier.fit_generator(
    training_set,
    steps_per_epoch=8000,
    epochs=5,
    validation_data=validation_set,
    validation_steps=2000,
)

# Testando com as imagens


# Função para exibir imagem
def display_image(image_path):
    img = PILImage.open(image_path)
    img.show()


# Primeira Imagem do Bart
test_image = image.load_img("./archive/test_set/bart/bart1.bmp", target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)

if result[0][0] == 1:
    prediction = "Homer"
else:
    prediction = "Bart"

print(f"Previsão para bart1.bmp: {prediction}")
display_image("./archive/test_set/bart/bart1.bmp")

# Segunda Imagem do Bart
test_image = image.load_img("./archive/test_set/bart/bart2.bmp", target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)

if result[0][0] == 1:
    prediction = "Homer"
else:
    prediction = "Bart"

print(f"Previsão para bart2.bmp: {prediction}")
display_image("./archive/test_set/bart/bart2.bmp")

# Primeira Imagem do Homer
test_image = image.load_img("./archive/test_set/homer/homer1.bmp", target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)

if result[0][0] == 1:
    prediction = "Homer"
else:
    prediction = "Bart"

print(f"Previsão para homer1.bmp: {prediction}")
display_image("./archive/test_set/homer/homer1.bmp")

# Segunda Imagem do Homer
test_image = image.load_img("./archive/test_set/homer/homer2.bmp", target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)

if result[0][0] == 1:
    prediction = "Homer"
else:
    prediction = "Bart"

print(f"Previsão para homer2.bmp: {prediction}")
display_image("./archive/test_set/homer/homer2.bmp")
