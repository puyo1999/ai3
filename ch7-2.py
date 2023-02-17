from tensorflow import keras

model = keras.Sequential()
model.add()

sgd = keras.optimizers.SGD(learning_rate=0.1)
sgd_nest = keras.optimizers.SGD(momentum=0.9, nesterov=True)

adagrad = keras.optimizers.Adagrad()
model.compile(optimizer=adagrad, loss='sparse_categorical_crossentropy')

model.add(keras.layers.Flatten(input_shape=(28,28)))





