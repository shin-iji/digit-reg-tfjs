import tensorflow as tf
from tensorflow import keras

(X_train, y_train),(X_test, y_test) = keras.datasets.mnist.load_data()

print(X_train.shape)
print(X_test.shape)
 
X_train = X_train.reshape([X_train.shape[0], 28, 28, 1])
X_test = X_test.reshape([X_test.shape[0], 28, 28, 1])
X_train = X_train/255.0
X_test = X_test/255.0
 
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

model = keras.Sequential([
    keras.layers.Conv2D(32, (5, 5), padding="same", input_shape=[28, 28, 1]),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Conv2D(64, (5, 5), padding="same"),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)
test_loss,test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

model.save("model.h5")