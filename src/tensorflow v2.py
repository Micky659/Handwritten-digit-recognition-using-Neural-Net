from PIL import Image
import numpy as np
import tensorflow as tf

# Importing the dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

validation_images = train_images[55000:, :]
train_images = train_images[:55000, :]
validation_labels = train_labels[55000:]
train_labels = train_labels[:55000]

# Defining the neural net layers
hidden_layer1 = 512  # 1st hidden layer
hidden_layer2 = 256  # 2nd hidden layer
hidden_layer3 = 128  # 3rd hidden layer
output_layer = 10  # output layer (0 - 9 digits)

# Initializing hyper parameters
num_epochs = 10

# Setting up layers of network
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(hidden_layer1, activation='relu'),
    tf.keras.layers.Dense(hidden_layer2, activation='relu'),
    tf.keras.layers.Dense(hidden_layer3, activation='relu'),
    tf.keras.layers.Dense(output_layer, activation='softmax')
])

# Initializing neural net
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_images, y=train_labels, epochs=num_epochs)

# Evaluating loss and accuracy
loss, accuracy = model.evaluate(test_images, test_labels)
print('Loss in test set: {0:.4f} \tAccuracy: {1:.2f}%'.format(loss, accuracy*100))

model.save('preprocessed_model/mnist.h5')

# Test on your image
img = np.invert(Image.open("Stock/test_img.png").convert('L')).ravel()
prediction = model.predict(img.reshape(1, 28, 28))
digit = np.argmax(prediction)
confidence = np.max(prediction) * 100

print('Prediction: Input image have digit {0} with {1:.2f}% confidence'.format(digit, confidence))
