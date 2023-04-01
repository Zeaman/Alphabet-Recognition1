"""
Install all dependency libraries
pip install python-mnist
pip install numpy
pip install scikit-learn
pip install tensorflow
"""
# import the MNIST dataset
import numpy
from mnist import MNIST
from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = MNIST(path='D:/AI/alphabet_recognition/dataset', return_type='numpy')
dataset.select_emnist('letters')
X, y = dataset.load_training()
print(X.shape)
print(y.shape)
# Reshape the dataset to 28x28 for X and 1 for y
X = X.reshape(X.shape[0], 28, 28)
y = y.reshape(y.shape[0], 1)
y = y - 1

# Prepare the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Map the dataset of 0-255 to 0-1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convert the integer to one hot vector(binary class matrix)
y_train = np_utils.to_categorical(y_train, num_classes=26)  # 26 classes since we have 26 letters
y_test = np_utils.to_categorical(y_test, num_classes=26)  # 26 classes since we have 26 letters

# Define the CNN model
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  # change the 2D input(28,28) array to 1D (28x28 =784) input array
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))  # For not overfitting when train
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))  # For not overfitting when train
model.add(Dense(26, activation='softmax'))  # For 26 letters, the output layer must have 26 neuron
model.summary()  # Print the summary of the model

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Calculate the accuracy of the model before training
score = model.evaluate(X_test, y_test, verbose=0)
accuracy_before = 100*score[1]
print("The test accuracy before training is:", accuracy_before)

# Save the model after training using callback
check_pointer = ModelCheckpoint(filepath='D:/AI/alphabet_recognition/best_model.h5', verbose=1, save_best_only=True)

# Train the model using fit
model.fit(X_train, y_train, batch_size=128, epochs=15, validation_split=0.2, callbacks=[check_pointer], verbose=1, shuffle=True)

# Load the model (best_model.h5)
loaded_model = model.load_model('D:/AI/alphabet_recognition/best_model.h5')

# Calculate the accuracy of the model after training
score = model.evaluate(X_test, y_test, verbose=0)
accuracy_after = 100*score[1]
print("The test accuracy after training is:", accuracy_after)

