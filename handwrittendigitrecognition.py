import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib

# Set the matplotlib backend to TkAgg to enable interactive plotting
matplotlib.use('TkAgg') 

# Load the MNIST dataset from TensorFlow Keras
mnist = tf.keras.datasets.mnist 

# Split the dataset into training and testing sets
(x_train, y_train), (x_test, y_test) = mnist.load_data() 

# Normalize the pixel values to be between 0 and 1 for both training and testing sets
x_train = tf.keras.utils.normalize(x_train, axis=1)  
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Create a Sequential neural network model
model = tf.keras.models.Sequential()

# Flatten the input data from 28x28 pixels to a 1D array
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# The input layer flattens the 2D image (28x28 pixels) into a 1D vector of 784 pixels.
# This transformation is necessary because the dense layers that follow expect 1D input.

# Add a fully connected layer with 128 neurons and ReLU activation
model.add(tf.keras.layers.Dense(128, activation='relu'))
# The first hidden layer consists of 128 neurons, which allows the network to learn and capture complex patterns.
# ReLU (Rectified Linear Unit) activation function introduces non-linearity by outputting the input directly if positive, otherwise zero.
# This non-linearity helps the network learn complex functions and patterns.

# Add another fully connected layer with 128 neurons and ReLU activation
model.add(tf.keras.layers.Dense(128, activation='relu'))
# The second hidden layer also has 128 neurons, providing additional capacity for learning complex features.
# Using ReLU again helps in maintaining non-linearity, which is crucial for the network's ability to learn from data.

# Add the output layer with 10 neurons (one for each digit) and softmax activation
model.add(tf.keras.layers.Dense(10, activation='softmax'))
# The output layer has 10 neurons, each corresponding to a digit (0-9).
# Softmax activation function converts the output into a probability distribution over the 10 classes.
# Each output neuron represents the probability of the input image belonging to a particular digit class.

# Compile the model with the Adam optimizer, sparse categorical crossentropy loss function, and accuracy metric
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Adam optimizer is used for training the model, which adjusts weights efficiently during training.
# Sparse categorical crossentropy is the loss function used for multi-class classification problems.
# It measures the difference between the predicted probabilities and the true class labels.
# Accuracy metric is used to evaluate the performance of the model during training and testing.

# Train the model with the training data for 3 epochs
model.fit(x_train, y_train, epochs=3)
# Training the model involves adjusting the weights based on the training data over a specified number of epochs (iterations through the dataset).
# Three epochs is a starting point, and you may adjust based on performance.

# Save the trained model to a file named 'handwritten.keras'
model.save('handwritten.keras')
# Saving the model allows you to reuse it later without retraining, which is computationally expensive.

# Load the model from the saved file to avoid retraining
model = tf.keras.models.load_model('handwritten.keras')
# Loading the saved model is essential for evaluating and making predictions without needing to retrain.

# Evaluate the model performance on the test data and print the loss and accuracy
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')
# Evaluating the model measures how well it generalizes to new data (test data) and provides insights into its performance.

# Initialize the image number for testing
image_number = 1

# Loop through image files in the 'digits' directory
while os.path.isfile(f'digits/digit{image_number}.png'):
    try: 
        # Read the image file and convert it to grayscale
        img = cv2.imread(f'digits/digit{image_number}.png')[:,:,0]
        
        # Invert the pixel values (MNIST images are white on black background)
        img = np.invert(np.array([img]))
        
        # Predict the digit in the image using the trained model
        prediction = model.predict(img)
        
        # Print the predicted digit
        print(f"This digit is probably a {np.argmax(prediction)}")
        
        # Display the image using matplotlib
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        
    except Exception as e:
        # Print an error message if an exception occurs
        print(f"Error: {e}")
        
    finally:
        # Increment the image number for the next iteration
        image_number += 1
