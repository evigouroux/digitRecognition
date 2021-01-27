import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(trainingImages, trainingLabels),(testImages, testLabels) = mnist.load_data()
 
classes = 10
imageHeight = 28
imageWidth = 28
epochNumber = 5

# Display the caracteristics of our trainings image
#   
#print(trainingImages.shape)
#print(trainingImages[0])

# Normalise the data so that our images become black and white (detail this in the report)
trainingImages = trainingImages / 255.0
testImages = testImages / 255.0

# layer creation, the input size is 28x28 : the size of our images. The output size is 10 : our number of classes
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(imageWidth,imageHeight)), 
                                    tf.keras.layers.Dense(128, activation='relu'), 
                                    tf.keras.layers.Dense(classes, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(trainingImages, trainingLabels, epochs=epochNumber)


# Show the accuracy of the model
#
#print(model.evaluate(testImages,testLabels))

verificationIndex = 5

plt.imshow(testImages[verificationIndex])
prediction = model.predict(testImages)
print(np.argmax(prediction[verificationIndex]))