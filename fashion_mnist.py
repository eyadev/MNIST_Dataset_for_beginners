#importing tensorflow,keras,tensorflow_datasets
import tensorflow as tf
import tensorflow_datasets as tfds 
import tensorflow.keras as keras 

#Helpers libraries
import math
import matplotlib as plt
import numpy as np 

#To log errors 
import logging 
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#First we have to import MNIST dataset
#in the /*dataset*/ we have our fashion_mnist data 
#in the /*metadata*/ we have labels , dataset name, version etc.. like additional informations we get them by setting with_info param to true 
dataset,metadata = tfds.load('fashion_mnist',as_supervised=True,with_info=True)

#as we know now we have our data in dataset variable , fashion_mnist is composed of /*70 000 clothes images in 10 categories*/
#our dataset is divised 60 000 data for training and 10 000 for testing and we do that to avoid overfitting
#now let's put them in 2 different variabels
train_dataset, test_dataset= dataset['train'],dataset['test']

#each image belong to a category in other way each image is mapped to a single label and these infos we have them in the metadata varriable
#and not included with the dataset so let's store them so we can use them later  and print them to make sure 

class_names=metadata.features['label'].names
print("*******Categories names*******:{}".format(class_names))

#now let's explore out data 
#let's see number of training dataset and number of testing dataset we can have this info from our metadata variable

num_train_examples= metadata.splits['train'].num_examples
num_test_examples= metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of testing examples : {}".format(num_test_examples))

#it's time to being preprcessing the data
#we all now that a pixel can have a range value it depends on RGB or Grey scale image.
#as we are on a grey scale image so value of each pixel in a n integer in the range [0,255] , Typically zero is taken to be black, and 255 is taken to be white.
#for our model to work propery we need to contract th range to [0,1], it's called 'normalisation', Normalizing the data generally speeds up learning and leads to faster convergence
#so now let's create our normalisation function so we can apply it to each of our images 

def normalize(images, labels):
  images=tf.cast(images,tf.float32)
  images/=255
  return images, labels 

#now we need to apply this normalize function to the train and test datasets using map function
train_dataset= train_dataset.map(normalize)
test_dataset= test_dataset.map(normalize)

#now we can do a trick to make our treaning faster 
#so the first time you use your dataset it' s gonna be loaded from your disk so now you can keep them in memery means we can Cache them 

train_dataset= train_dataset.cache()
test_dataset = test_dataset.cache()

############BUILD THE MODEL 

#let's no begin to build our model
#first thing first is to setup our layers
#we gonna use a simple layer that called the DENSE layers
#the Dense layer feeds all outputs from the previous layer to all its neurons, each neuron providing one output to the next layer so it's fully conneted
#as parameter for our layers we have to give what's called an activation function to help the network learn complex patterns in the data
#the activation function is at the end deciding what is to be fired to the next neuron 
#we have many activation function , we gonna use relu in the hidden layer and softmax in the last layer.
#Softmax is a mathematical function that converts a vector of numbers into a vector of probabilities, where the probabilities of each value are proportional to the relative scale of each value in the vector.
#so for  example in the end we have a probality 0.8 that is a T-shirt and 0.3 that is a shoes that means that our image has a probability of 80%  that can be a T-shirt and 30% that can be a shoe , so obviously it's a T-shirt

#NB: we have to use flatten in the first layer means like reashape it to 1D means this layer transforms the images from a 2d-array of 28  ×  28 pixels, to a 1d-array of 784 pixels (28*28). 
#after defining all our layers we have to assemle them in a model for this example we gonna use the sequential model 
#A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

#so now we have a model

#now that we finished from seting up our layers we can compile our model
#to compile our model we need to define some params:
#1-Loss function:it's gonna help us to quantify our errors other words, it's gonna tell us how far our model is far from the desired output, (this function has to be minimized)
#2-Optimizer:as we said we have to minimise the loss function so the optimzer is an algorithm for adjusting the inner parameters of the model in order to minimize loss ex: ADAM optimizer
#3-Metrics:Used to monitor the training and testing steps.

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

#after compiling our model now it's time to train this model
# so we need to define some few parameters 
#The dataset.shuffle randomizes the order so our model cannot learn anything from the order of the examples
#dataset.batch(32) tells model.fit to use batches of 32 images and labels when updating the model variables.

BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

#so now the main method to train our model is fit()
#The epochs=5 parameter limits training to 5 full iterations of the training dataset, so a total of 5 * 60000 = 300000 examples.
model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

#now we need to evaluate our dataset means compare how the model performs on the test dataset

test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))
print('Accuracy on test dataset:', test_accuracy)

#So now that we have our model trained we can use it to make some predictions on some images
for test_images, test_labels in test_dataset.take(1):
  test_images = test_images.numpy()
  test_labels = test_labels.numpy()
  predictions = model.predict(test_images)

predictions.shape
predictions[0]
np.argmax(predictions[0])
