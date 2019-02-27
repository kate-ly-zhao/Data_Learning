import tensorflow as tf
import os
import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from skimage.color import rgb2gray
import random

# Loading Belgian traffic data
def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "C:/Users/Downloads/PythonLearning/"
train_data_dir = os.path.join(ROOT_PATH, "TrafficSigns/Training")
test_data_dir = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

images, labels = load_data(train_data_dir)

# Conversion from list to array
images = np.array(images)
labels = np.array(labels)

# Traffic sign distribution with 62 bins
plt.hist(labels, 62)
plt.show()

# Viewing images based on random indices
traffic_signs = [300, 2250, 3650, 4000]

# Filling out subplot with defined random images
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)
    plt.show()    
plt.show() # Note: Images are different sizes

# Image rescaling
images28 = [transform.resize(image, (28, 28)) for image in images]

# Image conversion to array, then grayscale
images28 = np.array(images28)
images28 = rgb2gray(images28)

# Plotting grayscale images
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images28[traffic_signs[i]], cmap="gray")
    plt.subplots_adjust(wspace=0.5)
plt.show()


# ------- Deep Learning
# Placeholders
x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
y = tf.placeholder(dtype = tf.int32, shape = [None])

# Flatten input data - gives array of shape [None, 784] instead of [None, 28, 28] of grayscale images
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer - generates logits of size [None, 62]
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# Loss function definition
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))

# Optimizer definition
train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

# Conversion from logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Accuracy metric definition
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Running NN
tf.set_random_seed(1234)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(201):
    print('EPOCH', i)
    _, accuracy_val = sess.run([train_op, accuracy], feed_dict = {x: images28, y: labels})
    if i%10 == 0:
        print("Loss: ", loss)
    print('DONE WITH EPOCH')

# Evaluating NN with random images

sample_indexes = random.sample(range(len(images28)), 10)
sample_images = [images28[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

predicted = sess.run([correct_pred], feed_dict = {x: sample_images})[0]

print(sample_labels)
print(predicted)

fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2, 1+i)
    plt.axis('off')
    color = 'green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth: {0}\nPrediction: {1}".format(truth, prediction), fontsize = 12, color = color)
    plt.imshow(sample_images[i], cmap = "gray")
plt.show()

# Evaluating NN with test data
test_images, test_labels = load_data(test_data_dir)

test_images28 = [transform.resize(image, (28, 28)) for image in test_images]
test_images28 = rgb2gray(np.array(test_images28))

predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
accuracy = match_count/len(test_labels)
print("Accuracy: {:.3f}".format(accuracy))

sess.close()

# Data Source: https://btsd.ethz.ch/shareddata/
# Tutorial: https://www.datacamp.com/community/tutorials/tensorflow-tutorial
