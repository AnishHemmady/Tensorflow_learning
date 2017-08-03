import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data

'''initial steps'''
#tensorflow loading mnist data
#loading data as one hot vector
'''one hot vector-it means data labels (classes) is converted into columns representing binary representation of labels.same number
of rows but but number of columns is equal to number of classes.(0 or 1 representation)'''

data=input_data.read_data_sets("data/MNIST/", one_hot=True)

print("size od data loaded:")
print("Training set size:\t {}".format(len(data.train.labels)))
print("Test set size:\t {}".format(len(data.test.labels)))
print("Validation set size:\t {}".format(len(data.validation.labels)))
#index in one hot vector represents actual class whether its number 1,2,...8,9.
data.test.cls=np.array([label.argmax() for label in data.test.labels])

#default mnist image size is 28 by 28:
img_size=28

#storing images in one dimensional array like a onr dimensional vector represntation.length of one dim array[28*28]
img_flat=img_size*img_size

#shape of image with heigth,width
img_shape=(img_size,img_size)

#number of classses
no_of_classes=10

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
		
# Get the first images from the test-set.
images = data.test.images[0:9]

# Get the true classes for those images.
cls_true = data.test.cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)

#none means it will accept any number of input images with each vector representing size of 28 by 28.
x=tf.placeholder(tf.float32,[None,img_flat])


y_true=tf.placeholder(tf.int64,[None,no_of_classes])

y_true_cls = tf.placeholder(tf.int64, [None])

#2-D tensor representing weights
weights=tf.Variable(tf.zeros([img_flat,no_of_classes]))

#1-D tensor representing biases
biases=tf.Variable(tf.zeros([no_of_classes]))

#now trying to build a  model
#the element of the ith row and jth column is an estimate of how likely the ith input image is to be of the jth class.
logits = tf.matmul(x, weights) + biases

y_pred = tf.nn.softmax(logits)

#predicting the label
y_pred_cls = tf.argmax(y_pred, dimension=1)

#cost function calculation.i.e. comparing true labels with predicted labels.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_true)


#in earlier step cross_entropy was calculated for each image.Now use all cross_entropy values and calculate its means to know
#how our model well our model is evaluating.
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#########################################################################################

'''Here tensorflow graph generation code ends '''

'''Actual working starts from below'''
#########################################################################################
session = tf.Session()
#this below line intializes the tensorflow weights and biases.You need to initialize them otherwise error would be caused.
session.run(tf.global_variables_initializer())
batch_size=1000

#actual working of code of above graph created starts from here
def optimize(number_of_iterations):
	for i in range(number_of_iterations):
		x_batch, y_true_batch = data.train.next_batch(batch_size)
		feed_dict_train = {x: x_batch,y_true: y_true_batch}
		session.run(optimizer, feed_dict=feed_dict_train)
		
feed_dict_test = {x: data.test.images,y_true: data.test.labels,y_true_cls: data.test.cls}
		
def print_accuracy():
    # Use TensorFlow to compute the accuracy.
	acc = session.run(accuracy, feed_dict=feed_dict_test)
    
    # Print the accuracy.
	print("Accuracy on test-set: {0:.1%}".format(acc))
	
	
def plot_example_errors():
	# Use TensorFlow to get a list of boolean values
	# whether each test-image has been correctly classified,
	# and a list for the predicted class of each image.
	correct, cls_pred = session.run([correct_prediction, y_pred_cls],
									feed_dict=feed_dict_test)

	# Negate the boolean array.
	incorrect = (correct == False)

	# Get the images from the test-set that have been
	# incorrectly classified.
	images = data.test.images[incorrect]

	# Get the predicted classes for those images.
	cls_pred = cls_pred[incorrect]

	# Get the true classes for those images.
	cls_true = data.test.cls[incorrect]

	# Plot the first 9 images.
	plot_images(images=images[0:9],cls_true=cls_true[0:9],cls_pred=cls_pred[0:9])
	
	
	
	
def plot_weights():
	# Get the values for the weights from the TensorFlow variable.
	w = session.run(weights)

	# Get the lowest and highest values for the weights.
	# This is used to correct the colour intensity across
	# the images so they can be compared with each other.
	w_min = np.min(w)
	w_max = np.max(w)

	# Create figure with 3x4 sub-plots,
	# where the last 2 sub-plots are unused.
	fig, axes = plt.subplots(3, 4)
	fig.subplots_adjust(hspace=0.3, wspace=0.3)

	for i, ax in enumerate(axes.flat):
		# Only use the weights for the first 10 sub-plots.
		if i<10:
			# Get the weights for the i'th digit and reshape it.
			# Note that w.shape == (img_size_flat, 10)
			image = w[:, i].reshape(img_shape)

			# Set the label for the sub-plot.
			ax.set_xlabel("Weights: {0}".format(i))

			# Plot the image.
			ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

		# Remove ticks from each sub-plot.
		ax.set_xticks([])
		ax.set_yticks([])
	plt.show()
	
optimize(30)
print_accuracy()

plot_example_errors()