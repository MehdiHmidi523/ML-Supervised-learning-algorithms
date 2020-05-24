import numpy as np
import struct, random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import itertools
import scipy.io
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers


# define dataset locations
label_filename = 'A3_Dataset/train-labels-idx1-ubyte'
image_filename = 'A3_Dataset/train-images-idx3-ubyte'
# deal with label file
with open(label_filename, 'rb') as f:
    magic_label, n_labels = struct.unpack('>II', f.read(8))
    labels = np.fromfile(f, dtype=np.uint8)
f.close()
# deal with image file
with open(image_filename, 'rb') as f:
    magic_image, n_images, n_rows, n_cols = struct.unpack('>IIII', f.read(16))
    pic_data = np.fromfile(f, dtype=np.uint8)
f.close()

# reshape `images` to a n_images x n_rows x n_cols matrix and normalize all pixel to 255
pic_data = pic_data.reshape(n_images, n_rows, n_cols) / 255.
# define some variables
n_pixels_per_axis = n_rows  # number of pixels per dimension
n_samples = 16  # randomly select a small subset of data to visualize
n_samples_per_axis = int(np.sqrt(n_samples))
sampled_indices = np.array(random.choices(range(0, len(labels) - 1), k=n_samples))
sampled_images = pic_data[sampled_indices]
sampled_labels = labels[sampled_indices]
# set up a n_samples x n_samples plot
h = plt.figure(figsize=(7.5, 5.5))
gs = gridspec.GridSpec(n_samples_per_axis, n_samples_per_axis)  # divide plot into sub plots
gs.update(wspace=0.1, hspace=0.1)
#  each of x / y axis must have its number of bins = n_pixels_per_axis
edges = np.arange(0, n_pixels_per_axis + 1)
edges = edges[:-1] + 0.5 * (edges[1:] - edges[:-1])
#  loop through each selected image and plot
for index, image in enumerate(sampled_images):
    # location of the index-th subplot
    axis = h.add_subplot(gs[index])
    # inverse the content so it is not upside down
    content = image[::-1]
    # plot intensity
    axis.pcolormesh(edges, edges, content, vmin=0, vmax=1, cmap=plt.get_cmap('gray'))
    # hide all axes ticks
    axis.get_xaxis().set_ticks([])
    axis.get_yaxis().set_ticks([])
# save plot
plt.suptitle(str(n_samples) + ' samples of training data')
h.savefig('training_samples.png')
plt.close('all')


def model_learning_plot(model_history):
    plt.plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    plt.title('Learning Curve')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Function value')
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Purples):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_norm_grey.png')


data = scipy.io.loadmat('A3_DataSet/fashion_mnist.mat')

x_train = data['train_x']
y_train = data['train_y']

x_test = data['test_x']
y_test = data['test_y']

x_full = np.concatenate((x_train, x_test), axis=0)
y_full = np.concatenate((y_train, y_test), axis=0)

X = np.concatenate((x_full, y_full), axis=1)

np.random.shuffle(X)
sep = round(70000 * 2 / 3)

x_full = X[:, 0:784]
y_full = X[:, 784:]

x_train = x_full[0:sep]
y_train = y_full[0:sep]

x_test = x_full[sep:]
y_test = y_full[sep:]


mse_list = []
acc_list = []
time_list = []

# create model
model = Sequential()
model.add(Dense(100, input_dim=784, activation='relu'))  # 100 neurons in the first hidden layer for the 784 input params
model.add(Dense(10, activation='sigmoid'))
sgd = optimizers.SGD(lr=0.005, momentum=0.0, decay=0.0, nesterov=False)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mse', 'accuracy'])
model_info = model.fit(x_train, y_train, epochs=50, batch_size=1)

plt.figure()
model_learning_plot(model_info)

training_scores = model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], training_scores[1] * 100))
print("\n%s: %.2f%%" % (model.metrics_names[2], training_scores[2] * 100))

y_truth = y_test
y_pred = model.predict(x_test)
y_tr = [np.argmax(t) for t in y_truth]
y_pr = [np.argmax(t) for t in y_pred]

cm = confusion_matrix(y_tr, y_pr)
target_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(classification_report(y_tr, y_pr, target_names=target_names))
plot_confusion_matrix(cm, target_names, normalize=True, title='Confusion matrix')
