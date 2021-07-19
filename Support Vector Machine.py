import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
digits = datasets.load_digits()
# digits dataset consists of 8x8 pixel images of hand written digits
#images attribute of the dataset stores 8x8 arrays of grayscale values for each image
# We will use these arrays to visualize the first 4 images
# target attribute of the dataset stores the digit each image represents (included in title of 4 plots below)
_,axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)


#flatten the images
#i.e. turn each 2D array of grayscale values from shape (8, 8) into shape (64,)
# entire dataset will be of shape (n_samples,n_features)
#n_samples: no. of images; n_features: total number of pixels in each image
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
#-1: value is inferred from length of array & remaining dimensions

# Create a classifier: a support vector classifier
#Also try with 'poly' and ‘sigmoid' kernels and different values of gamma
clf = svm.SVC(kernel='rbf', gamma=0.001)

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Prediction: {prediction}')

disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print( f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()