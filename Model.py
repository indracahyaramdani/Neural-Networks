# MNIST dataset(train + test)
def create_data_mnist(path):

    # Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test =load_mnist_dataset('test',path)

    # And return all the data
    return X, y , X_test, y_test

fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

#read an image
image_data = cv2.imread('pants.png', cv2.IMREAD_GRAYSCALE)

#resize to the same size as Fashion MNIST images
image_data = cv2.resize(image_data(28, 28))

# invert image colors
image_data = 255- image_data

# reshape and scale pixel data
image_data = (image_data.reshape(1, -1).astype(np.float32)-127.5)/127.5

# load the model
model = Model.load('fashion_mnist.model')

# predict on the image
confidences = model.predict(image_data)

# get prediction instead of confidence levels
predictions = model.output_layer_activation.predictions(confidences)

# get label name from label index
prediction = fashion_mnist_labels[predictions[0]]

print(predictions)