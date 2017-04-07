import tflearn
from tflearn.layers.recurrent import lstm 
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from PIL import Image
import scipy
import numpy

image = "./threernn.png"

def showImg(image):
    im = Image.open(image)
    im.thumbnail((28, 28), Image.ANTIALIAS)
    imgplot = plt.imshow(im)
    plt.show()

def preprocessImg(image):
    img = scipy.misc.imread(image, flatten=True, mode='L')
    img.resize((28, 28))
    img = img.reshape([-1, 28, 28])
    imgopt = []
    for z in img[0]:
        imgopt.append(["{0:.8f}".format(round(1-i/255,8)) for i in z])
    imgopt = np.array(imgopt)
    imgopt = imgopt.reshape([-1, 28, 28])
    return(imgopt)

X, Y, test_x, test_y = mnist.load_data(one_hot=True)
X = X.reshape([-1, 28, 28])
test_x = test_x.reshape([-1, 28, 28])

convnet = input_data(shape=[None, 28, 28])

convnet = tflearn.lstm(convnet, 64, return_seq=True)

convnet = tflearn.lstm(convnet, 128)

convnet = fully_connected(convnet, 512, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)


##model.fit({'input': X}, {'targets': Y}, n_epoch=8, validation_set=({'input': test_x}, {'targets': test_y}), 
##    snapshot_step=500, show_metric=True, run_id='mnist')
##
##model.save('mnistrnn.model')

if __name__ == "__main__":
    model.load('mnistrnn.model')
    showImg(image)
    processedImg = preprocessImg(image)
    print( np.round(model.predict(processedImg)) )
