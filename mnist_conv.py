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

#displays image of choice
def showImg(image):
    im = Image.open(image)
    im.thumbnail((28, 28), Image.ANTIALIAS)
    imgplot = plt.imshow(im)
    plt.show()

#processes image of choice for model to make prediction
def preprocessImg(image):
    #read image file as grayscale matrix
    img = scipy.misc.imread(image, flatten=True, mode='L')
    
    #convert to 28 by 28
    img.resize((28, 28))
    img = img.reshape([-1, 28, 28])

    #turn pixel values (0-255) into decimals between 0 and 1
    #where the larger the number, the darker the pixel (mnist form)
    imgopt = []
    for z in img[0]:
        imgopt.append(["{0:.8f}".format(round(1-i/255,8)) for i in z])
    imgopt = np.array(imgopt)
    imgopt = imgopt.reshape([-1, 28, 28])
    return(imgopt)

#xs and ys
X, Y, test_x, test_y = mnist.load_data(one_hot=True)
X = X.reshape([-1, 28, 28])
test_x = test_x.reshape([-1, 28, 28])

#neural net
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
    
    #make prediction on the digit displayed in the image
    print( np.round(model.predict(processedImg)) )
