from keras.models import Sequential # init neural network
from keras.layers import Convolution2D # Add conv layers for images
from keras.layers import MaxPooling2D # add pooling layers
from keras.layers import Flatten # convert all pooled feature maps into large feature vector
from keras.layers import Dense # add fully connected layers in a neural network

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape=(28, 28, 3), activation="relu"))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
#classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(activation="relu", units=128))
classifier.add(Dense(activation="softmax", units=36))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (28, 28),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('test',
                                            target_size = (28, 28),
                                            batch_size = 32,
                                            class_mode = 'categorical')




#import os, os.path
#
#
## path joining version for other paths
#DIR = 'H'
#print (len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))


classifier.fit_generator(training_set,
                         samples_per_epoch = 24212,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 3662)

classifier.save('alphanumeric_model.h5')























