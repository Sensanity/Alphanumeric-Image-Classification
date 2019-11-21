from keras.models import load_model
from keras.preprocessing import image
import numpy as np


# dimensions of our images
img_width, img_height = 28, 28

# load the model we saved
model = load_model('alphanumeric_model.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

def classify(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=10)
    
    if classes == 0:
        print("0")
        
    if classes == 1:
        print("1")
    
    if classes == 2:
        print("2")
    
    if classes == 3:
        print("3")
    
    if classes == 4:
        print("4")
    
    if classes == 5:
        print("5")
    
    if classes == 6:
        print("6")
    
    if classes == 7:
        print("7")
    
    if classes == 8:
        print("8")
    
    if classes == 9:
        print("9")
    
    if classes == 10:
        print("A")
    
    if classes == 11:
        print("B")
    
    if classes == 12:
        print("C")
    
    if classes == 13:
        print("D")
    
    if classes == 14:
        print("E")
    
    if classes == 15:
        print("F")
    
    if classes == 16:
        print("G")
    
    if classes == 17:
        print("H")
    
    if classes == 18:
        print("I")
    
    if classes == 19:
        print("J")
    
    if classes == 20:
        print("K")
    
    if classes == 21:
        print("L")
    
    if classes == 22:
        print("M")
    
    if classes == 23:
        print("N")
    
    if classes == 24:
        print("O")
    
    if classes == 25:
        print("P")
    
    if classes == 26:
        print("Q")
    
    if classes == 27:
        print("R")
    
    if classes == 28:
        print("S")
    
    if classes == 29:
        print("T")
    
    if classes == 30:
        print("U")
    
    if classes == 31:
        print("V")
    
    if classes == 32:
        print("W")
    
    if classes == 33:
        print("X")
    
    if classes == 34:
        print("Y")
    
    if classes == 35:
        print("Z")
    
  
    return 

# predicting images
img = image.load_img('report/letter_H_2.jpg', target_size=(img_width, img_height))
classify(img)



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    