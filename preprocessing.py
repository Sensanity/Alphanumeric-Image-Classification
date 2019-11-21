import requests
import sys
from tqdm import tqdm
import os
import tarfile
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import string


# Determine root: Import and preprocess form root
try:
    root = os.path.dirname(os.path.abspath(__file__))
except NameError:
    import sys
    root = os.path.dirname(os.path.abspath(sys.argv[0]))
 
print (root)


fileurl = 'http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz'
filename = 'EnglishFnt.tgz'
if not os.path.exists(filename):
    # Gets response of 200 so the request was successful
    r = requests.get(fileurl, stream=True)
    print(r)
    with open(filename, 'wb') as f:
        #transfer file froms server (Server limit 1.024 GB)
        for chunk in tqdm(r.iter_content(1024), unit='KB', total=int(r.headers['Content-Length'])/1024): 
            f.write(chunk)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def rmdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
# Extract zip file 
with tarfile.open(filename, 'r') as tfile:
    print ('loading')
    members = tfile.getmembers()  
    for member in tqdm(members):
        if tarfile.TarInfo.isdir(member):
            mkdir(member.name)
            continue
        with open(member.name, 'wb') as f:
            f.write(tfile.extractfile(member).read())

# Rename first 10 folders 0-9
# Rename next 26 folders A-Z
path = 'English/Fnt/'    
file_count = 0
letters = list(string.ascii_uppercase)
letter_count = 0
for filename in os.listdir(path):
    if file_count <= 9:
        os.rename(os.path.join(path,filename), os.path.join(path,str(file_count)))
        file_count += 1
    elif file_count >= 10:
       os.rename(os.path.join(path,filename), os.path.join(path,letters[letter_count]))
       file_count += 1
       letter_count += 1
    elif file_count == 37:
        break
        
# ===============================================================================================

# Resize images: 28 by 28
def resize(rawimg):
    fx = 28.0 / rawimg.shape[0]
    fy = 28.0 / rawimg.shape[1]
    
    # Make height and withd equal for normalization
    fx = fy = min(fx, fy)
    
    # Resize the image based on x and y.
    # Uses a bicubic interpolation over 4×4 pixel neighborhood
    img = cv2.resize(rawimg, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    
    # Empty 28 b 28 
    outimg = np.ones((28, 28), dtype=np.uint8) * 255
    
    w = int(img.shape[1])
    h = int(img.shape[0])
    x = int((28 - w) / 2)
    y = int((28 - h) / 2)
    outimg[y:y+h, x:x+w] = img
    return outimg

def convert(imgpath):
    img = cv2.imread(imgpath)
    # Make image blackwhite
    gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    # if pixel value > threshold, pixel = x, else pixel = y (Adaptive Thresholding)
    # Made background black and object white for contour detection to work
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 25)
    
    # Join points with same color intensity
    # Make a shallow copy
    # return the extreme outer flags
    # Compress contours and remove redundant points
    # https://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html
    # https://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html
    
    ctrs, hier = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # make a up-right bounding rectangle for the exreme countours
    # https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=boundingrect#boundingrect
    # https://www.programcreek.com/python/example/89437/cv2.boundingRect
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    x, y, w, h = rects[-1]
    roi = gray[y:y+h, x:x+w]
    return resize(roi)

# Example comparison
imgpath = 'English/Fnt/Sample001/img001-00001.png'
img = cv2.imread(imgpath)
rsz = convert(imgpath)

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(rsz, cmap='gray')
# ===================================================================================================================

# Preprocess all the number images
for i in range(0,10):
    path = 'English/Fnt/%d/' % (i)
    trainpath = 'train/%d/' % i
    mkdir(trainpath)
    for filename in tqdm(os.listdir(path), desc=trainpath):
        try:
            cv2.imwrite(trainpath + filename, convert(path + filename))
        except:
            pass

# Preprocess all the alphabets
letters = list(string.ascii_uppercase)
for i in range(0,26):
    path = 'English/Fnt/%s/' % (letters[i])
    trainpath = 'train/%s/' % (letters[i])
    mkdir(trainpath)
    for filename in tqdm(os.listdir(path), desc=trainpath):
        try:
            cv2.imwrite(trainpath + filename, convert(path + filename))
        except:
            pass

# Split train and test sets for numbers
for i in range(0, 10):
    trainpath = 'train/%d/' % i
    testpath = 'test/%d/' % i
    mkdir(testpath)
    imgs = os.listdir(trainpath)
    trainimgs, testimgs = train_test_split(imgs, test_size=0.1)
    for filename in testimgs:
        os.rename(trainpath+filename, testpath+filename)

# Split train and test sets for alphabets
letters = list(string.ascii_uppercase)       
for i in range(0, 26):
    trainpath = 'train/%s/' % (letters[i])
    testpath = 'test/%s/' % (letters[i])
    mkdir(testpath)
    imgs = os.listdir(trainpath)
    trainimgs, testimgs = train_test_split(imgs, test_size=0.1)
    for filename in testimgs:
        os.rename(trainpath+filename, testpath+filename)

# Make validation set for numbers
for i in range(0, 10):
    trainpath = 'train/%d/' % i
    testpath = 'validation/%d/' % i
    mkdir(testpath)
    imgs = os.listdir(trainpath)
    trainimgs, testimgs = train_test_split(imgs, test_size=0.2)
    for filename in testimgs:
        os.rename(trainpath+filename, testpath+filename)

# Make validation set for alphabets
letters = list(string.ascii_uppercase)       
for i in range(0, 26):
    trainpath = 'train/%s/' % (letters[i])
    testpath = 'validation/%s/' % (letters[i])
    mkdir(testpath)
    imgs = os.listdir(trainpath)
    trainimgs, testimgs = train_test_split(imgs, test_size=0.2)
    for filename in testimgs:
        os.rename(trainpath+filename, testpath+filename)























