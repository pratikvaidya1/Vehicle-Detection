import cv2
import numpy as np
import glob
import time
import pickle
import os

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split

from lesson_functions import *


# Read in cars and notcars examples

cars = []
notcars = []

notcars_path = glob.glob('./data/non-vehicles/*/*.png')
for img in notcars_path:
    notcars.append(img)

cars_path = glob.glob('./data/vehicles/*/*.png')
for img in cars_path:
    cars.append(img)


# Reduce the sample size because the quiz evaluator times out after 13s of CPU time
# sample_size = 1000
# cars = cars[0:sample_size]
# notcars = notcars[0:sample_size]


# Tunning these parameters to visualise change in results.
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 16  # Number of histogram bins
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
#y_start_stop = [450, None]  # Min and max in y to search in slide_window()


# Extracting features for cars and notcars (refered from lecture videos)
car_features = extract_features(cars, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)


X = np.vstack((car_features, notcar_features)).astype(np.float64)
# standardize the dataset
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)


# label vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Spliting up dataset into randomized training set and test set
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:', orient, 'orientations', pix_per_cell,'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))

# Using SVC
svc = SVC(C=5.0, gamma='auto', kernel='rbf')


# Checking the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds elapsed while training SVC...')
# Checking the accuracy score of the SVC
print('Test_Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()



# storring parameters in a dictionary 
parameters = {'color_space': color_space,
              'orient': orient,
              'pix_per_cell': pix_per_cell,
              'cell_per_block': cell_per_block,
              'hog_channel': hog_channel,
              'spatial_size': spatial_size,
              'hist_bins': hist_bins,
              'spatial_feat': spatial_feat,
              'hist_feat': hist_feat,
              'hog_feat': hog_feat}


# Saving data in pickle file for easy accessibility
pickle_file_path = 'classifier_rbf.p'
if not os.path.isfile(pickle_file_path):
    print('Saving data to pickle file...')
    try:
        with open(pickle_file_path, 'wb') as pickle_file:            
            pickle.dump(
                {
                    'X_scaler': X_scaler,
                    'parameters': parameters,
                    'svc': svc,
                },
                pickle_file, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('failed to save data into', pickle_file_path, ':', e)
        raise























