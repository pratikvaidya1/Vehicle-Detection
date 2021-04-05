import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
import pickle
from scipy.ndimage.measurements import label




# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)






# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows






pickle_file = 'classifier_rbf.p'
with open(pickle_file, 'rb') as f:
    pickle_data = pickle.load(f)
    # X_train = pickle_data['train_dataset']
    # y_train = pickle_data['train_labels']
    # X_test = pickle_data['test_dataset']
    # y_test = pickle_data['test_labels']
    X_scaler = pickle_data['X_scaler']
    parameters = pickle_data['parameters']
    svc = pickle_data['svc']

    del pickle_data  # Free up memory

print('classifier Data and modules loaded.')

for k in parameters:
    print(k, ":", parameters[k])





def find_cars(image, parameters=parameters, X_scaler=X_scaler, scv=svc):


    color_space = parameters['color_space']
    orient = parameters['orient']
    pix_per_cell = parameters['pix_per_cell']
    cell_per_block = parameters['cell_per_block']
    hog_channel = parameters['hog_channel']
    spatial_size = parameters['spatial_size']
    hist_bins = parameters['hist_bins']
    spatial_feat = parameters['spatial_feat']
    hist_feat = parameters['hist_feat']
    hog_feat = parameters['hog_feat']
    
    draw_image = np.copy(image)
   
    image = image.astype(np.float32) / 255
    y_start_stop = [400, 650]


    # setting up 4 different sliding windows to find cars in the image and then drawwing the boxes

    # windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
    #                         xy_window=(32, 32), xy_overlap=(0.5, 0.5))
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400,500],
                            xy_window=(64, 64), xy_overlap=(0.8, 0.8))

    # raw_window_image = draw_boxes(draw_image, windows, color=(0, 0, 255), thick=6)


    windows += slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                            xy_window=(96, 96), xy_overlap=(0.8, 0.8))

    # raw_window_image = draw_boxes(raw_window_image, windows, color=(0, 255, 0), thick=6)

    windows += slide_window(image, x_start_stop=[1000, None], y_start_stop=y_start_stop,
                            xy_window=(128, 128), xy_overlap=(0.8, 0.8))

    # raw_window_image = draw_boxes(raw_window_image, windows, color=(255, 0, 0), thick=6)


    # windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[350,None],
    #                         xy_window=(256, 256), xy_overlap=(0.75, 0.75))

    # print('\nlooping over:',len(windows))

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)

    window_image = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    heat = np.zeros_like(image[:, :, 0]).astype(np.float)

    # Adding heat to each box in the box list
    heat = add_heat(heat, hot_windows)
    # Applying thresholds to help remove false positives
    heat = apply_threshold(heat, 1)
    # Visualizing the heatmap while displaying
    heatmap = np.clip(heat, 0, 255)
    # Finding final boxes from heatmap with the help of label function
    labels = label(heatmap)
    draw_image = draw_labeled_bboxes(np.copy(draw_image), labels)

    # plt.close("all")
    #
    # # fig = plt.figure()
    # # plt.figure(figsize=(20,10))
    #
    # plt.subplot(133)
    # plt.imshow(draw_image)
    # plt.title('Car Positions')
    # plt.subplot(132)
    # plt.imshow(heatmap, cmap='hot')
    # plt.title('Heat Map')
    # plt.subplot(131)
    # plt.imshow(window_image)
    # plt.title('Windows')
    # # fig.tight_layout()
    # # mng = plt.get_current_fig_manager()
    #
    # # mng.full_screen_toggle()
    # # plt.pause(0.05)
    #
    # # plt.imshow(window_img)
    # plt.show()
    return heatmap



# image = mpimg.imread('test_images/bbox-example-image.jpg')
test_images_path = glob.glob('test_images/*.jpg')


for test_image_path in test_images_path:
    image = mpimg.imread(test_image_path)
    out = find_cars(image)
    output_path = test_image_path.replace('test_images', 'images/heat/')
    #cv2.imwrite(output_path,out)
    #out_path = test_image_path.replace('test_images', 'output_images')
    #cv2.imwrite(out_path,out)
    mpimg.imsave(output_path, out)


# plt.imshow(out)
# plt.show()


#from moviepy.editor import VideoFileClip
#
# # output = 'test_cars.mp4'
# # clip1 = VideoFileClip("test.mp4")
# output = 'project_video_cars_1.mp4'
# clip1 = VideoFileClip("project_video.mp4")
# # output = 'test_video_cars.mp4'
# # clip1 = VideoFileClip("test_video.mp4")
#
#
# # left_lane = Line()
# # right_lane = Line()
# clip = clip1.fl_image(find_cars) #NOTE: this function expects color images!!
# # clip.show()
# clip.write_videofile(output, audio=False)


































