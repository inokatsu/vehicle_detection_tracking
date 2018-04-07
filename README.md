
# Vehicle Detection and Tracking

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/test_images.png
[image2]: ./output_images/hog_image_example.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./output_images/find_car_image.png
[image5]: ./output_images/pipeline_images.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png

---


## Histogram of Oriented Gradients (HOG)

#### 1. Get Train Images

The code for this step is contained in 'Get Train Image' of the Jupyter notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  The figures blow shoes a random samples of the both classe images.

![alt text][image1]

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed a image from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(14, 14)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and I settled on my final choice of HOG parameters based upon the accuracy of the SVM classifier produced using them. The final parameters chosen were : YUV colorspace, 11 orientations, 8 pixels per cell, 2 cells per block, and `ALL` channels of the colorspace. 

#### 3. Train classifier

I trained a linear SVM with the default classifier parameters and using HOG features, spatial intensity and channel intensity histogram features and was able to achieve a test accuracy of 98.82%.
<br />
<br />

## Sliding Window Search

#### 1. Hog Sub-sampling Window Search

Extracting HOG features from each individual window across the image turns out this is rather inefficient. To speed things up, extract HOG features just once for the entire region of interest and subsample that array for each sliding window. To do this, apply skimage.feature.hog() with the flag feature_vec=False, like this:  
  
```
from skimage.feature import hog  
orient = 11 
pix_per_cell = 14
cell_per_block = 2  
features, hog_image = hog(img, orientations=orient,
                            pixels_per_cell=(pix_per_cell, pix_per_cell),
                            cells_per_block=(cell_per_block, cell_per_block),
                            block_norm='L2-Hys',
                            transform_sqrt=False,
                            visualise=vis, feature_vector=feature_vec)
```  


#### 2. Pipeline

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector. Here are some example images:

![alt text][image4]
---

## Video Implementation

By using `process_image` function I created a video.
Here's a [link to my video result](https://youtu.be/sfdABTbb2hE)

  
<br />
      

## Filter for false positives and combining overlapping bounding boxes

From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here are example image with corresponding heatmaps,the resulting bounding boxes:

![alt text][image5]  
 
 
 
 Then I saved the heatmap across the last 20 frames by using  `deque` function to make the bounding boxes smoother and reduce the false positives. 


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

At the scene of the shadow area on the road, the classifier detect a lot of false positives. To filtering out of this false positives, I set a high threshold for the heat map. It successed to reduce the false positives, but it also reduce the positive detection area. Therefore it will be needed to find more better color space to avoid detecting the false positive or augment training image and train the model further.

