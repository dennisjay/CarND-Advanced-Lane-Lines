## Writeup for Project Advanced Lane Lines

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/undistortion.png "Road Undistorted"
[image3]: ./examples/bin.png "Biniarized image"
[image4]: ./examples/warp.png "Warped image image"
[image5]: ./examples/lane_line.png "Lane line detection"
[image6]: ./examples/pipeline_example.png "Lane line detection"
## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---


#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./lane_detection.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.


For the color trasformations I created the "binarize" function in the file "./lane_line_lib.py". At first I transformed the image to HLS color space. Then I applyied a Sobel filter to the l_channel. Afterwards I applied simple thresholds to the s_channel and the l_channel. 

I stacked all these three channels together with the "np.stack" function. Then I applied a simple noise_reduction filter to the images as seen in function "noise_reduction".

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 63 through 70 in the file `lane_line_lib.py` (or, for example, in the 6th code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src_pts = np.float32([[235, 697],[587,456],[700,456],[1061,690]])
tl = np.float32([src_pts[0, 0], 0])
tr = np.float32([src_pts[3, 0], 0])
dst_pts = np.float32([src_pts[0], tl, tr, src_pts[3]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 235, 697      | 235, 697      | 
| 587, 456      | 235, 0        |
| 700, 456      | 1061, 0       |
| 1061, 690     | 1061, 690     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The identification of lane line pixels is done with a simple windowing method. Which can be seen in line 104 to 127 in file `lane_line_lib.py`. If the number of active pixels in a certain window exceeds the number of 35 than the pixels are marked as lane line.
![alt text][image5]
The fitting was done in lines 139 through 141 in my code in `lane_line_lib.py` with the following code:

```python
# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
In in lines 216 through 248 in my code in `lane_line_lib.py` I calculate the curvature of the road as seen in udacity example.
 


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./processed_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

At first I had problems binarizing the image it was very hard to find the right parameters. 
A lot of time was spent to refactor the code in a good structure. After the review I fixed the curverad calculation. 

I recognized that the most sensitive part in the code is the binarization of the image. The algorithm will fail most likely 
at situations where the brightness changes because of shadows or old lane lines on the road. One proposal to make this more robust
is to add morphologyEx filter at the beginning of the pipeline. Another idea could be to use a CLAHE (Contrast Limited Adaptive Histogram Equalization) algorithm. 

Another weakness of this algorithm would be when there is a construction site or intersection
lane lines which are crossing or doubled can lead to that only one random of these lines is detected, for such situations this 
algorithm should be improved.
 