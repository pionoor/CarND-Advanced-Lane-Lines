
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
[image2]: ./output_images/undistorted/test1.jpg "Road Transformed"
[image3]: ./output_images/thresholded/test1.jpg "Binary Example"
[image4]: ./output_images/undistorted_points_drawn.jpg "Warp Example"
[image5]: ./output_images/birdeye.jpg "Warp Example"
[image6]: ./examples/color_fit_lines.jpg "Fit Visual"
[image7]: ./oput_images/pipeline_output/test1.jpg "Output"
[video1]: ./project_video.mp4 "Video"


### Camera Calibration


The code for this step is contained in the first code cell of the IPython notebook located in "./examples/main.ipynb." A helper function is called, calibrateCam(folderPath="camera_cal/", nx=9, ny=6), takes the path to the folder where the chessboard calibration images are located. The function returns   

Inside the "calibrateCam" function, inside `helperFunctions.py` file, "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

A function called, `threshold(image, thresh_min = 20, thresh_max = 100, s_thresh_min = 170, s_thresh_max = 255)`, inside helper 'helperFunctions.py' file, take an image and convert it into a binary one. Inside the function, I used different techniques to come up with a desired clear result, by converting the image into HLS color space and use the S channel and some thresholding. Here's an example of my output for this step:
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `transform(image)`, which appear is inside `helperFunctions.py`.  The  function takes as inputs an image (`img`) and return the wraped image. I chose the hardcode the source and destination points inside the function the following manner:

```python
  # The four points used as src points
    bottomLeft = [170,710] 
    bottomRight = [1150,710]
    topLeft = [570, 470]
    topRight = [720, 470]

    offset = 210;
    # for dst points
    distBottomLeft = [170 + offset, 710] 
    distBottomRight = [1150 - offset, 710]
    distTopLeft = [600 - offset, 0] 
    distTopRight = [700 + offset, 0]
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 170,710      | 380, 710        | 
| 1150,710     | 940, 710      |
| 1570, 470     | 390, 0     |
| 6720, 470      | 490, 0       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]
![alt text][image5]
#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Inside `helperFunctions.py`, the `detectLane(image)` takes a warped, binary image and finds the pixel that represents the lane lines, and then try to fit 2nd order polynomial lines. The function draw the left and the right fits on the image and return the result.

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this using the code below:

```python
ym_per_pix = 3.0/72.0 # meters per pixel in y dimension
xm_per_pix = 3.7/660.0 # meters per pixel in x dimension
y_eval = 700
midx = 650

y1 = (2*left_fit[0]*y_eval + left_fit[1])*xm_per_pix/ym_per_pix
y2 = 2*left_fit[0]*xm_per_pix/(ym_per_pix*ym_per_pix)

curvature = ((1 + y1*y1)**(1.5))/np.absolute(y2)
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
