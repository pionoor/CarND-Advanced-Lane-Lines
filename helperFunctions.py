# 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpImg
import glob
import sys

global_mtx = 0
global_dist = 0


def display2ImgsSideBySide(leftImgTitle, rightImgTitle, leftImg, rightImg):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.subplots_adjust(hspace = .2, wspace=.05)
    ax1.imshow(cv2.cvtColor(leftImg, cv2.COLOR_BGR2RGB))
    ax1.set_title(leftImgTitle, fontsize=30)
    ax2.imshow(cv2.cvtColor(rightImg, cv2.COLOR_BGR2RGB))
    ax2.set_title(rightImgTitle, fontsize=30)
    return

    
def undistort(image, mtx, dist):
    return cv2.undistort(image, mtx, dist, None, mtx)  



def threshold(image, thresh_min = 50, thresh_max = 250, s_thresh_min = 150, s_thresh_max = 255):
    
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal 
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # Convert to HLS color space and separate the S channel
    # Note: image is the undistorted image
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    sChannel = hls[:,:,2]
    s_binary = np.zeros_like(sChannel)
    s_binary[(sChannel >= s_thresh_min) & (sChannel <= s_thresh_max)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    combined_binary = combined_binary * 255
    
    return combined_binary 
    
def transform(image):
    imgShape = image.shape
    imgSize = (imgShape[1], imgShape[0]) # Image shape
    # The four points used as src points
    bottomLeft = [170,700] 
    bottomRight = [1150,700]
    topLeft = [570, 460]
    topRight = [720, 460]

    offset = 230;
    # for dst points
    distBottomLeft = [170 + offset, 710] 
    distBottomRight = [1150 - offset, 710]
    distTopLeft = [600 - offset, 0] 
    distTopRight = [700 + offset, 0]

    srcPoints = np.float32([bottomLeft, bottomRight , topLeft, topRight]) 
    dstPoints = np.float32([distBottomLeft, distBottomRight, distTopLeft, distTopRight])
    # Compute the perspective transform, M, given source and destination points:
    M = cv2.getPerspectiveTransform(srcPoints, dstPoints)
    Minv = cv2.getPerspectiveTransform(dstPoints, srcPoints)
    result = cv2.warpPerspective(image, M, imgSize, flags=cv2.INTER_LINEAR)
    return (Minv, result)

def areaOfInterest(img):

 
    leftLaneVertices = np.array([[(400, 720),(360, 0), (550, 0), (460, 720)]], dtype=np.int32)
    rightLaneVertices = np.array([[(880, 720),(810, 0), (1100, 0), (920, 720)]], dtype=np.int32)
  

    mask = np.zeros_like(img)   
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, leftLaneVertices, ignore_mask_color)
    cv2.fillPoly(mask, rightLaneVertices, ignore_mask_color)


    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# detect lane pixles and return lift and right fits.
def detectLane(image):
    binary_warped = image 
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
  
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    return (left_fit, right_fit, out_img)


def drawLane(undistortedImg, binary_warped, left_fit, right_fit, Minv):

    ym_per_pix = 30/720.0 # meters per pixel in y dimension
    xm_per_pix = 3.7/525.0 # meters per pixel in x dimension
    y_eval = 700
    midx = 680

    #binary_warped = cv2.cvtColor(binary_warped, cv2.COLOR_BGR2GRAY)
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (undistortedImg.shape[1], undistortedImg.shape[0])) 

    # Combine the result with the original image
    result = cv2.addWeighted(undistortedImg, 1, newwarp, 0.3, 0)
    y1 = (2*left_fit[0]*y_eval + left_fit[1])*xm_per_pix/ym_per_pix
    y2 = 2*left_fit[0]*xm_per_pix/(ym_per_pix*ym_per_pix)

    curvature = ((1 + y1*y1)**(1.5))/np.absolute(y2)
    cv2.putText(result,'Radius of Curvature: %.2fm' % curvature,(20,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    x_left_pix = left_fit[0]*(y_eval**2) + left_fit[1]*y_eval + left_fit[2]
    x_right_pix = right_fit[0]*(y_eval**2) + right_fit[1]*y_eval + right_fit[2]
    position_from_center = ((x_left_pix + x_right_pix)/2 - midx) * xm_per_pix
    if position_from_center < 0:
        text = 'left'
    else:
        text = 'right'
    cv2.putText(result,'Distance From Center: %.2fm %s' % (np.absolute(position_from_center), text),(20,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    return result



def pipeline(image, mtx=global_mtx, dist=global_dist):
    
    undistortedImg = undistort(image, mtx, dist)
    combined_binary = threshold(undistortedImg)
    Minv, binary_warped = transform(combined_binary)
    masked = areaOfInterest(binary_warped)
    left_fit, right_fit, out_img = detectLane(masked)
    result = drawLane(undistortedImg, masked, left_fit, right_fit, Minv)
    return result


def calibrateCam(folderPath, nx=9, ny=6):
    localPath = 'camera_cal/'
    nx = 9 # Number of the inside corners in x.
    ny = 6 # number of the inside corners in y.
    imgShape = mpImg.imread(folderPath + 'calibration1.jpg').shape[1::-1] # Image shape

    calibrationImgs = [] # Calibration images array.

    objPoints = [] # 3D object points in the real world space.
    imgPoints = [] # 3D points in the image plane.

    objP = np.zeros((nx*ny, 3), np.float32)
    objP[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2) # x, y coordinates.

    for imgFileName in glob.glob(folderPath + '*.jpg'):
        image = cv2.imread(imgFileName) #read each image.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to gray
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None) #find chessboard corners.

        if ret == True: # If corners found, add points, image points.
            imgPoints.append(corners)
            objPoints.append(objP)

            # Draw and display corners
            image = cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
            calibrationImgs.append(image)

    # Camera calibration, given object points, image points, and the shape of the grayscale image:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, imgShape, None, None)
    global_mtx = mtx
    globa_dist = dist
    return (mtx, dist, calibrationImgs)

