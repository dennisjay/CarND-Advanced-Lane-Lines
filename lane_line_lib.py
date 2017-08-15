import numpy as np
import cv2


class CameraCalibration:
    def __init__(self):
        self.calib_mtx = np.load('camera_cal/calib_mtx.npy')
        self.calib_dist = np.load('camera_cal/calib_dist.npy')
        self.objpoints = []
        self.imgpoints =[]
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((6 * 9, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    def add_calib_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret == True:
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners)


    def undistort(self, img):
        #undistort an image, use calib_mtx and calib_dist
        return cv2.undistort(img, self.calib_mtx, self.calib_dist, None, self.calib_mtx)


    def calc_params(self):
        # Do camera calibration given object points and image points
        _, self.calib_mtx, self.calib_dist, _, _ = cv2.calibrateCamera(self.objpoints, self.imgpoints, img_size, None, None)

        dst = cv2.undistort(img, mtx, dist, None, mtx)
        cv2.imwrite('calibration_wide/test_undist.jpg', dst)

        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        np.save('camera_cal/calib_mtx', mtx)
        np.save('camera_cal/calib_dist', dist)



class Binarizer:
    def __init__(self, gray_thresh=(20, 255), s_thresh=(170, 255), l_thresh=(30, 255), sobel_kernel=3):
        self.gray_threshold = gray_thresh
        self.s_treshold = s_thresh
        self.l_treshold = l_thresh
        self.sobel_kernel = sobel_kernel

    def binarize(self, img):
        image_copy = np.copy(img)

        # Convert to HLS color space and separate the S channel
        hls = cv2.cvtColor(image_copy, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]
        l_channel = hls[:, :, 1]

        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        thresh_min = self.gray_threshold[0]
        thresh_max = self.gray_threshold[1]
        sobel_x_binary = np.zeros_like(scaled_sobel)
        sobel_x_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_thresh_min = self.s_treshold[0]
        s_thresh_max = self.s_treshold[1]
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

        l_binary = np.zeros_like(l_channel)
        l_thresh_min = self.l_treshold[0]
        l_thresh_max = self.l_treshold[1]
        l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        binary = np.zeros_like(sobel_x_binary)
        binary[((l_binary == 1) & (s_binary == 1) | (sobel_x_binary == 1))] = 1
        binary = 255 * np.dstack((binary, binary, binary)).astype('uint8')

        #do a noise reduction
        return self.noise_reduction(binary)

    def noise_reduction(self, image, threshold=4):
        k = np.array([[1, 1, 1],
                      [1, 0, 1],
                      [1, 1, 1]])

        #apply 2d filter
        nb_neighbours = cv2.filter2D(image, ddepth=-1, kernel=k)
        image[nb_neighbours < threshold] = 0
        return image

class Warper:
    def __init__(self, src_points, dest_points):
        self.M = cv2.getPerspectiveTransform(src_points, dest_points)
        self.MI = cv2.getPerspectiveTransform(dest_points, src_points)

    def warp(self, img):
        #apply a perspective transform with given points
        warped = cv2.warpPerspective(img, self.M, img.shape[0:2][::-1], flags=cv2.INTER_LINEAR)
        return warped

    def unwarp(self, img ):
        warped = cv2.warpPerspective(img, self.MI, img.shape[0:2][::-1], flags=cv2.INTER_LINEAR)
        return warped

class LaneDetector:
    def __init__(self, nwindows=9, margin = 75, minpix = 35):
        # Choose the number of sliding windows
        self.nwindows = nwindows
        # Set the width of the windows +/- margin
        self.margin = margin
        # Set minimum number of pixels found to recenter window
        self.minpix = minpix

    def _histogram(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        binary_warped = binary_warped[:, :, 0]
        histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
        # Create an output image to draw on and  visualize the result
        self.out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        self.midpoint = np.int(histogram.shape[0] / 2)
        self.leftx_base = np.argmax(histogram[:self.midpoint])
        self.rightx_base = np.argmax(histogram[self.midpoint:]) + self.midpoint


    def _window_detection(self, binary_warped):
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        self.nonzero = binary_warped.nonzero()
        self.nonzeroy = np.array(self.nonzero[0])
        self.nonzerox = np.array(self.nonzero[1])
        # Current positions to be updated for each window
        leftx_current = self.leftx_base
        rightx_current = self.rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            # Draw the windows on the visualization image
            cv2.rectangle(self.out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(self.out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) & (self.nonzerox >= win_xleft_low) & (
                self.nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) & (self.nonzerox >= win_xright_low) & (
                self.nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(self.nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(self.nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        self.left_lane_inds = np.concatenate(left_lane_inds)
        self.right_lane_inds = np.concatenate(right_lane_inds)

    def extract_lane(self, binary_warped):
        self._histogram(binary_warped)
        self._window_detection(binary_warped)

        # Extract left and right line pixel positions
        self.leftx = self.nonzerox[self.left_lane_inds]
        self.lefty = self.nonzeroy[self.left_lane_inds]
        self.rightx = self.nonzerox[self.right_lane_inds]
        self.righty = self.nonzeroy[self.right_lane_inds]

        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
        self.right_fit = np.polyfit(self.righty, self.rightx, 2)

        self.ploty = np.linspace(0, self.out_img.shape[0] - 1, self.out_img.shape[0])
        self.left_fitx = self.left_fit[0] * self.ploty ** 2 + self.left_fit[1] * self.ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0] * self.ploty ** 2 + self.right_fit[1] * self.ploty + self.right_fit[2]


    def fill_lane_lines(self, image):
        self.image_size = image.shape
        #fills an area with given points with a green plane
        copy_image = np.zeros_like(image)
        fit_y = np.linspace(0, copy_image.shape[0] - 1, copy_image.shape[0])

        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, fit_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, fit_y])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(copy_image, np.int_([pts]), (0, 255, 0))

        return copy_image

    def calc_curverad(self):
        #calculates the curverad and lane deviation

        # first we calculate the intercept points at the bottom of our image
        left_intercept = self.left_fit[0] * self.image_size[0] ** 2 + self.left_fit[1] * self.image_size[0] + self.left_fit[2]
        right_intercept = self.right_fit[0] * self.image_size[0] ** 2 + self.right_fit[1] * self.image_size[0] + self.right_fit[2]

        # difference in pixels between left and right interceptor points
        road_width_in_pixels = right_intercept - left_intercept

        # average highway lane line width in US is about 3.7m
        meters_per_pixel_x_dir = 3.7 / road_width_in_pixels
        meters_per_pixel_y_dir = 30 / road_width_in_pixels

        # Recalculate road curvature in X-Y space
        ploty = np.linspace(0, 719, num=720)
        y_eval = np.max(ploty)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * meters_per_pixel_y_dir, self.left_fitx * meters_per_pixel_x_dir, 2)
        right_fit_cr = np.polyfit(ploty * meters_per_pixel_y_dir, self.right_fitx * meters_per_pixel_x_dir, 2)

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * meters_per_pixel_y_dir + left_fit_cr[1]) ** 2) ** 1.5) / \
                        np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * meters_per_pixel_y_dir + right_fit_cr[1]) ** 2) ** 1.5) / \
                         np.absolute(2 * right_fit_cr[0])


        # Next, we can lane deviation
        calculated_center = (left_intercept + right_intercept) / 2.0
        lane_deviation = (calculated_center - self.image_size[1] / 2.0) * meters_per_pixel_x_dir

        return left_curverad, right_curverad, lane_deviation

def pipeline(src_img):
    #the image pipeline
    #ready to process the video images
    src_pts = np.float32([[235, 697], [587, 456], [700, 456], [1061, 690]])
    tl = np.float32([src_pts[0, 0], 0])
    tr = np.float32([src_pts[3, 0], 0])
    dst_pts = np.float32([src_pts[0], tl, tr, src_pts[3]])

    #undistort
    calib = CameraCalibration()
    undistored = calib.undistort(src_img)

    #warp
    warper = Warper(src_pts, dst_pts)
    warped = warper.warp(undistored)

    #binarize
    binarizer = Binarizer()
    binarized = binarizer.binarize(warped)

    #detect
    detector = LaneDetector()
    detector.extract_lane(binarized)
    filled_image = detector.fill_lane_lines(warped)

    #unwarp and add
    unwarped_lanes = warper.unwarp(filled_image)
    result = cv2.addWeighted(undistored, 1, unwarped_lanes, 0.3, 0)

    #add measurments
    left_curvature, right_curvature, calculated_deviation = detector.calc_curverad()
    curvature_text = 'Left Curvature: {:.2f} m    Right Curvature: {:.2f} m'.format(left_curvature, right_curvature)

    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(result, curvature_text, (100, 50), font, 1, (221, 28, 119), 2)

    deviation_info = 'Lane Deviation: {:.3f} m'.format(calculated_deviation)
    cv2.putText(result, deviation_info, (100, 90), font, 1, (221, 28, 119), 2)

    return result



if __name__ == '__main__':
    #if this file is executed start creatiting the video clip
    from moviepy.editor import VideoFileClip


    output_file = './processed_project_video.mp4'
    input_file = './project_video.mp4'
    clip = VideoFileClip(input_file)
    out_clip = clip.fl_image(pipeline)
    out_clip.write_videofile(output_file, audio=False)