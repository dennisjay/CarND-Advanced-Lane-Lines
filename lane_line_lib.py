import numpy as np
import cv2


class LaneLineLib:
    def __init__(self):
        self.calib_mtx = np.load('camera_cal/calib_mtx.npy')
        self.calib_dist = np.load('camera_cal/calib_dist.npy')

    def undistort(self, img):
        return cv2.undistort(img, self.calib_mtx, self.calib_dist, None, self.calib_mtx)

    def binarize(self, img, gray_thresh=(20, 255), s_thresh=(170, 255), l_thresh=(30, 255), sobel_kernel=3):
        image_copy = np.copy(img)

        hls = cv2.cvtColor(image_copy, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]
        l_channel = hls[:, :, 1]

        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        thresh_min = gray_thresh[0]
        thresh_max = gray_thresh[1]
        sobel_x_binary = np.zeros_like(scaled_sobel)
        sobel_x_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        s_binary = np.zeros_like(s_channel)
        s_thresh_min = s_thresh[0]
        s_thresh_max = s_thresh[1]
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

        l_binary = np.zeros_like(l_channel)
        l_thresh_min = l_thresh[0]
        l_thresh_max = l_thresh[1]
        l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

        binary = np.zeros_like(sobel_x_binary)
        binary[((l_binary == 1) & (s_binary == 1) | (sobel_x_binary == 1))] = 1
        binary = 255 * np.dstack((binary, binary, binary)).astype('uint8')

        return self.noise_reduction(binary)

    def noise_reduction(self, image, threshold=4):
        k = np.array([[1, 1, 1],
                      [1, 0, 1],
                      [1, 1, 1]])
        nb_neighbours = cv2.filter2D(image, ddepth=-1, kernel=k)
        image[nb_neighbours < threshold] = 0
        return image

    def warp(self, img, src_points, dest_points):
        M = cv2.getPerspectiveTransform(src_points, dest_points)
        warped = cv2.warpPerspective(img, M, img.shape[0:2][::-1], flags=cv2.INTER_LINEAR)

        return warped

    def extract_lane(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        binary_warped = binary_warped[:, :, 0]
        histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 75
        # Set minimum number of pixels found to recenter window
        minpix = 35
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]
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

        ploty = np.linspace(0, out_img.shape[0] - 1, out_img.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        return out_img, left_fitx, right_fitx

    def fill_lane_lines(self, image, fit_left_x, fit_right_x):
        copy_image = np.zeros_like(image)
        fit_y = np.linspace(0, copy_image.shape[0] - 1, copy_image.shape[0])

        pts_left = np.array([np.transpose(np.vstack([fit_left_x, fit_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_right_x, fit_y])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(copy_image, np.int_([pts]), (0, 255, 0))

        return copy_image

    def pipeline(self, src_img):
        src_pts = np.float32([[235, 697], [587, 456], [700, 456], [1061, 690]])
        tl = np.float32([src_pts[0, 0], 0])
        tr = np.float32([src_pts[3, 0], 0])
        dst_pts = np.float32([src_pts[0], tl, tr, src_pts[3]])


        undistored = self.undistort(src_img)
        warped = self.warp(undistored, src_pts, dst_pts)
        binarized = self.binarize(warped)
        _, fitx_left, fitx_right = self.extract_lane(binarized)
        filled_image = self.fill_lane_lines(binarized, fitx_left, fitx_right)
        unwarped_lanes = self.warp(filled_image, dst_pts, src_pts)
        return cv2.addWeighted(undistored, 1, unwarped_lanes, 0.3, 0)


if __name__ == '__main__':
    from moviepy.editor import VideoFileClip

    lib = LaneLineLib()
    output_file = './processed_project_video.mp4'
    input_file = './project_video.mp4'
    clip = VideoFileClip(input_file)
    out_clip = clip.fl_image(lib.pipeline)
    out_clip.write_videofile(output_file, audio=False)