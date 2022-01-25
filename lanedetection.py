
import cv2
import numpy as np
import matplotlib.pyplot as plt
import lanefilter as filter

#A basic lane detection algorithm based adapted from from the approach outlined in
    #https://automaticaddison.com/the-ultimate-guide-to-real-time-lane-detection-using-opencv/
#and based on OpenCV

#Description: The Lane class and its functions which construct a lane object from an image

class Lane:
    """
    Digital model of a track lane
    """
    def __init__(self, original, roi_corners=None):
        """
        Default constructor

        param original: The original image as it comes from the camera
        param roi_corners: A an array of tuples containing the xy coordinates of the corners of the region of interest
        """

        self.original = original

        #Holds the dimensions of the original image (Height,Width)
        self.orig_dim = self.original.shape[:2][::-1]
        self.width = self.orig_dim[0]
        self.height = self.orig_dim[1] 

        #Holds the image once filtered to show lane lines
        self.filtered_lanes = None

        #Holds the "birds eye view" image (the product of a perspective transform) and its transformation and inverse transformation matrices
        self.pwarped = None
        self.color_pwarped = None
        self.pwarp_matrix = None
        self.inv_pwarp_matrix = None
        
        #Holds an array of tuples representing xy coordinates of the four corners of the region of interest
        if roi_corners is None:
            roi_corners = [(300,170), (0, 330), (self.width,330), (self.width-240 ,170)]

        self.roi_corners = np.float32(roi_corners)

        #Holds the target xy coordinates of the region of interest after perspective transform
        self.padding = int(0.25*self.width) #gap 
        self.target_roi_corners = np.float32([
            [self.padding, 0],  #Top-left corner
            [self.padding, self.height],  #Bottom-left corner
            [self.width-self.padding, self.height],  #Bottom right corner
            [self.width-self.padding, 0]])  #Top right corner

        #Holds a a histogram that shows the location of white peaks (representing lane lines) over the bottom half of the ROI
        self.histogram = None

        #Holds the sliding window parameters
        self.num_swindows = 10
        self.margin = int((1/12) * self.width) #The window width is 2*margin
        self.minpix = int((1/2) * self.margin) #The minimum number of pixels to relocate sliding window center

        #Holds the location data of the pixels that are part of lane lines
        self.llane_inds = None  #The indices of the left line pixels
        self.rlane_inds = None  #The indices of the right line pixels
        self.l_x = None  #The x coordinates of the left line pixels
        self.r_x = None  #The x coordinates of the right line pixels
        self.l_y = None  #The y coordinates of the left line pixels
        self.r_y = None  #The y coordinates of the right line pixels

        #Holds booleans which flag if the left and right lanes have been detected
        self.l_dtct = None
        self.r_dtct = None
        self.p_dtct = None
        #An identifier for which path was selected: 0=center of lane, 1=left of camera, 2=right of camera       
        self.p_iden = None

        #Holds the best fit polynomial lines for the left and right lanes
        self.lfit = None  #The left fit coefficients
        self.rfit = None  #The right fit coefficients
        self.ploty = None  #A vector of y coordinates
        self.lfit_x = None  #The x coordinates for the left line
        self.rfit_x = None  #The x coordinates for the right line

        #Holds the best fit polynomial lines for the center of lane path
        self.path_fit = None  #The path fit coefficients
        self.path_fit_x = None #The x coordinates of the path

        #Holds the state information from the lane detection computer vision algorithm
        self.cv_state = None

        #Conversions from pixels to meters in x and y dimensions 
        self.y_m2p = 10.0 / 1000 #meters per pixel in x direction


    def get_filtered(self, img=None, thresholds=None, median=False, canny=False, plot=False):
        """
        Applies the filtering functions from lanefilter.py to produce a binary image that shows lane lines

        param img: the input image
        return: A binary (0, 255) image containing the lane lines
        """
        if img is None:
            img = self.original
        
        if thresholds == None:
            #A tuple of tuples of thresholds for each of the thresholding operations
            thresholds = ((180, 255), (110, 255), (60,255), (180,255))
        
        #Converts the image from the BGR color space to the HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        #------------Isolates the edges of lane lines-----------#
        #Apply a binary (0,255) threshold to the lightness channel of the image
        _, l_thresholded = filter.threshold(hls[:,:,1], thresh=thresholds[0])
        
        if median == True:
            #Apply a median blur to the thresholded image to reduce noise
            l_blurred = filter.median_blur(l_thresholded, median_kernel=5)
        else:
            #Apply a gaussian blur to the thresholded image to reduce noise 
            l_blurred = filter.gaussian_blur(l_thresholded, gauss_kernel=3)

        if canny == True:
            #Apply Canny edge detection to the blurred image
            edge_detected = filter.canny(l_blurred, (170, 200), aperture_size=3)
        else:
            #Apply Sobel edge detection to the blurred image and make the image binary (0,1)
            edge_detected = filter.sobel(l_blurred, sobel_kernel=3, thresh=thresholds[1])

        #----------Isolates the interior (fill) of lane lines----------#
        #Apply a binary (0,255) threshold to the saturation channel of the image
        _, s_thresholded = filter.threshold(hls[:,:,2], thresholds[2])

        #Apply a binary (0,255) threshold to the red channel of the image
        _, r_thresholded = filter.threshold(img[:,:,2], thresholds[3])
        
        #Apply an and to combine the saturation and red thresholded image to filter for pixels that are red and have high saturation
        rsthresholded = cv2.bitwise_and(s_thresholded, r_thresholded)

        #----------Combine the edges and interior of lane lines----------#
        #Apply a or to combine the lane line egdes and interiors
        self.filtered_lanes = cv2.bitwise_or(rsthresholded, edge_detected.astype(np.uint8))

        if plot == True:
            #Plot the figures
            figure, ((axis1, axis4), (axis2, axis5), (axis3, axis6)) = plt.subplots(3,2)
            figure.set_size_inches(10, 10)
            figure.tight_layout(pad=3.0)
            axis1.imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
            axis2.imshow(l_thresholded, cmap='gray')
            axis3.imshow(edge_detected, cmap='gray')
            axis4.imshow(s_thresholded, cmap='gray')
            axis5.imshow(r_thresholded, cmap='gray')
            axis6.imshow(self.filtered_lanes, cmap='gray')
            axis1.set_title("Original")  
            axis2.set_title("Luminosity Thresholded")
            axis3.set_title("Edge Detected")
            axis4.set_title("Saturation Thresholded")
            axis5.set_title("Red Thresholded")
            axis6.set_title("Fully Filtered")
            plt.show()

        return self.filtered_lanes

    def perspective_transform(self, img=None, set=False, plot=False):
        """
        Transform the image to a birds eye view

        param img: the input image
        param set: a boolean of whether or not to set self.pwarped
        param plot: a boolean of whether or not plot the result
        return: A birds eye view of the region of interest
        """

        if img is None:
            img = self.filtered_lanes

        #Find the transformation matrix
        self.pwarp_matrix = cv2.getPerspectiveTransform(self.roi_corners, self.target_roi_corners)

        #Transform to birds eye view
        pwarped = cv2.warpPerspective(img, self.pwarp_matrix, self.orig_dim, flags=cv2.INTER_LINEAR)

        if set is True:
            #Convert image to binary
            (thresh, binary_pwarped) = cv2.threshold(pwarped, 127, 255, cv2.THRESH_BINARY)
            self.pwarped = binary_pwarped
        
        #Display the perspective warped image
        if plot == True:
            roi_display = self.pwarped.copy()
            roi_display = cv2.polylines(roi_display, np.int32([self.target_roi_corners]), True, (150,150,150), 3)
            figure, (axis1) = plt.subplots(1,1)
            figure.set_size_inches(10, 6)
            figure.tight_layout(pad=3.0)
            axis1.imshow(roi_display, cmap='gray')
            axis1.set_title("Perspective Transform")
            plt.show()

        return pwarped

    def generate_histogram(self, img=None, plot=False):
        """
        Generate a histogram to isolate peaks in the image

        param img: the input image
        param plot: a boolean of whether to plot the result
        return: the histogram
        """

        if img is None:
            img = self.pwarped
        
        padding = self.padding
        #Generate the histogram (summing over the bottom half of the image within the region of interest
        self.histogram = np.sum(img[int(img.shape[0]/2):, padding:img.shape[1] - padding], axis=0)

        if plot == True:
            figure, (axis1,axis2) = plt.subplots(2,1)
            figure.set_size_inches(10,5)
            figure.tight_layout(pad=3.0)
            axis1.imshow(img, cmap='gray')
            axis1.set_title("Warped Binary Frame")
            axis2.plot(self.histogram)
            axis2.set_title("Histogram Peaks")
            plt.show()

        return self.histogram

    def histogram_peaks(self):
        """
        Get the lane line associated peaks form the histogram

        return: the x coordinates of the left and right histogram peaks
        """
        #Store the middle of the histogram
        midpoint = int(self.histogram.shape[0]/2)

        #Find the left peak (the max left of the midpoint)
        leftx_coord = np.argmax(self.histogram[:midpoint]) + self.padding

        #Find the right peak (the max right of the midpoint)
        rightx_coord = np.argmax(self.histogram[midpoint:]) + midpoint + self.padding

        return leftx_coord, rightx_coord

    @staticmethod
    def gen_poly_points(ydim, fit):
        """
        A function to calculate the xy coordinates of a polynomial fit

        param ydim: The desired y dimension of the fit
        param fit: A 1D array (vector) of polynomial fit coefficients ordered from highest power to lowest
        return: Two arrays of ydim of y coordinates and x coordinates for the fit
        """

        n = fit.shape[0]
        powers = np.arange(n)[::-1]

        #Create a vector of y points
        ploty = np.linspace(0, ydim - 1, ydim)

        #Create an array of y point of size n*picture height
        plotyn = np.transpose([ploty]*n)
        
        #Raise each term to its respective power
        powered = np.power(plotyn, powers)

        #Multiply each term by its lfit coffiecient
        terms = fit * powered
        
        #Sum each ypoints terms
        fit_x = np.sum(terms, axis=1)

        return ploty, fit_x

    def find_laneline_polys(self, plot=False):
        """
        Get the indices of the lane line pixels with sliding windows

        param plot: a boolean of whether to plot the result
        return: best fit polynomials for the left and right lane lines
        """
        #Sliding winow margin (width=2*margin)
        margin = self.margin

        if plot == True:
            #Create a temporary copy of the warped image to use with the sliding window
            sw_display = self.pwarped.copy()

        #Calculate the sliding window height
        wheight = int(self.pwarped.shape[0]/self.num_swindows)

        #Store the x and y coordinates of all the nonzero pixels
        nonzero = self.pwarped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        #Initate list to store pixel indices for the left and right lane lines
        left_lane_pix = []
        right_lane_pix = []
        l_pixcount = 0
        r_pixcount = 0

        #Establish center of the sliding windows
        leftx_coord, rightx_coord = self.histogram_peaks()
        leftx_current = leftx_coord
        rightx_current = rightx_coord

        numwin = self.num_swindows

        for window in range(numwin):
            #Establish x y coordinates of window corners
            win_top = self.pwarped.shape[0] - (window + 1) * wheight
            win_bottom = self.pwarped.shape[0] - window * wheight
            lw_left = leftx_current - margin
            lw_right = leftx_current + margin
            rw_left = rightx_current - margin
            rw_right = rightx_current + margin

            if plot == True:
                #Draw the sliding window on the display image
                cv2.rectangle(sw_display,(lw_left, win_top),(lw_right, win_bottom), (255,255,255), 2)
                cv2.rectangle(sw_display,(rw_left, win_top),(rw_right, win_bottom), (255,255,255), 2)

            #Extract the indices (number in the "nonzero" list) of the nonzero pixels that fall within the window
            win_left_pix = ((nonzeroy >= win_top) & (nonzeroy < win_bottom) & (nonzerox >= lw_left) & (nonzerox < lw_right) & (nonzerox >= self.padding) & (nonzerox < self.pwarped.shape[1] - self.padding)).nonzero()[0]
            win_right_pix = ((nonzeroy >= win_top) & (nonzeroy < win_bottom) & (nonzerox >= rw_left) & (nonzerox < rw_right) & (nonzerox >= self.padding) & (nonzerox < self.pwarped.shape[1] - self.padding)).nonzero()[0]

            #Append (add) the array of indices of the nonzero pixels in the current window to the array of arrays
            left_lane_pix.append(win_left_pix)
            right_lane_pix.append(win_right_pix)

            #If > the minimum number of pixels were found, recenter the window at the mean x position of the located pixels
            minpix = self.minpix
            if len(win_left_pix) > minpix:
                leftx_current = int(np.mean(nonzerox[win_left_pix]))
                l_pixcount = l_pixcount + 1
            if len(win_right_pix) > minpix:
                rightx_current = int(np.mean(nonzerox[win_right_pix]))
                r_pixcount = r_pixcount + 1
        
        if l_pixcount > 1:
            #Check to see if at least 2 windows met the required number of pixels
            self.l_dtct = True
            #Concatenate the array of arrays into one array
            left_lane_pix = np.concatenate(left_lane_pix)

            #Get the xy coordinates of the pixels from the array of indices
            leftx = nonzerox[left_lane_pix]
            lefty = nonzeroy[left_lane_pix]

            #Fit a 2nd order polynomial curve to the pixel coordinates for each lane line
            lfit = np.polyfit(lefty, leftx, 2)
            self.lfit = lfit
        else:
            self.l_dtct = False
            print("No left lane detected")

        if r_pixcount > 1:
            #Repeat for right lane
            self.r_dtct = True
            right_lane_pix = np.concatenate(right_lane_pix)
            rightx = nonzerox[right_lane_pix]
            righty = nonzeroy[right_lane_pix]
            rfit = np.polyfit(righty, rightx, 2)
            self.rfit = rfit
        else:
            self.r_dtct = False
            print("No left lane detected")

        if (plot and self.l_dtct and self.r_dtct) == True:

            #Generate the x and y points for the polynomials
            self.ploty, lfit_x = self.gen_poly_points(self.pwarped.shape[0], lfit)
            _, rfit_x = self.gen_poly_points(self.pwarped.shape[0], rfit)

            #Generate images to draw on
            out_img = np.dstack((sw_display, sw_display, (sw_display)))*255

            #Add color to the left and right lane line pixels
            out_img[lefty, leftx] = [255, 0, 0]  #Make blue
            out_img[righty, rightx] = [0, 0, 255]  #Make red

            #Plot the figures
            figure, (axis1, axis2, axis3) = plt.subplots(3,1) # 3 rows, 1 column
            figure.set_size_inches(10, 10)
            figure.tight_layout(pad=3.0)
            axis1.imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
            axis2.imshow(sw_display, cmap='gray')
            axis3.imshow(out_img)
            axis3.plot(lfit_x, self.ploty, color='yellow')
            axis3.plot(rfit_x, self.ploty, color='yellow')
            axis1.set_title("Original Image")
            axis2.set_title("Warped Fram with Sliding Windows")
            axis3.set_title("Detected Lane Lines with Identified Lane Pixels")
            plt.show()

        return self.lfit, self.rfit

    def refine_lane_polys(self, lfit=None, rfit=None, plot=False):
        """
        Use the polynomial fit from find_lane_line_polys to create a "polynomial window" with the shape of the polynomial.
        Check if pixels are inside this window and use pixels that are to create a new fit

        param lfit: A polynomial function of the left lane line from find_lane_line_polys
        param rfit: A polynomial function of the right lane line from find_lane_line_polys
        param plot: A boolean of whether to plot the result
        return: The updated best fit polynomials for the left and right lane lines
        """
        margin = self.margin

        if lfit == None:
            lfit = self.lfit
        if rfit == None:
            rfit = self.rfit

        #Store the x and y coordinates of all the nonzero pixels
        nonzero = self.pwarped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        #----------Extract the indices (number in the "nonzero" list) of the nonzero pixels that fall within the polynomial window----------#

        if self.l_dtc == True:
            #Calculate the polynomial x value value for each nonzero pixel's y value
            lpolyx = lfit[0]*nonzeroy**2 + lfit[1]*nonzeroy + lfit[2]

            #TODO: Test to see if equivilant to polyx
            _, lpolyx2 = self.gen_poly_points(nonzeroy, lfit)
        
            #Extract the indices that fall within the polynomial window
            left_lane_pix = (nonzerox > lpolyx - margin) & (nonzerox < lpolyx + margin)
            self.llane_inds = left_lane_pix
            
            #Get the xy coordinates of the pixels from the array of indices
            leftx = nonzerox[left_lane_pix]
            lefty = nonzeroy[left_lane_pix]
            
            self.l_x = leftx
            self.l_y = lefty

            #Fit a 2nd order polynomial curve to the pixel coordinates for each lane line
            lfit = np.polyfit(lefty, leftx, 2)
            self.lfit = lfit

        if self.r_dtct == True:
            #Repeat for right lane
            rpolyx = rfit[0]*nonzeroy**2 + rfit[1]*nonzeroy + rfit[2]
            _, rpolyx2 = self.gen_poly_points(nonzeroy, rfit)
            right_lane_pix = (nonzerox > rpolyx - margin) & (nonzerox < rpolyx + margin)
            self.rlane_inds = right_lane_pix
            rightx = nonzerox[right_lane_pix]
            righty = nonzeroy[right_lane_pix]
            self.r_x = rightx
            self.r_y = righty
            rfit = np.polyfit(righty, rightx, 2)
            self.rfit = rfit


        if (plot and self.l_dtct and self.r_dtct) == True:

            #Generate the x and y points for the polynomials
            self.ploty, lfit_x = self.gen_poly_points(self.pwarped.shape[0], lfit)
            _, rfit_x = self.gen_poly_points(self.pwarped.shape[0], rfit)
            self.lfit_x = lfit_x
            self.rfit_x = rfit_x

            #Generate images to draw on
            out_img = np.dstack((self.pwarped, self.pwarped, (self.pwarped)))*255
            polywin_display = np.zeros_like(out_img)

            #Add color to the left and right lane line pixels
            out_img[nonzeroy[left_lane_pix], nonzerox[left_lane_pix]] = [255, 0, 0]  #Make blue
            out_img[nonzeroy[right_lane_pix], nonzerox[right_lane_pix]] = [0, 0, 255]  #Make red

            #Draw the polynomial window
            lwin_lline = np.array([np.transpose(np.vstack([lfit_x - margin, self.ploty]))])
            lwin_rline = np.array([np.transpose(np.vstack([lfit_x + margin, self.ploty]))])
            lwin_pts = np.hstack((lwin_lline, lwin_rline))
            rwin_lline = np.array([np.transpose(np.vstack([rfit_x - margin, self.ploty]))])
            rwin_rline = np.array([np.transpose(np.vstack([rfit_x + margin, self.ploty]))])
            rwin_pts = np.hstack((rwin_lline, rwin_rline))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(polywin_display, np.int_([lwin_pts]), (0,255, 0))
            cv2.fillPoly(polywin_display, np.int_([rwin_pts]), (0,255, 0))
            result = cv2.addWeighted(out_img, 1, polywin_display, 0.3, 0)

            #Plot the figures
            figure, (axis1, axis2, axis3) = plt.subplots(3,1) # 3 rows, 1 column
            figure.set_size_inches(10, 10)
            figure.tight_layout(pad=3.0)
            axis1.imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
            axis2.imshow(self.pwarped, cmap='gray')
            axis3.imshow(result)
            axis3.plot(lfit_x, self.ploty, color='yellow')
            axis3.plot(rfit_x, self.ploty, color='yellow')
            axis1.set_title("Original Image")  
            axis2.set_title("Warped Image")
            axis3.set_title("Warped Image With Polynomial Window")
            plt.show()

        return lfit, rfit

    def plot_lane_zone(self):
        """
        Overlay lane lines on the original frame

        param plot: A boolean of whether to plot the result 
        """

        #Generate an image to draw the lane lines on 
        warp_zero = np.zeros_like(self.pwarped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))       
                
        #Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.lfit_x, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.rfit_x, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))
                
        #Draw lane on the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        #Generate the inverse perspective matrix and warp the image back to original perspective
        self.inv_pwarp_matrix = cv2.getPerspectiveTransform(self.target_roi_corners, self.roi_corners) 
        newwarp = cv2.warpPerspective(color_warp, self.inv_pwarp_matrix, (self.original.shape[1], self.original.shape[0]))
            
        #Combine the result with the original image
        result = cv2.addWeighted(self.original, 1, newwarp, 0.3, 0)
            
        #Plot the figures 
        figure, (axis1, axis2) = plt.subplots(2,1) # 2 rows, 1 column
        figure.set_size_inches(10, 10)
        figure.tight_layout(pad=3.0)
        axis1.imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
        axis2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axis1.set_title("Original Frame")  
        axis2.set_title("Original Frame With Lane Overlay")
        plt.show()

    def get_path(self, lfit=None, rfit=None, plot=False):
        """
        A function to calculate the center of lane path given the polynomial fits for the lane lines

        param lfit:
        param rfit: 
        param plot: A boolean of whether to plot the result
        return: The fit coefficients for the center of lane path
        """

        if lfit == None:
            lfit = self.lfit
        if rfit == None:
            rfit = self.rfit

        #If both lanes are detected, set the path to be the middle of the lane
        if (self.l_dtct and self.r_dtct) == True:
            path_fit = (lfit + rfit) / 2
            self.p_dtct = True
            self.p_iden = 0
        #If just the left lane is detected, set it as the path
        elif (self.l_dtct and not self.r_dtct):
            path_fit = self.l_dtct
            self.p_dtct = True
            self.p_iden = 1
        #If just the right lane is detected, set it as the path
        elif (self.r_dtct and not self.l_dtct):
            path_fit = self.r_dtct
            self.p_dtct = True
            self.p_iden = 2
        #If no lane is detected, inform the higher level program
        else:
            path_fit = None
            print("No path detected")
            self.p_dtct=False
        
        self.path_fit = path_fit

        if plot == True:
            #Generate the path line
            ploty, path_fit_x = self.gen_poly_points(self.pwarped.shape[0], path_fit)
            self.path_fit = path_fit
            self.path_fit_x = path_fit_x

            #Warp the original color image
            color_pwarped = self.perspective_transform(img=self.original)
            self.color_pwarped = color_pwarped
            #Plot the lines
            figure, (axis1) = plt.subplots(1,1) # 3 rows, 1 column
            figure.set_size_inches(10, 10)
            axis1.imshow(cv2.cvtColor(color_pwarped, cv2.COLOR_BGR2RGB))
            axis1.plot(self.lfit_x, self.ploty, color='yellow')
            axis1.plot(self.rfit_x, self.ploty, color='yellow')
            axis1.plot(path_fit_x, ploty, color='red')
            axis1.set_title("Original Image")
            plt.show()
        
        return path_fit

    def gen_cv_state(self, p_dtct=None, path_fit=None, p_iden=None, pwarped=None):
        if p_dtct==None:
            p_dtct = self.p_dtct
        if path_fit==None:
            path_fit = self.path_fit
        if p_iden==None:
            p_iden = self.p_iden
        if pwarped==None:
            pwarped = self.pwarped

        cv_state = (p_dtct, path_fit, p_iden, pwarped)

        self.cv_state = cv_state
        

def pipeline(img):

        #Create a instance of Lane
        lane = Lane(original=img)

        #Filter image to isolate lane line data
        lane.get_filtered(plot=False)
        
        #Perform perspective transform
        lane.perspective_transform(set=True, plot=False)

        #Generate a histogram to to locate starting points for sliding windows
        lane.generate_histogram(plot=False)

        #Get initial polynomials using the sliding windows method
        lfit, rfit = lane.find_laneline_polys(plot=False)

        #Refine polynomials using the polynomial window method
        lfit, rfit = lane.refine_lane_polys(plot=False)

        #Gets the available path
        path_fit = lane.get_path(plot=False)

        #Generates the cv state information
        cv_state = lane.gen_cv_state()

        return cv_state

        

def main():

    #Load the image
    test_image = cv2.imread('testimage.jpeg')

    #Create a instance of Lane
    lane = Lane(original=test_image)

    #Filter image to isolate lane line data
    filtered_lanes = lane.get_filtered(plot=True)
    
    #Perform perspective transform
    pwarped = lane.perspective_transform(set=True, plot=True)

    #Generate a histogram to to locate starting points for sliding windows
    histogram = lane.generate_histogram(plot=True)

    #Get initial polynomials using the sliding windows method
    lfit, rfit = lane.find_laneline_polys(plot=True)

    #Refine polynomials using the polynomial window method
    lfit, rfit = lane.refine_lane_polys(lfit, rfit, plot=True)

    #Show the lane zone
    lane.plot_lane_zone(plot=True)

    #Get the center of lane path
    path_fit = lane.get_path(lfit, rfit, plot=True)

    return path_fit





            


            




    





        


















        





    

        





        






















        






