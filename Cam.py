import cv2 as cv
import numpy as np

# initialize threshold with HSV color range for green colored objects
threshold_lower = (50, 100, 100)
threshold_upper = (70, 255, 255)

# initialize background subtractor
background_subtraction = cv.createBackgroundSubtractorMOG2(1000, 50)

# initialize kalman filter
kalman_filter = cv.KalmanFilter(4, 2)

# Start capturing the video from webcam
video_capture = cv.VideoCapture(0)
roi = None

FONT = cv.FONT_HERSHEY_TRIPLEX

def cam():
    global frame # global frame so it can be used in mouse_get_threshold()
    global roi # global region of interest so it's stored after being select in the first frame
    
    if not video_capture.isOpened():
        video_capture.open(0)
    # Store the current frame of the video in the variable frame
    ret, frame = video_capture.read()
    
    # Flip the image to make it right
    frame = cv.flip(frame,1)
    
    # If region of interest isn't defined prompt user to define one
    if roi is None:
        roi = cv.selectROI('Select Roi', frame, False)
        cv.destroyWindow('Select Roi')
        if roi == ((0,0,0,0)): # if selection is canceled use whole frame
            height, width, _ = frame.shape
            roi = ((0,0,width,height))
        
    # Crop image with roi's dimensions
    frame_roi = frame[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
        
    # Convert the frame to HSV as it allows better segmentation.
    frame_hsv = cv.cvtColor(frame_roi, cv.COLOR_BGR2HSV)
    
    # Blur the frame using a Gaussian Filter of kernel size 5, to remove excessive noise
    frame_blurred = cv.GaussianBlur(frame_hsv, (5,5), 0)

    # Create a mask for the frame, showing threshold values
    frame_segmented = cv.inRange(frame_blurred, threshold_lower, threshold_upper)
    
    # Remove background, isolating moving objects
    frame_movement = background_subtraction.apply(frame_segmented)
    
    # Filter smaller areas out
    frame_thresholded = cv.threshold(src=frame_movement, thresh=20, maxval=255, type=cv.THRESH_BINARY)[1]
    
    # Erode the masked output to delete small white dots present in the masked image
    frame_eroded  = cv.erode(frame_thresholded, None, 10)
    # Dilate the resultant image to restore our target
    frame_masked = cv.dilate(frame_eroded, None, 10)

    # Draw a contour around the detected motion nad return it's center
    center_mass = draw_contours(frame_roi, frame_masked)

    # Display the masked frame in a window located at (x,y) 
    show_output('Masked Output', frame_masked, 0, 550) # / 300, 200

    # Show the output frame in a window located at (x,y) 
    show_output('Camera Output',frame, 0, 0) # / 950, 200 
    cv.setMouseCallback('Camera Output',mouse_get_threshold)

    if center_mass is not None:
        # compensate for roi's different resolution
        paddle_x = int(center_mass[0]) * 610 / roi[2]
        return paddle_x

# correct current position and predict the next, returns the prediction
def predict_movement(coord_x, coord_y):
    kalman_filter.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    kalman_filter.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
    curent = np.array([[np.float32(coord_x)], [np.float32(coord_y)]])
    kalman_filter.correct(curent)
    predicted = kalman_filter.predict()
    return predicted

def draw_contours(frame_draw, frame_contours):
    # draw a blue rectangle where roi is located at
    cv.rectangle(frame_draw, (0, 0), (roi[2]-1, roi[3]-1), (255, 0, 0), 2)
    
    # Filter only clear differences
    _, frame_contours = cv.threshold(frame_contours, 254, 255, cv.THRESH_BINARY)
    
    # Find all contours in the masked image
    contours, _ = cv.findContours(frame_contours.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Define center of the object to be detected and it's predicted center as None
    center_mass_current = None
    center_mass_predicted = None

    # check if there's at least 1 object with the segmented color
    if len(contours) > 0:
        # draw a green rectangle on top of the blue rectangle to indicate that movement was detected
        cv.rectangle(frame_draw, (0, 0), (roi[2]-1, roi[3]-1), (0, 255, 0), 2)
        
        # Find the contour with maximum area
        contours_max = max(contours, key=cv.contourArea)

        # Calculate the centroid of the object
        # "formula from (https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/)"
        M = cv.moments(contours_max)
        center_mass_current = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        #  rotated bounding rectangle (https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html)
        rect = cv.minAreaRect(contours_max)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(frame_draw, [box], 0, (0, 0, 255), 2)
        # draw text to identify the current position
        cv.putText(frame_draw, "Current", (int(center_mass_current[0]), int(center_mass_current[1] + 50)), FONT ,0.6, [0,0,255])
        
        center_mass_predicted = predict_movement(center_mass_current[0],center_mass_current[1])
        
        #  draw the same rotated bounding rectangle but offset to the predicted position
        for point in box:
            point[0] += int(center_mass_predicted[2])
            point[1] += int(center_mass_predicted[3])
        cv.drawContours(frame_draw, [box], 0, (255, 0, 0), 2)
        # draw text to identify the predicted position
        cv.putText(frame_draw, "Predicted", (int(center_mass_predicted[0]), int(center_mass_predicted[1] - 50)), FONT , 0.6, [255, 0, 0])

    return center_mass_predicted

# show window and move it to a specified location so both can be seen simultaneosly
def show_output(title_output, frame_output, window_x, window_y):
    cv.imshow(title_output,frame_output)
    cv.moveWindow(title_output, window_x, window_y)

def mouse_get_threshold(mouse_event,mouse_x,mouse_y,flags,param):
    if mouse_event == cv.EVENT_LBUTTONDOWN: # checks mouse left button down condition
        # convert rgb to hsv format
        color_hsv = cv.cvtColor(np.uint8([[[frame[mouse_y,mouse_x,0] ,frame[mouse_y,mouse_x,1],frame[mouse_y,mouse_x,2] ]]]),cv.COLOR_BGR2HSV)
        
        # create a threshold based on the color values
        temp_lower = color_hsv[0][0][0]  - 10, 100, 100
        temp_upper = color_hsv[0][0][0] + 10, 255, 255
        
        global threshold_lower
        global threshold_upper
        # set the threshold
        threshold_lower = np.array(temp_lower)
        threshold_upper = np.array(temp_upper)
    
def main(): # used for debugging the camera without running the breakout game 
    while(True):
        cam()
        if cv.waitKey(20) & 0xFF == 27: #  wait for Esc key
            break
    video_capture.release()
    cv.destroyAllWindows()

if __name__ == '__main__': # only run main() if executing directly from it's file and not being imported
    main()
    