import cv2 as cv
import numpy as np
import yolov5

# load yolov5model
models = yolov5.load('../yolov5n.pt')
models.conf = 0.33

# Start capturing the video from webcam
video_capture = cv.VideoCapture(0)
roi = None

FONT = cv.FONT_HERSHEY_TRIPLEX


def cam():
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
    frame_rgb = cv.cvtColor(frame_roi, cv.COLOR_BGR2RGB)
    
    # Analyse the frame with the loaded yolov5 models 
    results = models(frame_rgb)

    # Draw a contour around the detected motion nad return it's center
    center = draw_contours(frame_roi, results)

    # Show the output frame in a window located at (x,y) 
    show_output('Camera Output',frame, 0, 0) # / 950, 200 

    if center is not None:
        # compensate for roi's different resolution
        paddle_x = int(center[0]) * 610 / roi[2]
        return paddle_x


def draw_contours(frame_draw, results):
    # draw a blue rectangle where roi is located at
    cv.rectangle(frame_draw, (0, 0), (roi[2]-1, roi[3]-1), (255, 0, 0), 2)

    # Define center of the object to be detected and it's predicted center as None
    center = None

    # check every object to see if it is the selected object
    for pred in enumerate(results.pred):
        im_boxes = pred[1]
        for *box, conf, cls in im_boxes:
            box_class = int(cls)
            if results.names[box_class] == "person":
                # draw a green rectangle on top of the blue rectangle to indicate that selected object as detected
                cv.rectangle(frame_draw, (0, 0), (roi[2]-1, roi[3]-1), (0, 255, 0), 2)
                
                conf = float(conf)
                pt1 = np.array(np.round((float(box[0]), float(box[1]))), dtype=int)
                pt2 = np.array(np.round((float(box[2]), float(box[3]))), dtype=int)
                box_color = (0, 0, 255)
                
                # draw rectangle around the detected object using it's parameters
                cv.rectangle(img = frame_draw,
                                pt1 = pt1,
                                pt2 = pt2,
                                color = box_color,
                                thickness = 1)

                # write the text indenting the object
                cv.putText(img = frame_draw,
                            text = "{}:{:.2f}".format(results.names[box_class], conf),
                            org = np.array(np.round((float(box[0]), float(box[1]-1))), dtype = int),
                            fontFace = FONT,
                            fontScale = 0.5,
                            color = box_color,
                            thickness = 1)
                
                # store the aproximate center of the selected object to return
                center = (int(box[0] + box[2]/2), int(box[1] + box[3]/2))

    return center

# show window and move it to a specified location so both can be seen simultaneosly
def show_output(title_output, frame_output, window_x, window_y):
    cv.imshow(title_output,frame_output)
    cv.moveWindow(title_output, window_x, window_y)
    
def main(): # used for debugging the camera without running the breakout game 
    while(True):
        cam()
        if cv.waitKey(20) & 0xFF == 27: #  wait for Esc key
            break
    video_capture.release()
    cv.destroyAllWindows()

if __name__ == '__main__': # only run main() if executing directly from it's file and not being imported
    main()
    