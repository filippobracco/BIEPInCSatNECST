# # Biepincs application
# Acceleration of OpenCV application integrating an hardware accelerator loaded on FPGA. Application analizes a video file containing a stimulation test of a injured patient, taken with an infrared camera. Kinematic and geometric parameters of pupils are measured and saved. 

# ### Overlay as to be iported and loaded on device:
from pynq import Overlay
ol = Overlay("biepincs2.bit")
ol.download()
ol.bitstream.timestamp


# ### Import libraries to be used:
import cv2
import numpy as np
import math
import threading
from pynq.drivers import DMA
import time
from cffi import FFI
ffi = FFI()

# ### Constant definition:
# W and halfWidth are the dimension and the half-dimension of ROI image, CH is the number of channel of imageand TH is the value applied within the threshold filter.
W = 50
halfWidth = 0.5*W
CH = 3
TH = 17

# ### DMA inizialization: 
# Two differents DMAs are instantiated at the same address, one for reading and one for writing
dmaOut = DMA(0x40400000,1)
dmaIn = DMA(0x40400000,0)

# Two buffers are created containing data to be tranfered to PL (dmaIn) and output results (dmaOut)
dmaIn.create_buf(W*W*CH+1)
dmaOut.create_buf(W*W)

# Pointers to buffers are taken in order to allow to write and read data. Another buffer is allocated (at the same position of the DMA's out one) to allow to read data using numpy.frombuffer function.
# First value of input buffer is set with threshold value.
pointIn = ffi.cast("uint8_t *", dmaIn.get_buf())
pointOut = ffi.cast("uint8_t *", dmaOut.get_buf())
c_buffer = ffi.buffer(pointOut, W*W)
pointIn[0] = TH


# ### Setting parameters for video analysis :
# Path to source folder.
source_folder = "./Assets/"

# Parameters for Cascadeclassifier function.
sc_fct = 1.3              
min_neigh = 3            
min_eye = (40, 40)
max_eye = (60, 60)
eyes = []
eyeCascade=cv2.CascadeClassifier(source_folder + "haarcascade_eye.xml")

# Inizialize lists containing values calculated.
CENTERS_RIGHTS = [(0,0)]
DIAMS_RIGHTS = [0]
CENTERS_LEFTS = [(0,0)]
DIAMS_LEFTS = [0]
r_check = []
l_check = []

# Loading video file to be analized and extract main features on dimension and time.
video_file = source_folder + "Stimolazione_dinamica.mp4"
video_capture = cv2.VideoCapture(video_file)
video_width = int(video_capture.get(3)) 
video_height = int(video_capture.get(4))
frps = int(video_capture.get(5))
frames = int(video_capture.get(7))
Tc = 1/frps

# MAX_DISTANCE is the maximum allowed variation of distance (pixel) between centers of eyes, used as a threshold to classify if a pupil is lost. 
# MAX_PUPIL_LOST is the maximum consecutive frames without a pupil, befor calling tha Cascadeclassifier.
# MIN_RAD is the minimum allowed value of pupil radius.
# Other variables halp to manage calls of Cascadeclassifier function.
MAX_DISTANCE = 40
MAX_PUPIL_LOST = 10
MIN_RAD = W/20
rightEyeLost = False
leftEyeLost = False
LeftPupilCounter = 0
RightPupilCounter = 0

# Defining a conversion factor between a space Unit of Measure (uom) and pixel. It dependes on video acquisition setup.
K = 0.24 #from pixel to mm
uom = 'mm'


# ### Defining functions:
# This function takes as input a list of eyes and returns the biggest one.
def max_area_eye(eye_list):
    eye = []
    MArea = 0;
    for ey in eye_list:
        if ey[2]*ey[3] > MArea:
            eye = ey
            
    return eye

# Function calculates countours of an image, selects the biggest and approximates it as a circle. Output parameters are center, as a list of integer x and y values, and radius. If there are no contours, function returns None as center and -1 value for radius.
def findpupil(roi):
    _, contours,_ = cv2.findContours(roi,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    lengthCnt = len(contours) 
    
    if lengthCnt == 0:
        return None, -1
    
    if lengthCnt >=2:
        maxArea = 0;
        MAindex = 0;
        cIndex = 0;
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < maxArea:
                maxArea = area
                MAindex = cIndex
            cIndex = cIndex+1
        (cen, rad) = cv2.minEnclosingCircle(contours[MAindex]);
        cx, cy = cen;
        return [int(cx), int(cy)], rad
    else:
        (cen, rad) = cv2.minEnclosingCircle(contours[0]);
        cx, cy = cen;
        return [int(cx), int(cy)], rad

    return None, -1

# Function loads images from video file as a buffer of two frames. It is executed with a thread.
def loadFrames(eFrame, eMain):
    tm = []
    global frame1
    global frame2
    global ret
    while ret:
        ret, frame1 = video_capture.read()
        eFrame.set()
        eMain.wait()
        eMain.clear()
        
        ret, frame2 = video_capture.read()
        eFrame.set()
        eMain.wait()
        eMain.clear()



# ## First part of application:
# This part is non iterative. Finds the firsts two eyes using Cascadeclassifier function and saves their positions.
while len(eyes) != 2:
    ret, frame = video_capture.read()
    eyes=eyeCascade.detectMultiScale(frame, scaleFactor=sc_fct, minNeighbors=min_neigh, minSize=min_eye, maxSize=max_eye)
    
if eyes[0][0] > eyes[1][0]:
    right_pup_x = eyes [0][0]
    right_pup_y = eyes [0][1]
    
    left_pup_x = eyes [1][0]
    left_pup_y = eyes [1][1]
else:
    left_pup_x = eyes [0][0]
    left_pup_y = eyes [0][1]
    
    right_pup_x = eyes [1][0]
    right_pup_y = eyes [1][1]

radius_l = 0;
radius_r = 0;
distancePups = np.sqrt((left_pup_x - right_pup_x)**2 + (left_pup_y - right_pup_y)**2)
video_capture = cv2.VideoCapture(video_file)
count = 0 


# ## Main part of application 
# At the beginning a thread that is in charge of loading all frames, one by one, is created and started. 
# In the iterative part, if pupil is lost in the previous frame eye is searched. Then iterative part of analysis is started.

video_capture = cv2.VideoCapture(video_file)
ret = True
eFrame = threading.Event()
eMain = threading.Event()
frame1 = True
frame2 = True
thread = threading.Thread(target=loadFrames, args=(eFrame, eMain))
thread.start()

while(True):
    eFrame.wait()
    if ret:
        #search right eye if lost
        if (RightPupilCounter > MAX_PUPIL_LOST or rightEyeLost):
            eyes=eyeCascade.detectMultiScale(frame1[:, int(0.5*video_width):video_width], scaleFactor=sc_fct, minNeighbors=min_neigh, minSize=min_eye, maxSize=max_eye)
            if len(eyes) != 0: 
                r = max_area_eye(eyes)
                r[0] = r[0] + int(0.5*video_width);
                if r[2] > 0:
                    rightEyeLost = False
                    right_pup_x = r[0]
                    right_pup_y = r[1]
        #search left eye if lost
        if (LeftPupilCounter > MAX_PUPIL_LOST or leftEyeLost):
            eyes=eyeCascade.detectMultiScale(frame1[:, 0:int(0.5*video_width)], scaleFactor=sc_fct, minNeighbors=min_neigh, minSize=min_eye, maxSize=max_eye)
            if len(eyes) != 0: 
                l = max_area_eye(eyes)
                if l[2] > 0:
                    leftEyeLost = False
                    left_pup_x = l[0]
                    left_pup_y = l[1]
    
###____________________STARTING MAIN COMPUTATIONAL CHAIN FOR FIRST FRAME____________________
    
        #roi copy of the rigth eye
        roiRightEye = frame1[int(right_pup_y): int(right_pup_y+W), int(right_pup_x):int(right_pup_x+W)].copy()
        #pointer to image (roi)
        pointerToRoiRightEye = ffi.cast("uint8_t *", ffi.from_buffer(roiRightEye))
        #transfer image data to buffer
        ffi.memmove(pointIn+1, pointerToRoiRightEye, W*W*CH) 
        
        #DMA tranfer
        dmaOut.transfer(W*W, 1)
        dmaIn.transfer(W*W*CH, 0)
        
        #roi copy of left eye
        roiLeftEye = frame1[int(left_pup_y): int(left_pup_y+W), int(left_pup_x):int(left_pup_x+W)].copy()
        #pointer to image (roi)
        pointerToRoiLeftEye = ffi.cast("uint8_t *", ffi.from_buffer(roiLeftEye))
        #transfer image data to buffer
        ffi.memmove(pointIn+1, pointerToRoiLeftEye, W*W*CH) 
        
        #get image analysed from buffer
        resultRight = np.frombuffer(c_buffer, dtype=np.uint8)
        resultRight = resultRight.reshape(W,W)
        threshRight = resultRight.copy()
        
        #DMA tranfer
        dmaOut.transfer(W*W, 1)
        dmaIn.transfer(W*W*CH, 0)
    
        #get center and radius
        center_r, radius_r = findpupil(threshRight)
        center_r[0] = right_pup_x;
        center_r[1] = rigth_pup_y;
    
        #get image analysed from buffer
        resultLeft = np.frombuffer(c_buffer, dtype=np.uint8)
        resultLeft = resultLeft.reshape(W,W)
        threshLeft = resultLeft.copy()
        
        #get center and radius
        center_l, radius_l = findpupil(threshLeft)
        center_l[0] = left_pup_x;
        center_l[1] = left_pup_y;
        
        if radius_r > MIN_RAD:
            if RightPupilCounter != 0:
                RightPupilCounter = 0
            #update x and y position of ROI centered on the new pupil center
            right_pup_x = center_r[0] - halfWidth + right_pup_x
            right_pup_y = center_r[1] - halfWidth + right_pup_y
            #store new center and diameter values
            CENTERS_RIGHTS.append(center_r)
            DIAMS_RIGHTS.append(2*K*radius_r)
            r_check.append(True)
        elif count != 0: #if pupil is not found, we store the previous valid value
            RightPupilCounter = RightPupilCounter +1
            CENTERS_RIGHTS.append(CENTERS_RIGHTS[count-1])
            DIAMS_RIGHTS.append(DIAMS_RIGHTS[count-1])
            r_check.append(False)

        if radius_l > MIN_RAD:
            if LeftPupilCounter != 0:
                LeftPupilCounter = 0 
            #update x and y position of ROI centered on the new pupil center
            left_pup_x = center_l[0] - halfWidth + left_pup_x
            left_pup_y = center_l[1] - halfWidth + left_pup_y
            #store new center and diameter values
            CENTERS_LEFTS.append(center_l)
            DIAMS_LEFTS.append(2*K*radius_l)
            l_check.append(True)
        elif count != 0: #if pupil is not found, we store the previous valid value
            LeftPupilCounter = LeftPupilCounter +1
            CENTERS_LEFTS.append(CENTERS_LEFTS[count-1])
            DIAMS_LEFTS.append(DIAMS_LEFTS[count-1])
            l_check.append(False)

        #Calculate the distance between centers of eyes
        distance = np.sqrt((left_pup_x - right_pup_x)**2 + (left_pup_y - right_pup_y)**2) 
        
        #apply a classification based on the difference of distance between current value and a weighted value of previous frames right eye
        if (abs(distance - distancePups) > MAX_DISTANCE and count !=0):
            dist_lefts = np.sqrt((CENTERS_LEFTS[count-1][0] - CENTERS_LEFTS[count-2][0])**2 + (CENTERS_LEFTS[count-1][1] - CENTERS_LEFTS[count-2][1])**2)
            dist_rights = np.sqrt((CENTERS_RIGHTS[count-1][0] - CENTERS_RIGHTS[count-2][0])**2 + (CENTERS_RIGHTS[count-1][1] - CENTERS_RIGHTS[count-2][1])**2)
            if dist_lefts > dist_rights:
                leftEyeLost = True; 
                DIAMS_LEFTS[count] = DIAMS_LEFTS[count-1];
                CENTERS_LEFTS[count] = CENTERS_LEFTS[count-1];
            else:
                rightEyeLost = True;
                DIAMS_RIGHTS[count] = DIAMS_RIGHTS[count-1];
                CENTERS_RIGHTS[count] = CENTERS_RIGHTS[count-1]; 
        
        #apply a classification based on the difference of distance between current value and a weighted value of previous frames for left eye
        if (abs(distance - distancePups) > MAX_DISTANCE and count !=0):
            dist_lefts = np.sqrt((CENTERS_LEFTS[count-1][0] - CENTERS_LEFTS[count-2][0])**2 + (CENTERS_LEFTS[count-1][1] - CENTERS_LEFTS[count-2][1])**2)
            dist_rights = np.sqrt((CENTERS_RIGHTS[count-1][0] - CENTERS_RIGHTS[count-2][0])**2 + (CENTERS_RIGHTS[count-1][1] - CENTERS_RIGHTS[count-2][1])**2)
            if dist_lefts > dist_rights:
                leftEyeLost = True;
                l_check[count-1] = False;
                DIAMS_LEFTS[count] = DIAMS_LEFTS[count-1];
                CENTERS_LEFTS[count] = CENTERS_LEFTS[count-1];
            else:
                rightEyeLost = True;
                r_check[count-1] = False;
                DIAMS_RIGHTS[count] = DIAMS_RIGHTS[count-1];
                CENTERS_RIGHTS[count] = CENTERS_RIGHTS[count-1];
        
        #update the weighted mean of distance        
        distancePups = (count*distancePups + distance)/(count+1);
        
        eFrame.clear()
        eMain.set()
        eFrame.wait()
    else:
        break
    count += 1    
    
    #we applied the same procedure to the second frame loaded of the buffer
    
    if ret:
        #search right eye if lost
        if (RightPupilCounter > MAX_PUPIL_LOST or rightEyeLost):
            eyes=eyeCascade.detectMultiScale(frame2[:, int(0.5*video_width):video_width], scaleFactor=sc_fct, minNeighbors=min_neigh, minSize=min_eye, maxSize=max_eye)
            if len(eyes) != 0: 
                r = max_area_eye(eyes)
                r[0] = r[0] + int(0.5*video_width);
                if r[2] > 0:
                    rightEyeLost = False
                    right_pup_x = r[0]
                    right_pup_y = r[1]
        
        #search left eye if lost
        if (LeftPupilCounter > MAX_PUPIL_LOST or leftEyeLost):
            eyes=eyeCascade.detectMultiScale(frame2[:, 0:int(0.5*video_width)], scaleFactor=sc_fct, minNeighbors=min_neigh, minSize=min_eye, maxSize=max_eye)
            if len(eyes) != 0: 
                l = max_area_eye(eyes)
                if l[2] > 0:
                    leftEyeLost = False
                    left_pup_x = l[0]
                    left_pup_y = l[1]
    
###____________________STARTING MAIN COMPUTATIONAL CHAIN FOR SECOND FRAME____________________
        
        #roi copy of the rigth eye
        roiRightEye = frame2[int(right_pup_y): int(right_pup_y+W), int(right_pup_x):int(right_pup_x+W)].copy()
        #pointer to image (roi)
        pointerToRoiRightEye = ffi.cast("uint8_t *", ffi.from_buffer(roiRightEye))
        #transfer image data to buffer
        ffi.memmove(pointIn+1, pointerToRoiRightEye, W*W*CH) 

        #dma tranfer
        dmaOut.transfer(W*W, 1)
        dmaIn.transfer(W*W*CH, 0)
        
        #roi copy of left eye
        roiLeftEye = frame2[int(left_pup_y): int(left_pup_y+W), int(left_pup_x):int(left_pup_x+W)].copy()
        #pointer to image (roi)
        pointerToRoiLeftEye = ffi.cast("uint8_t *", ffi.from_buffer(roiLeftEye))
        #transfer image data to buffer
        ffi.memmove(pointIn+1, pointerToRoiLeftEye, W*W*CH) 
        
        #get image analysed from buffer
        resultRight = np.frombuffer(c_buffer, dtype=np.uint8)
        resultRight = resultRight.reshape(W,W)
        threshRight = resultRight.copy()
        
         #dma tranfer
        dmaOut.transfer(W*W, 1)
        dmaIn.transfer(W*W*CH, 0)
        
        #get center and radius
        center_r, radius_r = findpupil(threshRight)
        center_r[0] = right_pup_x;
        center_r[1] = rigth_pup_y;
        
        #get image analysed from buffer
        resultLeft = np.frombuffer(c_buffer, dtype=np.uint8)
        resultLeft = resultLeft.reshape(W,W)
        threshLeft = resultLeft.copy()
        
        #get center and radius
        center_l, radius_l = findpupil(threshLeft)
        center_l[0] = left_pup_x;
        center_l[1] = left_pup_y;
        
        if radius_r > MIN_RAD:
            if RightPupilCounter != 0:
                RightPupilCounter = 0 
            #update x and y position of ROI centered on the new pupil center
            right_pup_x = center_r[0] - halfWidth + right_pup_x
            right_pup_y = center_r[1] - halfWidth + right_pup_y
            #store new center and diameter values
            CENTERS_RIGHTS.append(center_r)
            DIAMS_RIGHTS.append(2*K*radius_r)
            r_check.append(True)
        elif count != 0: #if pupil is not found, we store the previous valid value
            RightPupilCounter = RightPupilCounter +1
            CENTERS_RIGHTS.append(CENTERS_RIGHTS[count-1])
            DIAMS_RIGHTS.append(DIAMS_RIGHTS[count-1])
            r_check.append(False)
            
        if radius_l > MIN_RAD:
            if LeftPupilCounter != 0:
                LeftPupilCounter = 0 
            #update x and y position of ROI centered on the new pupil center
            left_pup_x = center_l[0] - halfWidth + left_pup_x
            left_pup_y = center_l[1] - halfWidth + left_pup_y
            #store new center and diameter values
            CENTERS_LEFTS.append(center_l)
            DIAMS_LEFTS.append(2*K*radius_l)
            l_check.append(True)
        elif count != 0: #if pupil is not found, we store the previous valid value
            LeftPupilCounter = LeftPupilCounter +1
            CENTERS_LEFTS.append(CENTERS_LEFTS[count-1])
            DIAMS_LEFTS.append(DIAMS_LEFTS[count-1])
            l_check.append(False)
            
        #Calculate the distance between centers of eyes    
        distance = np.sqrt((left_pup_x - right_pup_x)**2 + (left_pup_y - right_pup_y)**2) 
        
        #apply a classification based on the difference of distance between current value and a weighted value of previous frames right eye
        if (abs(distance - distancePups) > MAX_DISTANCE and count !=0):
            dist_lefts = np.sqrt((CENTERS_LEFTS[count-1][0] - CENTERS_LEFTS[count-2][0])**2 + (CENTERS_LEFTS[count-1][1] - CENTERS_LEFTS[count-2][1])**2)
            dist_rights = np.sqrt((CENTERS_RIGHTS[count-1][0] - CENTERS_RIGHTS[count-2][0])**2 + (CENTERS_RIGHTS[count-1][1] - CENTERS_RIGHTS[count-2][1])**2)
            if dist_lefts > dist_rights:
                leftEyeLost = True;
                DIAMS_LEFTS[count] = DIAMS_LEFTS[count-1];
                CENTERS_LEFTS[count] = CENTERS_LEFTS[count-1];
            else:
                rightEyeLost = True;
                DIAMS_RIGHTS[count] = DIAMS_RIGHTS[count-1];
                CENTERS_RIGHTS[count] = CENTERS_RIGHTS[count-1]; 
        
        #apply a classification based on the difference of distance between current value and a weighted value of previous frames left eye
        if (abs(distance - distancePups) > MAX_DISTANCE and count !=0):
            dist_lefts = np.sqrt((CENTERS_LEFTS[count-1][0] - CENTERS_LEFTS[count-2][0])**2 + (CENTERS_LEFTS[count-1][1] - CENTERS_LEFTS[count-2][1])**2)
            dist_rights = np.sqrt((CENTERS_RIGHTS[count-1][0] - CENTERS_RIGHTS[count-2][0])**2 + (CENTERS_RIGHTS[count-1][1] - CENTERS_RIGHTS[count-2][1])**2)
            if dist_lefts > dist_rights:
                leftEyeLost = True;
                DIAMS_LEFTS[count] = DIAMS_LEFTS[count-1];
                CENTERS_LEFTS[count] = CENTERS_LEFTS[count-1];
            else:
                rightEyeLost = True;
                DIAMS_RIGHTS[count] = DIAMS_RIGHTS[count-1];
                CENTERS_RIGHTS[count] = CENTERS_RIGHTS[count-1];
        #update the weighted mean of distance          
        distancePups = (count*distancePups + distance)/(count+1);
        
        eFrame.clear()
        eMain.set()
        eFrame.wait()
    else:
        break

    count += 1
        
del DIAMS_LEFTS[0]
del DIAMS_RIGHTS[0]

# ### Store values in a file:
# File containing values computed is saved.

outputFile = open('Output.txt','w')
for i in range(count):
    outputFile.write(str(CENTERS_RIGHTS[i][0])+' '+str(CENTERS_RIGHTS[i][1])+' '+str(DIAMS_RIGHTS[i])+' '+str(CENTERS_LEFTS[i][0])+' '+str(CENTERS_LEFTS[i][1])+' '+str(DIAMS_LEFTS[i])+'\n')
outputFile.close()

