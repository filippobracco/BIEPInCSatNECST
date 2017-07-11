
# coding: utf-8

# # BIE-PInCS project: python implementation
# ## Firts implementation of video analysis for pupil tracking for identification and evaluation of Traumatic Brain Injury 

# Import required libraries:

# In[ ]:

import cv2
import numpy as np
import math
import time


# ### Initializing phase:

# Parameters for eye cascade function end eye classification

# In[ ]:

sc_fct = 1.3              
min_neigh = 5             
min_eye = (10, 10)
max_eye = (50, 50)
eyes = []
eyeCascade=cv2.CascadeClassifier("./Assets/haarcascade_eye.xml")
dist_TH = 10
res_fac = 4
bound = 10


# Variables containing data about pupils and eyes

# In[ ]:

CENTERS_RIGHTS = [(0,0)]
DIAMS_RIGHTS = [0]
CENTERS_LEFTS = [(0,0)]
DIAMS_LEFTS = [0]
r_check = []
l_check = []


# Parameters about source file to analyze

# In[ ]:

video_file = "./Assets/light_stimulation.mp4"
#video_file = "./Assets/dinamic_stimulation.mp4"
video_capture = cv2.VideoCapture(video_file)
video_width = int(video_capture.get(3)) 
video_height = int(video_capture.get(4))
frps = int(video_capture.get(5))
frames = int(video_capture.get(7))
Tc = 1/frps


# Parameters for image analysis

# In[ ]:

pup_TH = 30
windowClose = np.ones((5,5),np.uint8)
windowOpen = np.ones((2,2),np.uint8)
windowErode = np.ones((2,2),np.uint8)


# Backgroung image

# In[ ]:

bground = cv2.imread('./Assets/Background.png')
bgH,bgW, _ = bground.shape


# Video output

# In[ ]:

fourcc = cv2.VideoWriter_fourcc(*'jpeg')
out = cv2.VideoWriter('output.mp4',fourcc, frps, (bgW, bgH), True)


# Conversion factor from pixel to mm and unit of measure

# In[ ]:

K = 160/268 #from pixel to mm
uom = 'mm'


# Factors to write on the image output text and other images

# In[ ]:

h1 = int(bgH/2 + 20)
h_line = h1 + 30
h2 = h1 + 80
h3 = h2 + 30
h4 = h3 + 30
h5 = h4 + 50
center = int(bgW/2)


# ### Functions:

# The function puts an image on other, starting from a point (i, j)

# In[ ]:

def draw(img1, img2, i, j):          # i = vertical, j = horizontal
    h,w, _ = img2.shape
    img1[i:i+h, j:j+w] = img2[:h, :w]


# The function makes an image totally black

# In[ ]:

def rubber(img):
    h,w, _ = img.shape
    img[:h, :w] = (0,0,0)


# The function writes, on a predetermined point, factors taken in input: value of scale, diameters and time on output image.

# In[ ]:

def write_img(image, count, diam_r, diam_l, scl):
    cv2.putText(image, text='SCALE: '+str(scl)+uom, org=(center-70, h1), fontFace=2, fontScale=0.6, color=(255,255,255), thickness=1)
    cv2.putText(image, text='DIAMETERS:', org=(center-70, h2), fontFace=2, fontScale=0.6, color=(255,255,255), thickness=1)
    cv2.putText(image, text='right: '+str(round(diam_r,1))+uom, org=(center-70, h3), fontFace=2, fontScale=0.6, color=(255,255,255), thickness=1)
    cv2.putText(image, text='left: '+str(round(diam_l,1))+uom, org=(center-70, h4), fontFace=2, fontScale=0.6, color=(255,255,255), thickness=1)
    t = count*Tc*1000
    cv2.putText(image, text='time: '+str(int(t))+'ms', org=(center-70, h5), fontFace=2, fontScale=0.6, color=(255,255,255), thickness=1)
    # comparing line 
    shift = int((0.5*scl*res_fac)/K)
    cv2.line(image, (center-shift,h_line), (center+shift,h_line), color=(255,255,255), thickness=2) 
    cv2.line(image, (center-shift,h_line+10), (center-shift,h_line-10), color=(255,255,255), thickness=2)
    cv2.line(image, (center+shift,h_line+10), (center+shift,h_line-10), color=(255,255,255), thickness=2)


# The function classifies eyes in rights and lefts; classification is based on position of ROI within the image

# In[ ]:

def check_eyes(eyes, rx0, ry0, lx0, ly0):
    r = []
    l = []
    for (x,y,w,h) in eyes:
        if x > video_width*0.5 and (abs(x-rx0) < dist_TH and abs(y-ry0) < dist_TH):
            r.append((x,y,w,h))
        elif x < video_width*0.5 and (abs(x-lx0) < dist_TH and abs(y-ly0) < dist_TH):
            l.append((x,y,w,h))
    return r,l


# The function return values of the correct eye among a vector; it choises the closest eye from an established point of previous eye position

# In[ ]:

def correct_eye(eye_list, point):    
    eyes = []
    min_dist = video_width
    
    for (x,y,w,h) in eye_list:            
        distance = abs(x-point)
        if (distance < min_dist):
            ex = x
            ey = y
            ew = w
            eh = h
            min_dist = distance
            
    return ex,ey,ew,eh


# The function takes in input a ROI containing eye and return, if it is possible, center position and diameter of finded pupil. 

# In[ ]:

def findpupil(roi, x, y, w):
    max_rad = w/3
    min_rad = w/10
    
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    pupilFrame = cv2.equalizeHist(roi)
    pupilO = pupilFrame
    ret, pupilFrame = cv2.threshold(pupilFrame,pup_TH,255,cv2.THRESH_BINARY)
    pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_CLOSE, windowClose)
    pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_ERODE, windowErode)
    pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_OPEN, windowOpen)
    
    threshold = cv2.inRange(pupilFrame,250,255)
    _, contours, hierarchy = cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) >= 2:
        maxArea = 0
        MAindex = 0
        distanceX = []
        currentIndex = 0 
        for cnt in contours:
            area = cv2.contourArea(cnt)
            center = cv2.moments(cnt)
            if center['m00'] != 0:
                cx,cy = int(center['m10']/center['m00']), int(center['m01']/center['m00'])
                distanceX.append(cx)
                if area > maxArea:
                    maxArea = area
                    MAindex = currentIndex
                currentIndex += 1
        
        del contours[MAindex]
        del distanceX[MAindex]

        eye = 'right'

        if len(contours) >= 2:
            if eye == 'right':
                edgeOfEye = distanceX.index(min(distanceX))
            else:
                edgeOfEye = distanceX.index(max(distanceX))
            del contours[edgeOfEye]
            del distanceX[edgeOfEye]

        if len(contours) >= 1:
            maxArea = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > maxArea:
                    maxArea = area
                    largeBlob = cnt

        if len(largeBlob) > 0:
            center = cv2.moments(largeBlob)
            if center['m00'] != 0:
                cx,cy = int(center['m10']/center['m00']), int(center['m01']/center['m00'])
                rad = (center['m00'] / math.pi)**0.5
                if rad < max_rad and rad > min_rad:
                    return ((int(cx+x), int(cy+y)), rad)
            else:
                return None, None
        
    return None, None


# ### Main code of kernel:

# At first algorithm search the first frame containing exactly two eyes, and then classifies them in left and right, assigning the values of initial eye position.

# In[ ]:

while len(eyes) != 2:
    ret, frame = video_capture.read()
    eyes=eyeCascade.detectMultiScale(frame, scaleFactor=sc_fct, minNeighbors=min_neigh, minSize=min_eye, maxSize=max_eye)
    
if eyes[0][0] > eyes[1][0]:
    
    x0r = eyes [0][0]
    y0r = eyes [0][1]
    w0r = eyes [0][2]
    h0r = eyes [0][3]
    
    x0l = eyes [1][0]
    y0l = eyes [1][1]
    w0l = eyes [1][2]
    h0l = eyes [1][3]
    
else:
    x0l = eyes [0][0]
    y0l = eyes [0][1]
    w0l = eyes [0][2]
    h0l = eyes [0][3]
    
    x0r = eyes [1][0]
    y0r = eyes [1][1]
    w0r = eyes [1][2]
    h0r = eyes [1][3]
  


# Scale factor is calculatet and initialized counter and re-loaded the file video to allow to analyze from first frame. 

# In[ ]:

scale_f = int(0.25*(w0r+w0l)*K)

video_capture = cv2.VideoCapture(video_file)
count = 0 


# This is the foundamental core of the algorithm. While video file contains frames, each image is loaded, analized and the output video is created, adding output frame for each iteration. Finally video writer is release.

# In[ ]:

while(video_capture.isOpened()):
    ret, frame = video_capture.read()
    
    if ret:
        eyes=eyeCascade.detectMultiScale(frame, scaleFactor=sc_fct, minNeighbors=min_neigh, minSize=min_eye, maxSize=max_eye)

        rights, lefts = check_eyes(eyes, x0r, y0r, x0l, y0l)

        if len(rights) != 0:                             #right eye
            xr, yr, wr, hr = correct_eye(rights, x0r)
            roi_right = frame[yr:yr+hr , xr:xr+wr]

            center_r, radius_r = findpupil(roi_right, xr, yr, wr)
            if radius_r:
                CENTERS_RIGHTS.append(center_r)
                DIAMS_RIGHTS.append(2*K*radius_r)
                r_check.append(True)

            else:
                CENTERS_RIGHTS.append(CENTERS_RIGHTS[count-1])
                DIAMS_RIGHTS.append(DIAMS_RIGHTS[count-1])
                r_check.append(False)
        else:
            CENTERS_RIGHTS.append(CENTERS_RIGHTS[count-1])
            DIAMS_RIGHTS.append(DIAMS_RIGHTS[count-1])
            r_check.append(False)

        if len(lefts) != 0:                              #left eye
            xl, yl, wl, hl = correct_eye(lefts, x0l)
            roi_left = frame[yl:yl+hl , xl:xl+wl]

            center_l, radius_l = findpupil(roi_left, xl, yl, wl)
            if radius_l:
                CENTERS_LEFTS.append(center_l)
                DIAMS_LEFTS.append(2*K*radius_l)
                l_check.append(True)

            else:
                CENTERS_LEFTS.append(CENTERS_LEFTS[count-1])
                DIAMS_LEFTS.append(DIAMS_LEFTS[count-1])
                l_check.append(False)
        else:
            CENTERS_LEFTS.append(CENTERS_LEFTS[count-1])
            DIAMS_LEFTS.append(DIAMS_LEFTS[count-1])
            l_check.append(False)
            
        # output management
        rubber(bground)
        
        roi_r = frame[y0r-bound: y0r+h0r+bound , x0r-bound:x0r+w0r+bound]
        roi_l = frame[y0l-bound:y0l+h0l+bound , x0l-bound:x0l+w0l+bound]
        res_right = cv2.resize(roi_r,None, fx=res_fac, fy=res_fac, interpolation = cv2.INTER_LINEAR) #resized eyes
        res_left = cv2.resize(roi_l, None,fx=res_fac, fy=res_fac, interpolation = cv2.INTER_LINEAR)
        
        cv2.circle(frame,CENTERS_RIGHTS[count],int(0.5*DIAMS_RIGHTS[count]/K),(0,0,255),1)
        cv2.circle(frame,CENTERS_LEFTS[count],int(0.5*DIAMS_LEFTS[count]/K),(0,0,255),1)
        
        pos_r = int(bgW - res_fac*(w0r + 2*bound))                 #drawing resized eyes
        pos_vl = int(bgH - res_fac*(h0l + 2*bound))
        pos_vr = int(bgH - res_fac*(h0r + 2*bound))
        draw(bground, frame, 0, 0)
        draw(bground, res_left, pos_vl, 0)
        draw(bground, res_right, pos_vr, pos_r)
        
        write_img(bground, count, DIAMS_RIGHTS[count], DIAMS_LEFTS[count], scale_f)
        
        count += 1
        if not(count%60):                   #update eye positions
            x0r = xr
            y0r = yr

            x0l = xl
            y0l = yl   
        out.write(bground)
    else:
        break
        
del DIAMS_LEFTS[0]
del DIAMS_RIGHTS[0]

video_capture.release()
out.release()

