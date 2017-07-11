#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <math.h>

#define FRM 950
#define W 50
#define MAX_DISTANCE 10
#define MAX_PUPIL_LOST 5

using namespace std;
using namespace cv;

// Source folder
String source_folder = "<Path>/Assets/";
String output_folder = "<Path>/Output/";

// eye cascade parameters
double sc_fct = 1.3;
int min_neigh = 3;
Size min_eye = Size(20, 20);
Size max_eye = Size(60, 60);
String eyes_cascade_name = source_folder + "haarcascade_eye.xml";
CascadeClassifier eyes_cascade;
int dist_TH = 30;
bool rightEyeLost = false;
bool leftEyeLost = false;

// eyes
vector<Rect> eyes;

// Background image (base of output)
String bg_file = source_folder + "Background.png";
Mat bground = imread(bg_file);
int bgH = bground.size[0];
int bgW = bground.size[1];

// suorce file to analize:
String video_file = source_folder + "stimulation_video.mp4";
VideoCapture cap(video_file);
int video_width = cap.get(3);
int video_height = cap.get(4);
int frps = cap.get(5);
int frames = cap.get(7);

//video output
int codec = CV_FOURCC('M', 'J', 'P', 'G');
string filename = output_folder + "output.avi";
VideoWriter out(filename, codec, frps, Size(bgW, bgH), true);

// resize eyes
int res_fac = 4;
int bound = 10;

// conversion factors
float K = 0.24; //from pixel to mm
int scale_f;
String uom = "mm";

// pupil function
double pupTH = 15;
Mat window_3 = Mat::ones(3, 3, CV_8U);
float radius;
Point2f centerRightPup, centerLeftPup;
float DIAMS_RIGHTS[FRM], DIAMS_LEFTS[FRM];
Point CENTERS_RIGHTS[FRM], CENTERS_LEFTS[FRM];
bool r_ch[FRM], l_ch[FRM];
u_int LeftPupilCounter=0, RightPupilCounter=0;
float distancePups;

//writing factors (points in background image where insert text)
int h1 = bgH/2 + 20;
int h_line = h1 + 30;
int h2 = h1 + 80;
int h3 = h2 + 30;
int h4 = h3 + 30;
int h5 = h4 + 50;
int bg_center = bgW/2;

Mat frame;
Rect r;
Rect l;
Rect roi_right;
Rect roi_left;
Mat res_right;
Mat res_left;

//Position of pupils
Rect right_pup;
Rect left_pup;
vector<u_int> errorRightEye, errorLeftEye;

void draw(Mat img1, Mat img2, int i, int j); //write an image over another
void rubber(Mat image); //rubber function on image
void findpupil(Mat roi, int w, Point2f& center, float& radius); //find pupil within ROI eye
void write(Mat image, int count, float diam_r, float diam_l, int scl); //write text on output image
float dst(Point p1, Point p2);
Rect maxEye(vector<Rect>& eyesVec);

int main(int argc, char** argv)
{
    while(eyes.size() != 2) {
        Mat frame;
        cap >> frame;
        eyes_cascade.detectMultiScale(frame, eyes, sc_fct, min_neigh, 0, min_eye, max_eye);
    }
    if (eyes[0].x > eyes[1].x) {
        r = eyes[0];
        l = eyes[1];
    }else{
        r = eyes[1];
        l = eyes[0];
    }
    scale_f = 0.5*(r.width+l.width)*K;
    int count=0;
    VideoCapture cap(video_file);
    
    //initial ROI eye within search pupils
    right_pup.x = r.x;
    right_pup.y = r.y;
    right_pup.width = W;
    right_pup.height = W;
    
    left_pup.x = l.x;
    left_pup.y = l.y;
    left_pup.width = W;
    left_pup.height = W;
    
    
    
    //____________________________ITERATIVE PART____________________________
    //part in charge of analysis of stimulation video and output generation
    while(cap.isOpened()) {
        
        cap >> frame;
        if( frame.empty() ) break; // end of video stream
        
        if (RightPupilCounter > MAX_PUPIL_LOST || rightEyeLost) {
            eyes_cascade.detectMultiScale(frame(Rect(video_width*0.5,0,video_width*0.5,video_height)), eyes, sc_fct, min_neigh, 0, min_eye, max_eye);
            if (eyes.size() > 0) {
                r = maxEye(eyes);
                r.x = r.x + 0.5*video_width;
                if (r.width > 0) {
                    rightEyeLost = false;
                    right_pup.x = r.x;
                    right_pup.y = r.y;
                }
            }
        }
        if (LeftPupilCounter > MAX_PUPIL_LOST || leftEyeLost) {
            eyes_cascade.detectMultiScale(frame(Rect(0,0,video_width*0.5,video_height)), eyes, sc_fct, min_neigh, 0, min_eye, max_eye);
            if (eyes.size() > 0) {
                l = maxEye(eyes);
                if (l.width > 0) {
                    leftEyeLost = false;
                    left_pup.x = l.x;
                    left_pup.y = l.y;
                }
            }
        }
        
        //right pupil
        findpupil(frame(right_pup), r.width, centerRightPup, radius);
        
        if(radius > 0) {
            if (RightPupilCounter != 0) RightPupilCounter = 0;
            centerRightPup.x = centerRightPup.x + right_pup.x;  //adjust x position of center (by x pos of ROI)
            centerRightPup.y = centerRightPup.y + right_pup.y;  //adjuste y position of center (by y pos of ROI)
            right_pup.x = centerRightPup.x - 0.5*right_pup.width;   //update new x of eye ROI with pupil at center
            right_pup.y = centerRightPup.y - 0.5*right_pup.height;  //update new y of eye ROI with pupil at center
            
            DIAMS_RIGHTS[count] = 2*K*radius;   //save new diamater
            CENTERS_RIGHTS[count] = centerRightPup;     //save new center
            r_ch[count] = true;     //check finded pupil
        }else if (count!= 0){       //if there is not pupil, save previous values
            RightPupilCounter++;
            DIAMS_RIGHTS[count] = DIAMS_RIGHTS[count-1];
            CENTERS_RIGHTS[count] = CENTERS_RIGHTS[count-1];
            r_ch[count] = false;
        }
        
        // left pupil
        findpupil(frame(left_pup), l.width, centerLeftPup, radius);
        
        if(radius > 0) {
            if (LeftPupilCounter != 0) LeftPupilCounter = 0;
            centerLeftPup.x = centerLeftPup.x + left_pup.x;   //adjust x position of center (by x pos of ROI)
            centerLeftPup.y = centerLeftPup.y + left_pup.y;   //adjust y position of center (by y pos of ROI)
            left_pup.x = centerLeftPup.x - 0.5*left_pup.width;   //update new x of eye ROI with pupil at center
            left_pup.y = centerLeftPup.y - 0.5*left_pup.height;    //update new y of eye ROI with pupil at center
            
            DIAMS_LEFTS[count] = 2*K*radius;    //save new diamater
            CENTERS_LEFTS[count] = centerLeftPup;     //save new center
            l_ch[count] = true;     //check finded pupil
        }else if (count!= 0){       //if there is not pupil, save previous values
            LeftPupilCounter++;
            DIAMS_LEFTS[count] = DIAMS_LEFTS[count-1];
            CENTERS_LEFTS[count] = CENTERS_LEFTS[count-1];
            l_ch[count] = false;
        }
        
        //pupils check
        if (abs(dst(centerLeftPup, centerRightPup) - distancePups) > MAX_DISTANCE && count!=0) {
            if (dst(CENTERS_LEFTS[count-1], CENTERS_LEFTS[count-2]) - dst(CENTERS_RIGHTS[count-1], CENTERS_RIGHTS[count-2]) > 0) {
                leftEyeLost = true;
                l_ch[count] = false;
                DIAMS_LEFTS[count] = DIAMS_LEFTS[count-1];
                CENTERS_LEFTS[count] = CENTERS_LEFTS[count-1];
                errorLeftEye.push_back(count);
            }else if (dst(CENTERS_LEFTS[count-1], CENTERS_LEFTS[count-2]) - dst(CENTERS_RIGHTS[count-1], CENTERS_RIGHTS[count-2]) < 0) {
                rightEyeLost = true;
                r_ch[count] = false;
                DIAMS_RIGHTS[count] = DIAMS_RIGHTS[count-1];
                CENTERS_RIGHTS[count] = CENTERS_RIGHTS[count-1];
                errorRightEye.push_back(count);
            }
        }
        if (count !=0) distancePups = (count*distancePups + dst(centerRightPup, centerLeftPup))/(count+1);
        else distancePups = dst(centerRightPup, centerLeftPup);
        
        
        //______________________________OUTPUT GENERATION______________________________
        
        //clean background output image
        rubber(bground);
        //write text on background abou parameters
        write(bground, count, DIAMS_RIGHTS[count], DIAMS_LEFTS[count], scale_f);
        
        //resized eyes
        resize(frame(right_pup), res_right, res_right.size(), res_fac, res_fac, 1);
        resize(frame(left_pup), res_left, res_left.size(), res_fac, res_fac, 1);
        
        //circle pupils
        circle(frame, CENTERS_LEFTS[count], 0.5*DIAMS_LEFTS[count]/K, Scalar(0,0,255)); //circle left pupil
        circle(frame, CENTERS_RIGHTS[count], 0.5*DIAMS_RIGHTS[count]/K, Scalar(0,0,255));   //circle right pupil
        draw(bground, res_left, 0, bgH*0.5); //draw resized left ROI on output
        draw(bground, res_right, (bgW-res_fac*W), bgH*0.5); //draw resized right ROI on output
        draw(bground, frame, 0, 0); //draw frame from video on output background
        
        //image storage
        out.write(bground);
    
        count++;
        
        if( waitKey(1) == 27 ) break; // stop capturing by pressing ESC
    }
    
    //______________________________FILE STORAGE______________________________
    FILE *fd, *fs, *fdst, *fcd, *fcs;
    
    fd=fopen("<Path>/right_diameters.txt", "w");
    fs=fopen("<Path>/left_diameter.txt", "w");
    fcd=fopen("<Path>/right_centres.txt", "w");
    fcs=fopen("<Path>left_centres.txt", "w");
    
    for(int i=0; i<frames; i++) {
        fprintf(fd, "%f\n", DIAMS_RIGHTS[i]);
        fprintf(fs, "%f\n", DIAMS_LEFTS[i]);
        fprintf(fcd, "%d %d\n", CENTERS_RIGHTS[i].x, CENTERS_RIGHTS[i].y );
        fprintf(fcs, "%d %d\n", CENTERS_LEFTS[i].x, CENTERS_LEFTS[i].y );
    }
    
    fclose(fd); fclose(fs); fclose(fCs); fclose(fCs);
    
    return 0;
}

//______________________________FUNCTIONS______________________________

//Function draws an image into another larger one
void draw(Mat img1, Mat img2, int i, int j) {
    int h = img2.rows;
    int w = img2.cols;
    img2.copyTo(img1(Rect(i, j, w, h)));
    
}

//Function makes all black an image
void rubber(Mat image) {
    image = Scalar(0,0,0);
}

//Function filters a ROI image, calculates countours and approximates pupil with a cirlce updating center and radius values
void findpupil(Mat roi_eye, int w, Point2f& center, float& radius) {
    float max_rad = w*0.375;
    float min_rad = w*0.25;
    vector<vector<Point>> contours;
    double area;
    
    Mat roi, pupilFrame;
    cvtColor(roi_eye, roi, CV_BGR2GRAY);
    equalizeHist(roi, pupilFrame);
    threshold(pupilFrame, pupilFrame, pupTH, 255, 0);
    morphologyEx(pupilFrame, pupilFrame, MORPH_OPEN, window_3);
    morphologyEx(pupilFrame, pupilFrame, MORPH_CLOSE, window_3);
    inRange(pupilFrame, 250, 255, pupilFrame);
    findContours(pupilFrame, contours, 1, 1);

    if (contours.size() >= 2) {
        int maxArea = 0;
        int MAindex = 0;
        int cIndex = 0;
        for (int i=0; i<contours.size(); i++) {
            area = contourArea(contours[i]);
            if (area < maxArea) {
                maxArea = area;
                MAindex = cIndex;
            }
            cIndex++;
        }
        contours[0] = contours[MAindex];
    }
    minEnclosingCircle(contours[0], center, radius);
    if (radius < min_rad || radius > max_rad) { radius = -1;}
}

//Function manages outupt generation writing extracted values on image to be added to video output
void write(Mat image, int count, float diam_r, float diam_l, int scl) {
    int t = (count*1000)/frps;
    int shift = 0.5*(scl*res_fac)/K;
    
    stringstream s1, s2;
    s1 << setprecision(3) << diam_r;
    string right = s1.str();
    s2 << setprecision(3) << diam_l;
    string left = s2.str();
    putText(image, "SCALE "+ to_string(scl)+uom, Point(bg_center-70, h1), 2, 0.6, Scalar(255,255,255));
    putText(image, "DIAMETERS: ", Point(bg_center-70, h2), 2, 0.6, Scalar(255,255,255));
    putText(image, "Right:  "+ right +uom, Point(bg_center-70, h3), 2, 0.6, Scalar(255,255,255));
    putText(image, "Left:  "+ left +uom, Point(bg_center-70, h4), 2, 0.6, Scalar(255,255,255));
    putText(image, "Time:  "+ to_string(t) +" ms", Point(bg_center-70, h5), 2, 0.6, Scalar(255,255,255));
    
    line(image, Point(bg_center-shift,h_line), Point(bg_center+shift,h_line), Scalar(255,255,255), 2);
    line(image, Point(bg_center-shift,h_line+10), Point(bg_center-shift,h_line-10), Scalar(255,255,255), 2);
    line(image, Point(bg_center+shift,h_line+10), Point(bg_center+shift,h_line-10), Scalar(255,255,255), 2);
}

//This function takes as input a list of eyes and returns the biggest one.
Rect maxEye(vector<Rect>& eyesVec) {
    Rect eye;
    int  MArea=0;
    
    for( size_t j = 0; j < eyesVec.size(); j++ ) {
        if (eyesVec[j].width * eyesVec[j].height > MArea) {
            eye = eyesVec[j];
            MArea = eyesVec[j].width * eyesVec[j].height;
        }
    }
    
    return eye;
}
