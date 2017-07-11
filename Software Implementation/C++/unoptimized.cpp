#include <opencv2/core/core.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <ctime>
#include <sys/time.h>

#define FRM 950

using namespace std;
using namespace cv;


timespec start, t1, t2, temp;
timespec diff(timespec start, timespec end);
long ms;


// Source folder
String source_folder = "/Users/filippobracco/Code_cpp/opencv_test/Assets/";
String output_folder = "/Users/filippobracco/Code_cpp/";

// eye cascade parameters
double sc_fct = 1.3;
int min_neigh = 3;
Size min_eye = Size(20, 20);
Size max_eye = Size(60, 60);
String eyes_cascade_name = source_folder + "haarcascade_eye.xml";
CascadeClassifier eyes_cascade;
int dist_TH = 30;
bool check_r = false;
bool check_l = false;

// eyes
vector<Rect> eyes;
int x0r;
int y0r;
int w0r;
int h0r;
int x0l;
int y0l;
int w0l;
int h0l;

// Bg image
String bg_file = source_folder + "Background.png";
Mat bground = imread(bg_file);
int bgH = bground.size[0];
int bgW = bground.size[1];


// suorce file to analize:
String video_file = source_folder + "Stimolazione_dinamica.mp4";
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
int res_fac = 3.5;
int bound = 10;

// conversion factors
float K = 0.597; //from pixel to mm
int scale_f;
String uom = "mm";

// pupil function
double pupTH = 25;
Mat window_5 = Mat::ones(5, 5, CV_8U);
Mat window_2 = Mat::ones(2, 2, CV_8U);
float radius;
Point2f center;
float DIAMS_RIGHTS[FRM], DIAMS_LEFTS[FRM];
Point CENTERS_RIGHTS[FRM], CENTERS_LEFTS[FRM];
bool r_ch[FRM], l_ch[FRM];


//writing factors
int h1 = bgH/2 + 20;
int h_line = h1 + 30;
int h2 = h1 + 80;
int h3 = h2 + 30;
int h4 = h3 + 30;
int h5 = h4 + 50;
int bg_center = bgW/2;
int pos_r;
int pos_vl;
int pos_vr;


Mat frame;
Rect r;
Rect l;
Rect roi_right;
Rect roi_left;
int min_dist;
Mat res_right;
Mat res_left;



void draw(Mat img1, Mat img2, int i, int j);
void rubber(Mat image);
void findpupil(Mat roi, int w, Point2f& center, float& radius);
void write(Mat image, int count, float diam_r, float diam_l, int scl);

int main(int argc, char** argv)
{
    clock_gettime(CLOCK_MONOTONIC, &start);
    if(!cap.isOpened())
    {
        printf("Error in video loading");
        return -1;
    }
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    
    while(eyes.size() != 2)
    {
        Mat frame;
        cap >> frame;
        eyes_cascade.detectMultiScale(frame, eyes, sc_fct, min_neigh, 0, min_eye, max_eye);
    }
    
    if (eyes[0].x > eyes[1].x)
    {
        x0r = eyes[0].x;
        y0r = eyes[0].y;
        w0r = eyes[0].width;
        h0r = eyes[0].height;
        x0l = eyes[1].x;
        y0l = eyes[1].y;
        w0l = eyes[1].width;
        h0l = eyes[1].height;
    }else{
        x0r = eyes[1].x;
        y0r = eyes[1].y;
        w0r = eyes[1].width;
        h0r = eyes[1].height;
        x0l = eyes[0].x;
        y0l = eyes[0].y;
        w0l = eyes[0].width;
        h0l = eyes[0].height;
    }
    
    scale_f = 0.5*(w0r+w0l)*K;
    int count=0;
    VideoCapture cap(video_file);
    
    pos_r = bgW - res_fac*(w0r + 2*bound);
    pos_vl = bgH - res_fac*(h0l + 2*bound);
    pos_vr = bgH - res_fac*(h0r + 2*bound);
    
    
    while(cap.isOpened())
    {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        cap >> frame;
        if( frame.empty() ) break; // end of video stream
        vector<Rect> eyes;
        vector<Rect> rights;
        vector<Rect> lefts;
        eyes_cascade.detectMultiScale(frame, eyes, sc_fct, min_neigh, 0, min_eye, max_eye);
        
        check_l = false;
        check_r = false;
        
        //searching for eyes
        for( size_t j = 0; j < eyes.size(); j++ )
        {
            if (eyes[j].x > video_width*0.5 && (abs(eyes[j].x-x0r) < dist_TH && abs(eyes[j].y-y0r) < dist_TH))
            {
                rights.push_back(eyes[j]);
            }else if(eyes[j].x < video_width*0.5 && (abs(eyes[j].x-x0l) < dist_TH && abs(eyes[j].y-y0l) < dist_TH))
            {
                lefts.push_back(eyes[j]);
            }
            
            //rectangle(frame, eyes[j], Scalar( 0, 255, 0 ), 1, 8, 0);
        }
        
        //find correct left and right eye
        min_dist = video_width;
        for( size_t j = 0; j < rights.size(); j++ )
        {
            if (abs(rights[j].x - x0r) < min_dist) { r=rights[j]; min_dist = abs(rights[j].x - x0r); check_r = true;}
        }
        min_dist = video_width;
        for( size_t j = 0; j < lefts.size(); j++ )
        {
            if (abs(lefts[j].x - x0l) < min_dist) { l=lefts[j]; min_dist = abs(lefts[j].x - x0l); check_l = true;}
        }
        
        roi_right = Rect(x0r-bound, y0r-bound, w0r+2*bound, h0r+2*bound);
        roi_left = Rect(x0l-bound, y0l-bound, w0l+2*bound, h0l+2*bound);
        //resized eyes
        resize(frame(roi_right), res_right, res_right.size(), res_fac, res_fac,1);
        resize(frame(roi_left), res_left, res_left.size(), res_fac, res_fac,1);
        
        //right pupil
        if(check_r) {
            findpupil(frame(r), r.width, center, radius);
            if(radius > 0) {
                center.x = center.x + r.x;
                center.y = center.y + r.y;
                DIAMS_RIGHTS[count] = 2*K*radius;
                CENTERS_RIGHTS[count] = center;
                r_ch[count] = true;
            }else if (count!= 0){
                DIAMS_RIGHTS[count] = DIAMS_RIGHTS[count-1];
                CENTERS_RIGHTS[count] = CENTERS_RIGHTS[count-1];
                r_ch[count] = r_ch[count-1];
            }
        }else{
            DIAMS_RIGHTS[count] = DIAMS_RIGHTS[count-1];
            CENTERS_RIGHTS[count] = CENTERS_RIGHTS[count-1];
            r_ch[count] = r_ch[count-1];
        }
        
        circle(frame, CENTERS_RIGHTS[count], 0.5*DIAMS_RIGHTS[count]/K, Scalar(0,0,255));
        
        // left pupil
        if(check_l) {
            findpupil(frame(l), l.width, center, radius);
            if(radius > 0) {
                center.x = center.x + l.x;
                center.y = center.y + l.y;
                DIAMS_LEFTS[count] = 2*K*radius;
                CENTERS_LEFTS[count] = center;
                l_ch[count] = true;
            }else if (count!= 0){
                DIAMS_LEFTS[count] = DIAMS_LEFTS[count-1];
                CENTERS_LEFTS[count] = CENTERS_LEFTS[count-1];
                l_ch[count] = l_ch[count-1];
            }
        }else{
            DIAMS_LEFTS[count] = DIAMS_LEFTS[count-1];
            CENTERS_LEFTS[count] = CENTERS_LEFTS[count-1];
            l_ch[count] = l_ch[count-1];
        }
        
        circle(frame, CENTERS_LEFTS[count], 0.5*DIAMS_LEFTS[count]/K, Scalar(0,0,255));
        
        //rectangle(frame, r, Scalar( 255, 0, 0 ), 1, 8, 0);
        //rectangle(frame, l, Scalar( 255, 0, 0 ), 1, 8, 0);
        rubber(bground);
        write(bground, count, DIAMS_RIGHTS[count], DIAMS_LEFTS[count], scale_f);
        
        draw(bground, res_left, 0, pos_vl);
        draw(bground, res_right, pos_r, pos_vr);
        draw(bground, frame, 0, 0);
        
        
        
        out.write(bground);
        //imshow("Output", bground);
        count++;
        if (!(count%60))
        {
            if(r.x !=0) {x0r = r.x; y0r = r.y;}
            if(l.x !=0) {x0l = l.x; y0l = l.y;}
            pos_r = bgW - res_fac*(w0r + 2*bound);
            pos_vl = bgH - res_fac*(h0l + 2*bound);
            pos_vr = bgH - res_fac*(h0r + 2*bound);
        }
        clock_gettime(CLOCK_MONOTONIC, &t2);
        cout<<"Loop time: "<<diff(t1, t2).tv_sec<<"."<<diff(t1, t2).tv_nsec<<" s\n";
        if( waitKey(1) == 27 ) break; // stop capturing by pressing ESC
    }
    
    
    //write the file
    ofstream f_out;
    f_out.open(output_folder+"output.txt");
    for(int i=0; i<frames; i++) {
        f_out<<setprecision(4)<<DIAMS_RIGHTS[i]<<" "<<CENTERS_RIGHTS[i].x<<" "<<CENTERS_RIGHTS[i].y<<"   "<<DIAMS_LEFTS[i]<<" "<<CENTERS_LEFTS[i].x<<" "<<CENTERS_LEFTS[i].y<<"\n";
    }
    f_out.close();
    
    clock_gettime(CLOCK_MONOTONIC, &t2);
    cout<<"Total execution time: "<<diff(start, t2).tv_sec<<"."<<diff(start, t2).tv_nsec<<" s\n";
    return 0;
}
void draw(Mat img1, Mat img2, int i, int j) {
    int h = img2.rows;
    int w = img2.cols;
    img2.copyTo(img1(Rect(i, j, w, h)));
    
}

void rubber(Mat image) {
    image = Scalar(0,0,0);
}

void findpupil(Mat roi_eye, int w, Point2f& center, float& radius) {
    float max_rad = w/3;
    float min_rad = w/10;
    vector<vector<Point>> contours;
    double area;
    
    
    Mat roi, pupilFrame;
    cvtColor(roi_eye, roi, CV_BGR2GRAY);
    equalizeHist(roi, pupilFrame);
    threshold(pupilFrame, pupilFrame, pupTH, 255, 0);
    
    morphologyEx(pupilFrame, pupilFrame, 1, window_5);
    erode(pupilFrame, pupilFrame, window_2);
    morphologyEx(pupilFrame, pupilFrame, 0, window_2);
    
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
    
    drawContours(pupilFrame, contours, 0, Scalar(255,0,0));
    
}

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

timespec diff(timespec start, timespec end)
{
    timespec temp;
    if ((end.tv_nsec-start.tv_nsec)<0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec-start.tv_sec;
        temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    return temp;
}


