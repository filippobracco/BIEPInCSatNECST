#include <pxcsensemanager.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream> 
#include <stdio.h>

cv::Mat PXCImage2CVMat(PXCImage *pxcImage, PXCImage::PixelFormat format); // from PXC image to OpeCV

bool keepIMG = true;

int main(int argc, char*argv[]) {
    
    float frameRate = 60;
    
    //PXC parameters setup
	std::vector <int> parameters;
	parameters.push_back(CV_IMWRITE_PNG_COMPRESSION);
	parameters.push_back(9);
	PXCSession *session = PXCSession::CreateInstance();
	PXCSenseManager *sm = PXCSenseManager::CreateInstance();

    //OpenCV parameters setup
   cv::Size roi_Size = cv::Size(640, 250);
	cv::Size frameSize = cv::Size(640, 480);
	cv::namedWindow("IR", cv::WINDOW_NORMAL);
	cv::Mat frameIR = cv::Mat::zeros(frameSize, CV_8UC1);
	cv::Rect region_of_interest = cv::Rect(0, 10, 640, 250);

    // init camera for acquisition
	PXCSenseManager *pxcSenseManager = PXCSenseManager::CreateInstance();
	pxcSenseManager->EnableStream(PXCCapture::STREAM_TYPE_IR, frameSize.width, frameSize.height, frameRate);
	pxcSenseManager->Init();

	bool isColor = false;

    //video writer setup
	cv::VideoWriter writer;
	int codec = CV_FOURCC('F', 'M', 'P', '4');
	std::string filename = "Stimulation_video.mp4";
	writer.open(filename, codec, frameRate, roi_Size, isColor);

    //stimulation video loading
	cv::VideoCapture videoCapture;
	videoCapture.open("Stimulaion_test.mp4");

	cv::namedWindow("Test", cv::WINDOW_NORMAL);
	cv::Mat dotImage = cv::Mat::zeros(frameSize, CV_8UC1);

	if (!writer.isOpened()) {
		std::cout << "Could not open the output video file for write\n";
		return -1;
	}

	if (!videoCapture.isOpened()) {
		std::cout << "Could not open the input video file for reading\n";
		return -1;
	}

	bool keepRunning = true;
    
    
    //____________________________ITERATIVE PART____________________________
    //acquisition totally synch with stimulation video
	while (keepRunning) {
		pxcSenseManager->AcquireFrame();
		PXCCapture::Sample *sample = pxcSenseManager->QuerySample();

		frameIR = PXCImage2CVMat(sample->ir, PXCImage::PIXEL_FORMAT_Y8);
		cv::rectangle(frameIR, region_of_interest, 255, 1, 8);
		cv::Mat image_roi = frameIR(region_of_interest);
		cv::imshow("IR", image_roi);
		cv::imshow("IRroi", frameIR);

		if (videoCapture.read(dotImage)) cv::imshow("Test", dotImage);
		else keepRunning = false;
		
		int key = cv::waitKey(1);
		if (key == 27) keepRunning = false; // ESC key

		writer.write(image_roi); //add image in video

		pxcSenseManager->ReleaseFrame();
	}
	pxcSenseManager->Release();
	videoCapture.release();

	return 0;
}

//_______________FUNCTION: from PXC image to OpenCV Mat image_______________

cv::Mat PXCImage2CVMat(PXCImage *pxcImage, PXCImage::PixelFormat format) {
    PXCImage::ImageData data;
    pxcImage->AcquireAccess(PXCImage::ACCESS_READ, format, &data);
    
    int width = pxcImage->QueryInfo().width;
    int height = pxcImage->QueryInfo().height;
    
    if (!format) format = pxcImage->QueryInfo().format;
    
    int type;
    if (format == PXCImage::PIXEL_FORMAT_Y8) type = CV_8UC1;
    else if (format == PXCImage::PIXEL_FORMAT_RGB24) type = CV_8UC3;
    else if (format == PXCImage::PIXEL_FORMAT_DEPTH_F32) type = CV_32FC1;
    
    cv::Mat ocvImage = cv::Mat(cv::Size(width, height), type, data.planes[0]);
    
    pxcImage->ReleaseAccess(&data);
    
    return ocvImage;
}
