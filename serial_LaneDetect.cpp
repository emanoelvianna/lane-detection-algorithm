/**
 * Versão paralela do algoritmo Lane-Dectection com POSIX Threads.
 * 
 * Paralelização: Gabriell Araujo (gabriell.araujo@acad.pucrs.br)
 * Data da versão: (21/02/2017)
 * 
 * Comando de compilação:
 * g++ -Wall -g -std=c++1y -O3 serial_LaneDetect.cpp -o run_serial `pkg-config --cflags --libs opencv`
 *
 * Comando de execução:
 * ./run_serial auto.mp4 >> serial_log.txt
 */

#include "opencv2/highgui/highgui.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <chrono>
#include "linefinder.h"

using namespace cv;
using namespace std;

VideoWriter oVideoWriter;
int nframes=0;

int main(int argc, char* argv[]) {
	string arg = argv[1];

	setNumThreads(0); //Disabling internal OpenCV's support for multithreading. Necessary for more clear performance comparison.

	VideoCapture capture; 
	capture.open(arg);

	if (!capture.isOpened()) //if this fails, try to open as a video camera, through the use of an integer param
	{capture.open(atoi(arg.c_str()));}

	double dWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	double dHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

	Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));
	oVideoWriter.open("result_serial.avi", CV_FOURCC('P','I','M','1'), 20, frameSize, true); //initialize the VideoWriter object 

	auto tstart = std::chrono::high_resolution_clock::now();

	while (1){
		Mat image;
		capture >> image;

		if ((image).empty()){break;}
		//if ((*threadImage[1]).empty()){break;}
			
		nframes++;

		int houghVote = 200;
		Mat gray;
		cvtColor(image,gray,CV_RGB2GRAY);
		vector<string> codes;
		Mat corners;
		findDataMatrix(gray, codes, corners);
		drawDataMatrixCodes(image, codes, corners);

		Rect roi(0,image.cols/3,image.cols-1,image.rows - image.cols/3);// set the ROI for the image
		Mat imgROI = image(roi);

		// Canny algorithm
		Mat contours;
		Canny(imgROI,contours,50,250);
		Mat contoursInv;
		threshold(contours,contoursInv,128,255,THRESH_BINARY_INV);

		/* 
		   Hough tranform for line detection with feedback
		   Increase by 25 for the next frame if we found some lines.  
		   This is so we don't miss other lines that may crop up in the next frame
		   but at the same time we don't want to start the feed back loop from scratch. 
		 */
		vector<Vec2f> lines;

		if (houghVote < 1 || lines.size() > 2) { // we lost all lines. reset 
			houghVote = 200; }

		else{ houghVote += 25;} 

		while(lines.size() < 5 && houghVote > 0){
			HoughLines(contours,lines,1,PI/180, houghVote);
			houghVote -= 5;  
		}

		Mat result(imgROI.size(),CV_8U,Scalar(255));
		imgROI.copyTo(result);

		// Draw the limes
		vector<Vec2f>::const_iterator it;
		Mat hough(imgROI.size(),CV_8U,Scalar(0));
		it = lines.begin();

		while(it!=lines.end()) {

			float rho= (*it)[0];   // first element is distance rho
			float theta= (*it)[1]; // second element is angle theta				
			if ( (theta > 0.09 && theta < 1.48) || (theta < 3.14 && theta > 1.66) ) 
			{ // filter to remove vertical and horizontal lines

				// point of intersection of the line with first row
				Point pt1(rho/cos(theta),0);
				// point of intersection of the line with last row
				Point pt2((rho-result.rows*sin(theta))/cos(theta),result.rows);
				// draw a white line
				line(result, pt1, pt2, Scalar(255), 8); 
				line(hough, pt1, pt2, Scalar(255), 8);
			}
			++it;
		}

		// Create LineFinder instance			
		LineFinder ld;
		// Set probabilistic Hough parameters
		ld.setLineLengthAndGap(60,10);
		ld.setMinVote(4);

		// Detect lines
		vector<Vec4i> li= ld.findLines(contours);
		Mat houghP(imgROI.size(),CV_8U,Scalar(0));
		ld.setShift(0);
		ld.drawDetectedLines(houghP);

		// bitwise AND of the two hough images
		bitwise_and(houghP,hough,houghP);
		Mat houghPinv(imgROI.size(),CV_8U,Scalar(0));
		threshold(houghP,houghPinv,150,255,THRESH_BINARY_INV); // threshold and invert to black lines

		Canny(houghPinv,contours,100,350);
		li= ld.findLines(contours);

		// Set probabilistic Hough parameters
		ld.setLineLengthAndGap(5,2);
		ld.setMinVote(1);
		ld.setShift(image.cols/3);
		ld.drawDetectedLines(image);

		stringstream stream;
		stream << "Line Segments: " << lines.size();

		putText(image, stream.str(), Point(10,image.rows-10), 2, 0.8, Scalar(0,0,255),0);

		lines.clear();

		oVideoWriter.write(image);
	}		

	auto tend = std::chrono::high_resolution_clock::now();
	double TT; //TotalTime
	TT = std::chrono::duration<double>(tend-tstart).count();
	double TR = nframes/TT;
	cout << TT << " " << TR <<endl;
	//	cout << " NORMAL END OF EXECUTION. ===> Total time(seconds)= "<< TT<<endl;

	//printf("\n\n\nQUANTIDADE DE FRAMES = %d\n\n\n",nframes);

	return 0;
}
