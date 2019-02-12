#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <math.h>
#include <iterator>
#include "circlefit.hh"




namespace fs = std::experimental::filesystem;

using namespace std;
using namespace cv;
using namespace sw;

class Hole
{
public:
	int id;
	Rect rect;
	Point center;
	bool valid;

	/*Hole() {
		valid = true;
	}*/
};

void detectAndDisplay(Mat frame);
void detectAndDisplay2(Mat frame);
void detectHoughCircles(Mat frame);
int PtDistance(Point center1, Point center2);
vector<Hole> GetGoodHoles(vector<Rect> circuits);
int AverageDia(vector<Hole> holeLst);
bool compareRectSize(Hole i, Hole j);
circle_fit::circle_t getCircle(vector<Hole> holeLst);


CascadeClassifier circuit_cascade;
CascadeClassifier eyes_cascade;
string imageName;


// --------- Test Hough Circles Demo ------------ //
namespace
{
	// windows and trackbars name
	const std::string windowName = "Hough Circle Detection Demo";
	const std::string cannyThresholdTrackbarName = "Canny threshold";
	const std::string accumulatorThresholdTrackbarName = "Accumulator Threshold";
	const std::string usage = "Usage : tutorial_HoughCircle_Demo <path_to_input_image>\n";

	// initial and max values of the parameters of interests.
	const int cannyThresholdInitialValue = 100;
	const int accumulatorThresholdInitialValue = 50;
	const int maxAccumulatorThreshold = 200;
	const int maxCannyThreshold = 255;

	void HoughDetection(const Mat& src_gray, const Mat& src_display, int cannyThreshold, int accumulatorThreshold)
	{
		// will hold the results of the detection
		std::vector<Vec3f> circles;
		// runs the actual detection
		HoughCircles(src_gray, circles, HOUGH_GRADIENT, 1, src_gray.rows / 8, cannyThreshold, accumulatorThreshold, 0, 0);

		// clone the colour, input image for displaying purposes
		Mat display = src_display.clone();
		for (size_t i = 0; i < circles.size(); i++)
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			// circle center
			circle(display, center, 3, Scalar(0, 255, 0), -1, 8, 0);
			// circle outline
			circle(display, center, radius, Scalar(0, 0, 255), 3, 8, 0);
		}

		// shows the results
		resize(display, display, Size(display.cols / 3, display.rows / 3));
		imshow(windowName, display);
	}
}


// --------- END -------------------------------- //

int main(int argc, const char** argv)
{
	String circuit_cascade_name = "./cascade.xml";
	//String circuit_cascade_name = "./lbpCascade2020_11stages_5000negs.xml";
	String eyes_cascade_name = "./haarcascade_eye_tree_eyeglasses.xml";
	//-- 1. Load the cascades
	if (!circuit_cascade.load(circuit_cascade_name))
	{
		cout << "--(!)Error loading face cascade\n";
		return -1;
	};
	if (!eyes_cascade.load(eyes_cascade_name))
	{
		cout << "--(!)Error loading eyes cascade\n";
		return -1;
	};

	Mat frame;
	string path = "./images/testImages/OLD";

	for (int i = 1; i < 200; i++)
	{
		imageName = path + "/dist" + to_string(i) + ".jpg";
		//string imageName = entry.path().string();
		frame = imread(imageName, IMREAD_COLOR);
		Mat frameOrg = frame.clone();
		//putText(frame, imageName,Point(20,300), FONT_HERSHEY_SIMPLEX, 6, Scalar(51, 153, 45), 5);
		//resize(frame, frame, Size(), 0.5, 0.5);
		if (frame.empty()) {
			cout << "--(!) No captured frame -- Break!\n";
			break;
		}
		//detectHoughCircles(frame);
		detectAndDisplay2(frame);
		//detectAndDisplay(frameOrg);
		waitKey(0);
		cvDestroyWindow(imageName.c_str());
	}

	//while (capture.read(frame))
	//{
	//	if (frame.empty())
	//	{
	//		cout << "--(!) No captured frame -- Break!\n";
	//		break;
	//	}
	//	//-- 3. Apply the classifier to the frame
	//	detectAndDisplay(frame);
	//	if (waitKey(10) == 27)
	//	{
	//		break; // escape
	//	}
	//}
	return 0;
}
void detectAndDisplay(Mat frame)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	ofstream logFile;
	logFile.open("log.txt");

	//-- Detect circuit holes in distributor
	std::vector<Rect> circuits;
	circuit_cascade.detectMultiScale(frame_gray, circuits, 1.05, 2, 0, Size(50, 50), Size(300, 300));
	int holeCount = 0;
	for (size_t i = 0; i < circuits.size(); i++)
	{
		Point center(circuits[i].x + circuits[i].width / 2, circuits[i].y + circuits[i].height / 2);
		logFile << "id " + to_string(i + 1) + ": ";
		logFile << "(" + to_string(center.x) + ", " + to_string(center.y) + ")\n";
		holeCount++;
		String textBox = to_string(holeCount);
		Size textSize = getTextSize(textBox, FONT_HERSHEY_SIMPLEX, 1, 2, 0);
		Point textLoc = Point(center.x - textSize.width, center.y + (textSize.height / 2));
		putText(frame, textBox, textLoc, FONT_HERSHEY_SIMPLEX, 2, Scalar(51, 153, 45), 3);
		ellipse(frame, center, Size(circuits[i].width / 2, circuits[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 7);
	}

	//-- Show what you got
	logFile.close();
	putText(frame, "Detected Hole Count = " + to_string(holeCount), Point(50, 50), FONT_HERSHEY_SIMPLEX, 2, Scalar(51, 153, 45), 7, 8, false);
	resize(frame, frame, Size(frame.cols / 4, frame.rows / 4));
	namedWindow("Capture - Face detection", CV_WINDOW_AUTOSIZE);
	imshow("Capture - Face detection", frame);
}

void detectHoughCircles(Mat src) {
	Mat src_gray;

	// Convert it to gray
	cvtColor(src, src_gray, COLOR_BGR2GRAY);

	// Reduce the noise so we avoid false circle detection
	GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);

	//declare and initialize both parameters that are subjects to change
	int cannyThreshold = cannyThresholdInitialValue;
	int accumulatorThreshold = accumulatorThresholdInitialValue;

	// create the main window, and attach the trackbars
	namedWindow(windowName, WINDOW_AUTOSIZE);
	createTrackbar(cannyThresholdTrackbarName, windowName, &cannyThreshold, maxCannyThreshold);
	createTrackbar(accumulatorThresholdTrackbarName, windowName, &accumulatorThreshold, maxAccumulatorThreshold);

	// infinite loop to display
	// and refresh the content of the output image
	// until the user presses q or Q
	char key = 0;
	while (key != 'q' && key != 'Q')
	{
		// those parameters cannot be =0
		// so we must check here
		cannyThreshold = std::max(cannyThreshold, 1);
		accumulatorThreshold = std::max(accumulatorThreshold, 1);

		//runs the detection, and update the display
		HoughDetection(src_gray, src, cannyThreshold, accumulatorThreshold);

		// get user key
		key = (char)waitKey(10);
	}
}

static int PtDistance(Point center1, Point center2)
{
	float deltaX = center2.x - center1.x;
	float deltaY = center2.y - center1.y;

	return sqrt(pow(deltaX, 2) + pow(deltaY, 2));
}

Point GetCenterPoint(Rect circuit) {
	return Point(circuit.x + circuit.width / 2, circuit.y + circuit.height / 2);
}

vector<Hole> GetGoodHoles(vector<Rect> circuits)
{
	vector<Hole> goodHoles;
	vector<int> rectWidth;
	int minAllowDistance = 100;

	//populate vector with all circuit data
	for (int i = 0; i < circuits.size(); i++)
	{
		Hole potCircuit;
		potCircuit.id = i;
		potCircuit.rect = circuits[i];
		potCircuit.center = GetCenterPoint(circuits[i]);
		potCircuit.valid = true;
		goodHoles.push_back(potCircuit);

	}

	for (int k = 0, j = 1; j < circuits.size(); k++, j++)
	{
		rectWidth.push_back(goodHoles[k].rect.width);
		cout << "k = " + to_string(k) + "; j = " + to_string(j) + "\n";
		int distance = PtDistance(goodHoles[k].center, goodHoles[j].center);
		// if distance between two points is less than 100 pixels then most likely multiple points sharing a space;
		if (distance < minAllowDistance) {
			sort(rectWidth.begin(), rectWidth.end());
			auto mid = rectWidth.begin() + rectWidth.size() / 2;
			goodHoles[k].center = Point((goodHoles[k].center.x + goodHoles[j].center.x) / 2, (goodHoles[k].center.y + goodHoles[j].center.y) / 2);
			goodHoles[k].rect.width = *mid;
			goodHoles[k].rect.height = *mid;
			goodHoles[j].valid = false;
		}

	}
	return goodHoles;
}

void detectAndDisplay2(Mat frame)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	cvtColor(frame, frame, COLOR_BGR2GRAY);
	cvtColor(frame, frame, CV_GRAY2RGB);
	equalizeHist(frame_gray, frame_gray);
	ofstream logFile;
	logFile.open("log.txt");

	//-- Detect circuit holes in distributor
	std::vector<Rect> circuits;
	circuit_cascade.detectMultiScale(frame_gray, circuits, 1.01, 3, 0, Size(50, 50), Size(300, 300));

	vector<Hole> circuitHoles = GetGoodHoles(circuits);
	int averageDia = AverageDia(circuitHoles);


	int holeCount = 0;
	for (int i = 0; i < circuitHoles.size(); i++)
	{
		if (circuitHoles[i].valid) {
			float r = (float)circuitHoles[i].rect.width / (float)averageDia;
			if (.6 <= r && r <= 1.4) {
				holeCount++;

				cout << "id " + to_string(holeCount) + ": ";
				cout << "(" + to_string(circuitHoles[i].center.x) + ", " + to_string(circuitHoles[i].center.y) + ")\n";
				logFile << "id " + to_string(holeCount) + ": ";
				logFile << "(" + to_string(circuitHoles[i].center.x) + ", " + to_string(circuitHoles[i].center.y) + ")\n";
				String textBox = to_string(holeCount);
				Size textSize = getTextSize(textBox, FONT_HERSHEY_SIMPLEX, 1, 2, 0);
				Point textLoc = Point(circuitHoles[i].center.x - textSize.width, circuitHoles[i].center.y + (textSize.height / 2));
				putText(frame, textBox, textLoc, FONT_HERSHEY_SIMPLEX, 2, Scalar(51, 153, 45), 3);
				ellipse(frame, circuitHoles[i].center, Size(circuitHoles[i].rect.width / 2, circuitHoles[i].rect.height / 2), 0, 0, 360, Scalar(255, 0, 255), 20);
			}
			else {
				circuitHoles[i].valid = false;
				ellipse(frame, circuitHoles[i].center, Size(circuitHoles[i].rect.width / 2, circuitHoles[i].rect.height / 2), 0, 0, 360, Scalar(201, 50, 30), 20);
			}
		}
		else {
			ellipse(frame, circuitHoles[i].center, Size(circuitHoles[i].rect.width / 2, circuitHoles[i].rect.height / 2), 0, 0, 360, Scalar(201, 30, 87), 20);
		}
	}
	//-- Show what you got
	logFile.close();

	// draw pitch circle
	circle_fit::circle_t circle = getCircle(circuitHoles);
	Point pitchCircleCenter = Point(circle.x, circle.y);
	cv::circle(frame, pitchCircleCenter, 50, Scalar(158, 244, 66), 8);
	ellipse(frame, pitchCircleCenter, Size(circle.r, circle.r), 0, 0, 360, Scalar(158, 244, 66), 8);

	//putText(frame, imageName, Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(51, 153, 45), 4, 8, false);
	putText(frame, "Detected Hole Count = " + to_string(holeCount), Point(50, 200), FONT_HERSHEY_SIMPLEX, 4, Scalar(51, 153, 45), 7, 8, false);
	resize(frame, frame, Size(frame.cols / 4, frame.rows / 4));
	namedWindow(imageName, CV_WINDOW_AUTOSIZE);
	imshow(imageName, frame);
}

int AverageDia(vector<Hole> holeLst) {
	int sum = 0;
	int count = 0;
	std::sort(holeLst.begin(), holeLst.end(), compareRectSize);
	vector<Hole>::iterator iter;
	/*if (holeLst.size() > 5) {*/
		//int startPoint = (holeLst.size() / 2) - (holeLst.size() / 4);

		/*for (int i = startPoint; i < startPoint + 3; i++) {
			sum += holeLst[i].rect.width;
			count++;
		}*/
	for (iter = holeLst.begin() + holeLst.size() / 3; iter != (holeLst.end() - (holeLst.size() / 3)); iter++) {
		//sum += holeLst[i].rect.width;
		if (iter->valid) {
			sum += iter->rect.width;
			count++;
		}
	}
	//}
	/*else {
		for (int i = 0; i < holeLst.size(); i++)
		{
			if (holeLst[i].valid) {
				sum += holeLst[i].rect.width;
				count++;
			}
		}
	}*/

	if (count > 0) {
		return sum / count;
	}
	else {
		return 0;
	}
}

bool compareRectSize(Hole i, Hole j) {
	return (i.rect.width < j.rect.width);
}

circle_fit::circle_t getCircle(vector<Hole> holeLst)
{
	circle_fit::points_t points;
	for (int i = 0; i < holeLst.size(); i++) {

		points.push_back(circle_fit::point_t(holeLst[i].center.x, holeLst[i].center.y));
		//cout << "(" + to_string(holeLst[i].center.x) + ", " + to_string(holeLst[i].center.y) + ") \n";

	}
	circle_fit::circle_t circle = circle_fit::fit_geometric(points, .001);

	return circle;
}

