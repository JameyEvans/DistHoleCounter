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
	int nearNeighbors;
	int numDetections;

	Hole() {
		valid = true;
		nearNeighbors = 0;
		numDetections = 0;
	}

	
};

void detectAndDisplay(Mat frame);
void detectAndDisplay2(Mat frame);
int PtDistance(const Point& center1, const Point& center2);
vector<Hole> GetGoodHoles(vector<Rect> circuits, vector<int>& numDetections);
int AverageDia(vector<Hole>& holeLst);
bool compareRectSize(Hole i, Hole j);
circle_fit::circle_t getCircle(vector<Hole> &holeLst);


CascadeClassifier circuit_cascade;
string imageName;
vector<int> avgSizeLst;
ofstream logFile;




int main(int argc, const char** argv)
{
	String circuit_cascade_name = "./cascade.xml";
	//String circuit_cascade_name = "./lbpCascade2020_11stages_5000negs.xml";
	String eyes_cascade_name = "./haarcascade_eye_tree_eyeglasses.xml";
	//-- 1. Load the cascades
	if (!circuit_cascade.load(circuit_cascade_name))
	{
		cout << "--(!)Error loading distributor cascade\n";
		return -1;
	};
	
	logFile.open("log.txt");

	Mat frame;
	string path = "./images/testImages/quarter_021219";

	for (int i = 1; i < 200; i++)
	{
		

		imageName = path + "/dist" + to_string(i) + ".jpg";
		//string imageName = entry.path().string();
		frame = imread(imageName, IMREAD_COLOR);
		if (frame.empty()) {
			cout << "--(!) No captured frame -- Break!\n";
			break;
		}
		resize(frame, frame, Size(frame.cols / 2.5, frame.rows / 2.5));
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
	for (size_t i = 0; i < avgSizeLst.size(); i++)
	{
		logFile << "Average size: " + to_string(avgSizeLst[i]) << endl;
	}

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



static int PtDistance(const Point& center1, const Point& center2)
{
	float deltaX = center2.x - center1.x;
	float deltaY = center2.y - center1.y;

	return sqrt(pow(deltaX, 2) + pow(deltaY, 2));
}

Point GetCenterPoint(Rect& circuit) {
	return Point(circuit.x + circuit.width / 2, circuit.y + circuit.height / 2);
}



vector<Hole> GetGoodHoles(vector<Rect> circuits, vector<int>& numDetections)
{
	vector<Hole> goodHoles;
	vector<int> rectWidth;
	int maxAllowDistance = 200;  //400 for full size

	//populate vector with all circuit data
	for (int i = 0; i < circuits.size(); i++)
	{
		Hole potCircuit;
		potCircuit.id = i;
		potCircuit.rect = circuits[i];
		potCircuit.center = GetCenterPoint(circuits[i]);
		potCircuit.numDetections = numDetections[i];
		goodHoles.push_back(potCircuit);

	}

	// count number of near neighbors
	int maxMeasuredDistance = 0;
	int id = 0;
	for (size_t j = 0; j < goodHoles.size(); j++)
	{

		for (size_t k = 0; k < goodHoles.size(); k++)
		{
			
			if (j != k) {
				int distance = PtDistance(goodHoles[j].center, goodHoles[k].center);
				if (distance < maxAllowDistance) {
					goodHoles[j].nearNeighbors++;
					if (distance > maxMeasuredDistance) { maxMeasuredDistance = distance; id = j; }
				}
			}
		}
	}
	cout << "Max measured distance = " + to_string(maxMeasuredDistance) << endl;
	cout << "ID = " + to_string(id) << endl;

	// if near neighbors < 2 then probably not a valid hole
	for (size_t i = 0; i < goodHoles.size(); i++)
	{
		if (goodHoles[i].nearNeighbors < 2 && goodHoles[i].numDetections < 10) {
			goodHoles[i].valid = false;
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
	Mat frame_copy = frame.clone();
	//ofstream logFile;
	//logFile.open("log.txt");

	//-- Detect circuit holes in distributor
	std::vector<Rect> circuits;
	std::vector<int> numDetections;
	circuit_cascade.detectMultiScale(frame_gray, circuits, numDetections, 1.05, 1,0, Size(20, 20), Size(150, 150));   // was (50,50) and (300, 300)

	vector<Hole> circuitHoles = GetGoodHoles(circuits, numDetections);
	int averageDia = AverageDia(circuitHoles);

	// ---------------- Display all detections in separate window for testing -----------------------------------

	for (size_t i = 0; i < circuitHoles.size(); i++)
	{
		ellipse(frame_copy, circuitHoles[i].center, Size(circuitHoles[i].rect.width / 2, circuitHoles[i].rect.height / 2), 0, 0, 360, Scalar(201, 30, 87), 3);
		String textBox = to_string(circuitHoles[i].numDetections);
		Size textSize = getTextSize(textBox, FONT_HERSHEY_SIMPLEX, 1, 8, 0);
		Point textLoc = Point(circuitHoles[i].center.x - textSize.width, circuitHoles[i].center.y + (textSize.height / 2));
		putText(frame_copy, textBox, textLoc, FONT_HERSHEY_SIMPLEX, 1, Scalar(51, 153, 45), 3);
	}
	namedWindow("Original Detections", 1);
	imshow("Original Detections", frame_copy);

	// ---------------- End Test ------------------------------------------------------------------------------------

	int holeCount = 0;
	for (int i = 0; i < circuitHoles.size(); i++)
	{
		if (circuitHoles[i].valid) {
			float r = (float)circuitHoles[i].rect.width / (float)averageDia;
			if (.6 >= r || r >= 1.4) {
				if (circuitHoles[i].numDetections < 5) {
					circuitHoles[i].valid = false;				
				}
			}
		}
		else {
			ellipse(frame, circuitHoles[i].center, Size(circuitHoles[i].rect.width / 2, circuitHoles[i].rect.height / 2), 0, 0, 360, Scalar(201, 30, 87),3);  //thickness was 20
		}
	}
	//-- Show what you got
	//logFile.close();

	// draw pitch circle
	circle_fit::circle_t circle = getCircle(circuitHoles);
	Point pitchCircleCenter = Point(circle.x, circle.y);
	cv::circle(frame, pitchCircleCenter, 50, Scalar(158, 244, 66), 5);  //thickness was 8
	ellipse(frame, pitchCircleCenter, Size(circle.r, circle.r), 0, 0, 360, Scalar(158, 244, 66), 3); // was 8

	for (size_t i = 0; i < circuitHoles.size(); i++)
	{
		if (circuitHoles[i].valid) {
			holeCount++;
			//cout << "id " + to_string(holeCount) + ": ";
			//cout << "(" + to_string(circuitHoles[i].center.x) + ", " + to_string(circuitHoles[i].center.y) + ")\n";
			//logFile << "id " + to_string(holeCount) + ": ";
			//logFile << "(" + to_string(circuitHoles[i].center.x) + ", " + to_string(circuitHoles[i].center.y) + ")\n";
			String textBox = " " + to_string(holeCount);
			Size textSize = getTextSize(textBox, FONT_HERSHEY_SIMPLEX, 1, 2, 0);
			Point textLoc = Point(circuitHoles[i].center.x - textSize.width, circuitHoles[i].center.y + (textSize.height / 2));
			putText(frame, textBox, textLoc, FONT_HERSHEY_SIMPLEX, 1, Scalar(51, 153, 45), 1);
			ellipse(frame, circuitHoles[i].center, Size(circuitHoles[i].rect.width / 2, circuitHoles[i].rect.height / 2), 0, 0, 360, Scalar(255, 0, 255), 3); //was 20
		}
		
	}

	//putText(frame, imageName, Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(51, 153, 45), 4, 8, false);
	putText(frame, "Detected Hole Count = " + to_string(holeCount), Point(20, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(51, 153, 45), 2, 8, false);
	//resize(frame, frame, Size(frame.cols / 4, frame.rows / 4));
	namedWindow(imageName, CV_WINDOW_AUTOSIZE);
	imshow(imageName, frame);
}

int AverageDia(vector<Hole>& holeLst) {
	int sum = 0;
	int count = 0;
	std::sort(holeLst.begin(), holeLst.end(), compareRectSize);
	vector<Hole>::iterator iter;

	for (iter = holeLst.begin() + holeLst.size() / 3; iter != (holeLst.end() - (holeLst.size() / 3)); iter++) {
		//sum += holeLst[i].rect.width;
		if (iter->valid && iter->numDetections > 10) {
			sum += iter->rect.width;
			count++;
		}
	}

	if (count > 0) {
		avgSizeLst.push_back(sum / count);
		return sum / count;
	}
	else {
		return 0;
	}
}

bool compareRectSize(Hole i, Hole j) {
	return (i.rect.width < j.rect.width);
}

circle_fit::circle_t getCircle(vector<Hole> &holeLst)
{
	circle_fit::points_t points;
	for (int i = 0; i < holeLst.size(); i++) {
		if (holeLst[i].valid && holeLst[i].numDetections > 10) {
			points.push_back(circle_fit::point_t(holeLst[i].center.x, holeLst[i].center.y));
		}
		//cout << "(" + to_string(holeLst[i].center.x) + ", " + to_string(holeLst[i].center.y) + ") \n";

	}
	circle_fit::circle_t circle = circle_fit::fit(points);
	for (size_t i = 0; i < holeLst.size(); i++)
	{
		float pcr = circle.r;
		if (holeLst[i].valid) {
			float calcRad = sqrt(pow(holeLst[i].center.x - circle.x, 2) + pow(holeLst[i].center.y - circle.y, 2));
			float ratio = abs(calcRad / pcr);
			cout << "ratio: " + to_string(ratio) << endl;
			if (!(.8 < ratio && ratio < 1.2)) {
				holeLst[i].valid = false;
			}
		}

	}

	return circle;
}








// ------------------------- old functions ----------------------------------
//vector<Hole> GetGoodHolesOld(vector<Rect> circuits)
//{
//	vector<Hole> goodHoles;
//	vector<int> rectWidth;
//	int minAllowDistance = 100;
//
//	//populate vector with all circuit data
//	for (int i = 0; i < circuits.size(); i++)
//	{
//		Hole potCircuit;
//		potCircuit.id = i;
//		potCircuit.rect = circuits[i];
//		potCircuit.center = GetCenterPoint(circuits[i]);
//		potCircuit.valid = true;
//		goodHoles.push_back(potCircuit);
//
//	}
//
//	for (int k = 0, j = 1; j < circuits.size(); k++, j++)
//	{
//		rectWidth.push_back(goodHoles[k].rect.width);
//		cout << "k = " + to_string(k) + "; j = " + to_string(j) + "\n";
//		int distance = PtDistance(goodHoles[k].center, goodHoles[j].center);
//		// if distance between two points is less than 100 pixels then most likely multiple points sharing a space;
//		if (distance < minAllowDistance) {
//			sort(rectWidth.begin(), rectWidth.end());
//			auto mid = rectWidth.begin() + rectWidth.size() / 2;
//			goodHoles[k].center = Point((goodHoles[k].center.x + goodHoles[j].center.x) / 2, (goodHoles[k].center.y + goodHoles[j].center.y) / 2);
//			goodHoles[k].rect.width = *mid;
//			goodHoles[k].rect.height = *mid;
//			goodHoles[j].valid = false;
//		}
//
//	}
//	return goodHoles;
//}