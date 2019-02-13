#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <math.h>
#include <iterator>
#include "circlefit.hh"
#include <unordered_map>




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
	int quad;
	

	Hole() {
		valid = true;
		nearNeighbors = 0;
		numDetections = 0;
		quad = 0;
	}


};


void detectAndDisplay(Mat frame);
int PtDistance(const Point& center1, const Point& center2);
bool getImgFromCam(VideoCapture& cap, int& imgCounter, string& path);

vector<Hole> GetGoodHoles(vector<Rect> circuits, vector<int>& numDetections);
int AverageDia(vector<Hole>& holeLst);
bool compareRectSize(Hole i, Hole j);
bool detectNumSort(Hole i, Hole j);
bool sortQuad(Hole i, Hole j);
void orderByQuad(vector<Hole>& goodHoles, circle_fit::circle_t& pcd);
circle_fit::circle_t getCircle(vector<Hole> &holeLst);
string getHoleSize(const int& d);
int estimateHoleCount(vector<Hole>& goodHoles, circle_fit::circle_t& circle);
Point GetTextPoint(Hole& hole, circle_fit::circle_t& circle);


CascadeClassifier circuit_cascade;
string imageName;
vector<int> avgSizeLst;
ofstream logFile, counterFile;
circle_fit::circle_t pcd;
int d;




int main(int argc, const char** argv)
{
	String circuit_cascade_name = "./cascade.xml";

	//-- 1. Load the cascades
	if (!circuit_cascade.load(circuit_cascade_name))
	{
		cout << "--(!)Error loading distributor cascade\n";
		return -1;
	};

	VideoCapture cap(1); // open the default camera
	if (!cap.isOpened()) {
		cout << "Default camera was unable to be opened with VideoCapture(1)" << endl;
		return -1;
	}

	logFile.open("log.txt");
	int imgCounter = 0;
	//counterFile.open("counterFile.txt");

	ifstream input_file("counterFile.txt");
	double tempVar;
	vector<double> tempVector;

	if (input_file.good()) {		
		while (input_file >> tempVar) {
			tempVector.push_back(tempVar);
		}
		if (tempVector.size() > 0) {
			imgCounter = (int)tempVector[0];
		}		
	}

	Mat frame;
	string path = "./images/testImages/webcam";

	/*for (int i = 1; i < 200; i++)
	{
		imageName = path + "/dist" + to_string(i) + ".jpg";
		frame = imread(imageName, IMREAD_COLOR);
		if (frame.empty()) {
			cout << "--(!) No captured frame -- Break!\n";
			break;
		}
		resize(frame, frame, Size(frame.cols / 3, frame.rows / 3));
		Mat frameOrg = frame.clone();
		detectAndDisplay(frame);
		waitKey(0);
		cvDestroyWindow(imageName.c_str());
	}*/

	while (getImgFromCam(cap, imgCounter, path))
	{
		imageName = path + "/dist" + to_string(imgCounter) + ".png";
		frame = imread(imageName, IMREAD_COLOR);
		//namedWindow("test");
		//imshow("test", frame);
		//waitKey(0);
		if (frame.empty()) {
			cout << "--(!) No captured frame -- Break!\n";
			break;
		}
		resize(frame, frame, Size(frame.cols * 3, frame.rows * 3));
 		Mat frameOrg = frame.clone();
		detectAndDisplay(frame);
		waitKey(0);
		cvDestroyWindow(imageName.c_str());
	}
	counterFile.open("counterFile.txt");
	counterFile << imgCounter;
	counterFile.close();
	for (size_t i = 0; i < avgSizeLst.size(); i++)
	{
		logFile << "Average size: " + to_string(avgSizeLst[i]) << endl;
	}
	cap.release();
	return 0;
}


static int PtDistance(const Point& center1, const Point& center2)
{
	float deltaX = center2.x - center1.x;
	float deltaY = center2.y - center1.y;

	return sqrt(pow(deltaX, 2) + pow(deltaY, 2));
}

bool getImgFromCam(VideoCapture& cap, int & imgCounter, string& path)
{
	//VideoCapture cap(1); // open the default camera
	//if (!cap.isOpened()) {
	//	cout << "Default camera was unable to be opened with VideoCapture(0)" << endl;
	//	return false;
	//}

	namedWindow("cam preview", WINDOW_NORMAL);
	for (;;) {
		Mat camFrame;
		cap >> camFrame;  // get a new frame from camera
		imshow("cam preview", camFrame);
		int i = waitKey(1);
		if (i == 27) {
			// escape key pressed
			cout << "Escape hit, closing ..." << endl;
			return false;
		}
		else if(i > 0) {
			// some other button pressed
			imgCounter++;
			string imgName = path + "/dist" + to_string(imgCounter) + ".png";
			imwrite(imgName, camFrame);
			cout << imgName + " written!" << endl;
			destroyWindow("cam preview");
			break;
		}
	}
	//cap.release();
		
	return true;
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

	//sort goodHoles vector by numDetections largest to smallest
	std::sort(goodHoles.begin(), goodHoles.end(), detectNumSort);

	// detect pitch circle and throw out holes out of pcd
	pcd = getCircle(goodHoles);
	d = AverageDia(goodHoles);
	

	// count number of near neighbors
	float r = 1.5 * float(d) / 2;
	int id = 0;
	for (size_t j = 0; j < goodHoles.size(); j++)
	{
		if (goodHoles[j].valid) {
			for (size_t k = 0; k < goodHoles.size(); k++)
			{

				if (j != k && goodHoles[k].valid) {
					int distance = PtDistance(goodHoles[j].center, goodHoles[k].center);
					if (distance < r) {
						if (goodHoles[k].numDetections <= goodHoles[j].numDetections) {
							goodHoles[k].valid = false;
						}
						else {
							goodHoles[j].valid = false;
						}						
					}
				}
			}
		}
	}


	//// if near neighbors < 2 then probably not a valid hole
	//for (size_t i = 0; i < goodHoles.size(); i++)
	//{
	//	if (goodHoles[i].nearNeighbors < 2 && goodHoles[i].numDetections < 10) {
	//		goodHoles[i].valid = false;
	//	}
	//}
	return goodHoles;
}

void detectAndDisplay(Mat frame)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	cvtColor(frame, frame, COLOR_BGR2GRAY);
	cvtColor(frame, frame, CV_GRAY2RGB);
	equalizeHist(frame_gray, frame_gray);
	Mat frame_copy = frame.clone();

	//-- Detect circuit holes in distributor
	std::vector<Rect> circuits;
	std::vector<int> numDetections;
	circuit_cascade.detectMultiScale(frame_gray, circuits, numDetections, 1.05, 1, 0, Size(20, 20), Size(150, 150));   // was (50,50) and (300, 300)

	vector<Hole> circuitHoles = GetGoodHoles(circuits, numDetections);
	//int averageDia = AverageDia(circuitHoles);

	// ---------------- Display all detections in separate window for testing -----------------------------------

	/*for (size_t i = 0; i < circuitHoles.size(); i++)
	{
		ellipse(frame_copy, circuitHoles[i].center, Size(circuitHoles[i].rect.width / 2, circuitHoles[i].rect.height / 2), 0, 0, 360, Scalar(201, 30, 87), 3);
		String textBox = to_string(circuitHoles[i].numDetections);
		Size textSize = getTextSize(textBox, FONT_HERSHEY_SIMPLEX, 1, 8, 0);
		Point textLoc = Point(circuitHoles[i].center.x - textSize.width, circuitHoles[i].center.y + (textSize.height / 2));
		putText(frame_copy, textBox, textLoc, FONT_HERSHEY_SIMPLEX, 1, Scalar(51, 153, 45), 3);
	}
	namedWindow("Original Detections", 1);
	imshow("Original Detections", frame_copy);*/

	// ---------------- End Test ------------------------------------------------------------------------------------

	int holeCount = 0;
	

	// draw pitch circle
	Point pitchCircleCenter = Point(pcd.x, pcd.y);
	cout << "(h, k, r) = (" + to_string(pitchCircleCenter.x) + ", " + to_string(pitchCircleCenter.y) + ", " + to_string(pcd.r) + ")\n";
	cv::circle(frame, pitchCircleCenter, 50, Scalar(158, 244, 66), 2);  //thickness was 8
	ellipse(frame, pitchCircleCenter, Size(pcd.r, pcd.r), 0, 0, 360, Scalar(158, 244, 66), 1); // was 8

	orderByQuad(circuitHoles, pcd);

	for (size_t i = 0; i < circuitHoles.size(); i++)
	{
		if (circuitHoles[i].valid) {
			holeCount++;
			cout << "id " + to_string(holeCount) + ": ";
			cout << "(" + to_string(circuitHoles[i].center.x) + ", " + to_string(circuitHoles[i].center.y) + ", " + to_string(circuitHoles[i].rect.width)+")\n";
			//logFile << "id " + to_string(holeCount) + ": ";
			//logFile << "(" + to_string(circuitHoles[i].center.x) + ", " + to_string(circuitHoles[i].center.y) + ")\n";
			String textBox = " " + to_string(holeCount);
			Size textSize = getTextSize(textBox, FONT_HERSHEY_SIMPLEX, 1, 2, 0);
			Point textLoc = Point(circuitHoles[i].center.x - (textSize.width/2), circuitHoles[i].center.y + (textSize.height / 2));
			//Point textLoc = GetTextPoint(circuitHoles[i], pcd);
			putText(frame, textBox, textLoc, FONT_HERSHEY_SIMPLEX, 1, Scalar(51, 153, 45), 2);
			ellipse(frame, circuitHoles[i].center, Size(circuitHoles[i].rect.width / 2, circuitHoles[i].rect.height / 2), 0, 0, 360, Scalar(255, 0, 255), 3); //was 20
		}

	}

	//putText(frame, imageName, Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(51, 153, 45), 4, 8, false);
	rectangle(frame,Point(10,10),Point(500, 125),Scalar(255, 351, 249), FILLED);
	int expectedHoleCount = estimateHoleCount(circuitHoles, pcd);
	if (holeCount != expectedHoleCount) {
		rectangle(frame, Point(50, frame.rows - 100), Point(frame.cols - 50, frame.rows), Scalar(66, 244, 226), FILLED);
		putText(frame, "CAUTION: detected count != estimated hole count (" + to_string(estimateHoleCount(circuitHoles, pcd)) + ")", Point(50, frame.rows - 50), FONT_HERSHEY_DUPLEX, 1, Scalar(9, 43, 40), 2);
	}
	putText(frame, "Expected Hole Count = " + to_string(estimateHoleCount(circuitHoles, pcd)), Point(20, 150), FONT_HERSHEY_SIMPLEX, 1, Scalar(51, 153, 45), 2, 8, false);
	putText(frame, "Detected Hole Count = " + to_string(holeCount), Point(20, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(51, 153, 45), 2, 8, false);
	putText(frame, "Detected Hole Size = " + getHoleSize(d), Point(20, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(51, 153, 45), 2, 8, false);
	resize(frame, frame, Size(frame.cols / 1.5, frame.rows / 1.5));
	namedWindow(imageName, CV_WINDOW_AUTOSIZE);
	imshow(imageName, frame);
}

int AverageDia(vector<Hole>& holeLst) {
	int sum = 0;
	int count = 0;
	// holeLst should be sorted by numDetections
	
	for (size_t i = 0; i < holeLst.size(); i++)
	{
		if (i < 6) {
			sum += holeLst[i].rect.width;
			count++;
		}
		else {
			break;
		}

	}

	//for (iter = holeLst.begin() + holeLst.size() / 3; iter != (holeLst.end() - (holeLst.size() / 3)); iter++) {
	//	//sum += holeLst[i].rect.width;
	//	if (iter->valid && iter->numDetections > 10 &&) {
	//		sum += iter->rect.width;
	//		count++;
	//	}
	//}

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

bool detectNumSort(Hole i, Hole j) {
	return (i.numDetections > j.numDetections);
}

circle_fit::circle_t getCircle(vector<Hole> &holeLst)
{
	circle_fit::points_t points;
	int count = 0;
	int d = AverageDia(holeLst);
	for (int i = 0; i < holeLst.size(); i++) {
		if (count > 5) {
			break;
		}
		else {
			if (holeLst[i].rect.width < (d+20)) {
				points.push_back(circle_fit::point_t(holeLst[i].center.x, holeLst[i].center.y));
				count++;
			}
		}
		/*if (holeLst[i].valid && holeLst[i].numDetections > 10) {
			points.push_back(circle_fit::point_t(holeLst[i].center.x, holeLst[i].center.y));
		}*/
		//cout << "(" + to_string(holeLst[i].center.x) + ", " + to_string(holeLst[i].center.y) + ") \n";
	}
	circle_fit::circle_t circle = circle_fit::fit(points);
	for (size_t i = 0; i < holeLst.size(); i++)
	{
		float pcr = circle.r;
		if (holeLst[i].valid) {
			float calcRad = sqrt(pow(holeLst[i].center.x - circle.x, 2) + pow(holeLst[i].center.y - circle.y, 2));
			float ratio = abs(calcRad / pcr);
			//cout << "ratio: " + to_string(ratio) << endl;
			if (!(.85 < ratio && ratio < 1.15)) {
				holeLst[i].valid = false;
			}
		}

	}

	return circle;
}

string getHoleSize(const int& d) {
	
	if (40 < d && d < 90) {
		return "1/4";
	}
	else
	{
		return "unknown";
	}
}

void orderByQuad(vector<Hole>& goodHoles, circle_fit::circle_t& pcd) {

	for (size_t i = 0; i < goodHoles.size(); i++)
	{
		if (goodHoles[i].valid) {
			// q1
			if ((goodHoles[i].center.x >= pcd.x) && (goodHoles[i].center.y <= pcd.y)) {
				goodHoles[i].quad = 1;
			}
			else if ((goodHoles[i].center.x < pcd.x) && (goodHoles[i].center.y < pcd.y)) {
				goodHoles[i].quad = 2;
			}
			else if ((goodHoles[i].center.x <= pcd.x) && (goodHoles[i].center.y > pcd.y)) {
				goodHoles[i].quad = 3;
			}
			else if ((goodHoles[i].center.x > pcd.x) && (goodHoles[i].center.y > pcd.y)) {
				goodHoles[i].quad = 4;
			}
		}
	}

	std::sort(goodHoles.begin(), goodHoles.end(), sortQuad);	
}

int estimateHoleCount(vector<Hole>& goodHoles, circle_fit::circle_t& circle) {
	// good holes should be sorted by quadrant before running this function
	if (goodHoles.size() == 0) {
		return 0;
	}
	vector<int> guesses;
	float pi = 3.14159265;
	float h = circle.x;
	float r = circle.r;
	float theta1 = std::acos((goodHoles[0].center.x - h) / r) * (180 / pi);
	float theta2, delta;
	float sum=0, count=1;
	bool initialized = false;

	for (size_t i = 0; i < goodHoles.size(); i++)
	{
		if (goodHoles[i].valid) {
			if (!initialized) {
				theta1 = std::acos((goodHoles[i].center.x - h) / r) * (180 / pi);
				initialized = true;
			}
			else {
				theta2 = acos((goodHoles[i].center.x - h) / r) * (180 / pi);
				delta = theta2 - theta1;
				guesses.push_back((int)abs(round(360 / delta)));
				theta1 = theta2;
			}
			
		}
	}
	std::unordered_map<int, int> dict;
	int commonest;
	int maxcount = 0;
	for (int i : guesses) {
		if (++dict[i] > maxcount) {
			commonest = i;
			maxcount = dict[i];
		}
	}

	return commonest;
}

bool sortQuad(Hole i, Hole j) {
	if (j.quad > i.quad) {
		return true;
	}
	else if (j.quad < i.quad) {
		return false;
	}

	if (j.quad == i.quad) {
		if (j.quad == 1 || j.quad == 4) {
			return(j.center.y < i.center.y);
		}
		else if (j.quad == 2 || j.quad == 3) {
			return(j.center.y > i.center.y);
		}
		else {
			return false;
		}
	}
}

Point GetTextPoint(Hole& hole, circle_fit::circle_t& circle) {
	float theta;
	if (hole.quad == 1 || hole.quad == 2) {
		theta = acos((hole.center.x - circle.x) / circle.r);
	}
	else {
		theta = -acos((hole.center.x - circle.x) / circle.r);
	}

	int x = circle.x + (circle.r + 10)*cos(theta);
	int y = circle.y + (circle.r + 10) *sin(theta);

	return Point(x, y);
	
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