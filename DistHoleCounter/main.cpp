#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <math.h>
#include <iterator>
#include "circlefit.hh"
#include <unordered_map>

namespace patch  //patch to include to_string function in Linux.  Works without in windows.
{
	template < typename T > std::string to_string(const T& n)
	{
		std::ostringstream stm;
		stm << n;/**< /**<  */
		return stm.str();
	}
}


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

CascadeClassifier circuit_cascade;
string imageName;
vector<int> avgSizeLst;
ofstream logFile, counterFile;
circle_fit::circle_t pcd;
int d;

int PtDistance(const Point& center1, const Point& center2)
{
	float deltaX = center2.x - center1.x;
	float deltaY = center2.y - center1.y;

	return sqrt(pow(deltaX, 2) + pow(deltaY, 2));
}

Point GetCenterPoint(Rect& circuit) {
	return Point(circuit.x + circuit.width / 2, circuit.y + circuit.height / 2);
}

bool detectNumSort(Hole i, Hole j) {
	return (i.numDetections > j.numDetections);
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
	int count = 0;
	int d = AverageDia(holeLst);
	for (int i = 0; i < holeLst.size(); i++) {
		if (count > 5) {
			break;
		}
		else {
			if (holeLst[i].rect.width < (d + 20)) {
				points.push_back(circle_fit::point_t(holeLst[i].center.x, holeLst[i].center.y));
				count++;
			}
		}
	}
	circle_fit::circle_t circle = circle_fit::fit(points);
	for (size_t i = 0; i < holeLst.size(); i++)
	{
		float pcr = circle.r;
		if (holeLst[i].valid) {
			float calcRad = sqrt(pow(holeLst[i].center.x - circle.x, 2) + pow(holeLst[i].center.y - circle.y, 2));
			float ratio = abs(calcRad / pcr);
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
	float sum = 0, count = 1;
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
	if (std::isnan(pcd.x)) {
		// if pcd is not a number then
		pcd.x = 1;
		pcd.y = 1;
		pcd.r = 1;
		pcd.s = 1;
		pcd.g = 0;
		pcd.i = 1;
		pcd.j = 1;
		pcd.ok = false;

	}
	d = AverageDia(goodHoles);

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

	return goodHoles;
}

Mat detectHoles(Mat frame)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	cvtColor(frame, frame, COLOR_BGR2GRAY);
	cvtColor(frame, frame, CV_GRAY2RGB);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect circuit holes in distributor
	std::vector<Rect> circuits;
	std::vector<int> numDetections;
	circuit_cascade.detectMultiScale(frame_gray, circuits, numDetections, 1.05, 1, 0, Size(20, 20), Size(150, 150));   // was (50,50) and (300, 300)

	vector<Hole> circuitHoles = GetGoodHoles(circuits, numDetections);
	int holeCount = 0;

	// draw pitch circle
	Point pitchCircleCenter = Point(pcd.x, pcd.y);
	//cout << "(h, k, r) = (" + patch::to_string(pitchCircleCenter.x) + ", " + patch::to_string(pitchCircleCenter.y) + ", " + patch::to_string(pcd.r) + ")\n";
	cv::circle(frame, pitchCircleCenter, 50, Scalar(158, 244, 66), 1);  //thickness was 8
	ellipse(frame, pitchCircleCenter, Size(pcd.r, pcd.r), 0, 0, 360, Scalar(158, 244, 66), 1); // was 8

	orderByQuad(circuitHoles, pcd);

	for (size_t i = 0; i < circuitHoles.size(); i++)
	{
		if (circuitHoles[i].valid) {
			holeCount++;
			//cout << "id " + patch::to_string(holeCount) + ": ";
			//cout << "(" + patch::to_string(circuitHoles[i].center.x) + ", " + patch::to_string(circuitHoles[i].center.y) + ", " + to_string(circuitHoles[i].rect.width) + ")\n";

			String textBox = patch::to_string(holeCount);
			Size textSize = getTextSize(textBox, FONT_HERSHEY_SIMPLEX, .3, 1, 0);
			Point textLoc = Point(circuitHoles[i].center.x - (textSize.width / 2.), circuitHoles[i].center.y + (textSize.height / 2.));

			putText(frame, textBox, textLoc, FONT_HERSHEY_SIMPLEX, .3, Scalar(51, 153, 45), 1);
			ellipse(frame, circuitHoles[i].center, Size(circuitHoles[i].rect.width / 2, circuitHoles[i].rect.height / 2), 0, 0, 360, Scalar(255, 0, 255), 2); //was 20
		}

	}

	rectangle(frame, Point(10, 10), Point(300, 60), Scalar(255, 351, 249), FILLED);
	int expectedHoleCount = estimateHoleCount(circuitHoles, pcd);
	if (holeCount != expectedHoleCount) {
		rectangle(frame, Point(50, frame.rows - 25), Point(frame.cols - 50, frame.rows), Scalar(66, 244, 226), FILLED);
		putText(frame, "CAUTION: detected count != estimated hole count (" + patch::to_string(estimateHoleCount(circuitHoles, pcd)) + ")", Point(50, frame.rows - 15), FONT_HERSHEY_SIMPLEX, .50, Scalar(9, 43, 40), 1);
	}
	putText(frame, "Expected Hole Count = " + patch::to_string(estimateHoleCount(circuitHoles, pcd)), Point(10, 50), FONT_HERSHEY_SIMPLEX, .5, Scalar(51, 153, 45), 1, 3, false);
	putText(frame, "Detected Hole Count = " + patch::to_string(holeCount), Point(10, 25), FONT_HERSHEY_SIMPLEX, .5, Scalar(51, 153, 45), 1, 3, false);
	return frame;
}

void startStream(VideoCapture& cap)
{
	namedWindow("cam preview", WINDOW_NORMAL);
	cv::setWindowProperty("cam preview", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);

	for (;;) {
		Mat camFrame;
		cap >> camFrame;  // get a new frame from camera
		int i = waitKey(1000);
		if (i == 27) {
			// escape key pressed
			std::cout << "Escape hit, closing ..." << endl;
			break;
		}
		else {
			// some other button pressed
			Mat detectedFrame = detectHoles(camFrame);
			imshow("cam preview", detectedFrame);
			if (i == 32) {
				// if space bar is pressed save photo
				ifstream input_file("counterFile.txt");
				double tempVar;
				vector<double> tempVector;
				int imgCounter = 0;

				if (input_file.good()) {
					while (input_file >> tempVar) {
						tempVector.push_back(tempVar);
					}
					if (tempVector.size() > 0) {
						imgCounter = (int)tempVector[0];
					}
				}
				string path = "./images/testImages/webcam/failImg" + patch::to_string(imgCounter) + ".png";
				imwrite(path, camFrame);
				ofstream counterFile;
				imgCounter++;
				counterFile.open("counterFile.txt");
				counterFile << imgCounter;
				counterFile.close();

			}
		}
	}
}


int main()
{
	String circuit_cascade_name = "./cascade.xml";

	//-- 1. Load the cascades
	if (!circuit_cascade.load(circuit_cascade_name))
	{
		cout << "--(!)Error loading distributor cascade\n";
		return -1;
	};

	VideoCapture cap(1); // open the default camera; default = 0
	if (!cap.isOpened()) {
		cout << "Default camera was unable to be opened with VideoCapture(0)" << endl;
		return -1;
	}	
	startStream(cap);
	cap.release();
	return 0;
}














