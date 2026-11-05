#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <cstdio>

#include "kcftracker.hpp"

using namespace std;

static bool readPPM(const string &path, kcf::Image8 &img)
{
	ifstream f(path.c_str(), ios::binary);
	if (!f)
		return false;

	string magic;
	f >> magic;
	if (magic != "P6")
		return false;

	int vals[3], n = 0;
	while (n < 3)
	{
		f >> ws;
		if (f.peek() == '#')
		{
			string skip;
			getline(f, skip);
			continue;
		}
		if (!(f >> vals[n]))
			return false;
		n++;
	}
	int w = vals[0], h = vals[1], maxv = vals[2];
	if (w <= 0 || h <= 0 || maxv != 255)
		return false;
	f.get(); // one whitespace after maxval

	img.w = w;
	img.h = h;
	img.ch = 3;
	img.d.resize((size_t)w * h * 3);
	f.read((char *)img.d.data(), (streamsize)img.d.size());
	if ((size_t)f.gcount() != img.d.size())
		return false;

	for (size_t i = 0; i < img.d.size(); i += 3) // RGB -> BGR, like imread
		swap(img.d[i], img.d[i + 2]);
	return true;
}

int main(int argc, char *argv[])
{

	if (argc > 5)
		return -1;

	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = true;
	bool LAB = false;

	for (int i = 0; i < argc; i++)
	{
		if (strcmp(argv[i], "hog") == 0)
			HOG = true;
		if (strcmp(argv[i], "fixed_window") == 0)
			FIXEDWINDOW = true;
		if (strcmp(argv[i], "singlescale") == 0)
			MULTISCALE = false;
		if (strcmp(argv[i], "show") == 0)
			SILENT = false;
		if (strcmp(argv[i], "lab") == 0)
		{
			LAB = true;
			HOG = true;
		}
		if (strcmp(argv[i], "gray") == 0)
			HOG = false;
	}

	// Create KCFTracker object
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

	// Frame readed
	kcf::Image8 frame;

	// Tracker results
	kcf::Rect result;

	// Path to list.txt
	ifstream listFile;
	string fileName = "images.txt";
	listFile.open(fileName);

	// Read groundtruth for the 1st frame
	ifstream groundtruthFile;
	string groundtruth = "region.txt";
	groundtruthFile.open(groundtruth);
	string firstLine;
	getline(groundtruthFile, firstLine);
	groundtruthFile.close();

	istringstream ss(firstLine);

	// Read groundtruth like a dumb
	float x1, y1, x2, y2, x3, y3, x4, y4;
	char ch;
	ss >> x1;
	ss >> ch;
	ss >> y1;
	ss >> ch;
	ss >> x2;
	ss >> ch;
	ss >> y2;
	ss >> ch;
	ss >> x3;
	ss >> ch;
	ss >> y3;
	ss >> ch;
	ss >> x4;
	ss >> ch;
	ss >> y4;

	// Using min and max of X and Y for groundtruth rectangle
	float xMin = min(x1, min(x2, min(x3, x4)));
	float yMin = min(y1, min(y2, min(y3, y4)));
	float width = max(x1, max(x2, max(x3, x4))) - xMin;
	float height = max(y1, max(y2, max(y3, y4))) - yMin;

	// Read Images
	ifstream listFramesFile;
	string listFrames = "images.txt";
	listFramesFile.open(listFrames);
	string frameName;

	// Write Results
	ofstream resultsFile;
	string resultsPath = "output.txt";
	resultsFile.open(resultsPath);

	// Frame counter
	int nFrames = 0;

	while (getline(listFramesFile, frameName))
	{
		if (!frameName.empty() && frameName[frameName.size() - 1] == '\r')
			frameName.erase(frameName.size() - 1); // Windows line endings
		if (frameName.empty())
			continue;

		if (!readPPM(frameName, frame))
		{
			fprintf(stderr, "cannot read frame: '%s'\n", frameName.c_str());
			return -1;
		}

		if (nFrames == 0)
		{
			kcf::Rect roi;
			roi.x = (int)xMin;
			roi.y = (int)yMin;
			roi.width = (int)width;
			roi.height = (int)height;
			tracker.init(roi, frame);
			resultsFile << xMin << "," << yMin << "," << width << "," << height << endl;
		}
		else
		{
			result = tracker.update(frame);
			resultsFile << result.x << "," << result.y << ","
						<< result.width << "," << result.height << endl;
		}
		nFrames++;
	}
	resultsFile.close();

	listFile.close();
}
