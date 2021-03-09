#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace std;
using namespace cv;

int main()
{
        Mat I1 = imread("../IMG_0045.JPG", IMREAD_GRAYSCALE);
	Mat I2 = imread("../IMG_0046.JPG", IMREAD_GRAYSCALE);
	// Mat I2 = imread("../IMG_0046r.JPG", IMREAD_GRAYSCALE);

	imshow("I1", I1);
	imshow("I2", I2);

	// the goal of this lab is to create a panorama creation pipeline from 2 images, using opencv functions
	// that is thus a higher level lab, where the goal is to check that you understood the lecture concepts
	// that you are able to find and use their implementation in the opencv library and to put
	// in place the full pipeline.
	// the documentation is available at https://docs.opencv.org/master, mostly in the feature2D and calib3d modules.


	// Q1. use an opencv feature extractor and descriptor to detect and compute features on both images
	//Ptr<AKAZE> D = AKAZE::create();
	// ...
	//vector<KeyPoint> m1, m2;
	// ...
	
	//Mat J;
	//drawKeypoints(...
	
	// Q2. use a descriptor matcher, to compute feature correspondences
	//BFMatcher M ...

	// drawMatches ...
	

	// Q3. Organize the matched feature pairs into vectors and estimate an homography using RANSAC
	// and a model reprojection threshold of 3 pixels
	// provide a mask input to draw the inlier matches
	// vector<Point2f> matches1, matches2;
	// Mat H = findHomography(...
	// drawMatches ...

	// Q4. copy I1 to a new (bigger image) K using the identity homography
	// warp I2 to K using the computed homography
	// Mat K(2 * I1.cols, I1.rows, CV_8U);
	// warpPerspective( ...
	// show your panorama !

	// Q5. does it work when the images are rotated so that they are not approximately aligned at first ?
	// make a panorama with the rotated version IMG_0046r.JPG of image IMG_0046.JPG
	
	// Q6. make it work on 2 images of your own
	// submit this cpp file, a screenshot of the labs' panorama, your input files and a screenshot of your own panorama.

	// Q7. extra credits
	// program and submit a panorama with more than 2 images
	// you can use your own or images from https://github.com/holynski/cse576_sp20_hw5/tree/master/pano 
	
	waitKey(0);
	return 0;
}
