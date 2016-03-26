/*
* File: Dominant_Plane_Tracking.cpp
* Description: This is a program to track a dominant plane
* Created on: Feb 15, 2016
* Authors: Vikrant More, Arun Suresh and Chandrahas Jagadish Ramalad
*/

/* Includes */
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <iostream>
#include <vector>

/* Defines */
#define THRESHOLD		25
#define CANNY_THRESHOLD 2
//#define ENABLE_ALL_IMSHOW

/* Global Variables */
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
vector<Point> ROICordinates;
bool Roi = false;

int erosion_elem = 2;
int erosion_size = 1;
int dilation_elem = 1;
int dilation_size = 2;
int const max_elem = 2;

/* Static function prototypes */
static void MouseClicked(int event, int x, int y, int, void*);

//Converts match points to xy points
void ConvertMatchesToPoints(const vector<KeyPoint>& train, const vector<KeyPoint>& query,
					const std::vector<cv::DMatch>& matches, std::vector<cv::Point2f>& pts_train,
					std::vector<Point2f>& pts_query)
{
	pts_train.clear();
	pts_query.clear();
	pts_train.reserve(matches.size());
	pts_query.reserve(matches.size());

	size_t i = 0;

	for (; i < matches.size(); i++)
	{
		const DMatch & dmatch = matches[i];

		pts_query.push_back(query[dmatch.queryIdx].pt);
		pts_train.push_back(train[dmatch.trainIdx].pt);
	}
}

static void MouseClicked(int event, int x, int y, int, void*)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		ROICordinates.push_back(Point(x, y));
	}

	else if (event == EVENT_LBUTTONUP)
	{
		ROICordinates.push_back(Point(x, y));
		Roi = true;
	}
}

int main(int ac, char ** av)
{
	/* Local Variables */
	bool FirstTime = true;
	Mat imageOne, imageTwo, ImageOne_Desc, ImageTwo_Desc, HomoGraphy;
	VideoCapture Video(av[1]);
	Ptr<SIFT> detector = SIFT::create(400);
	FlannBasedMatcher flann_matcher;
	vector<DMatch> matches;
	vector<Point2f> ImageTwo_Pts, ImageOne_Pts;
	vector<KeyPoint> ImageTwo_Kpts, ImageOne_Kpts, ImageOne_Kpts_Roi, ImageKpts;
	vector<unsigned char> match_mask;

	int erosion_type = 0;
	if (erosion_elem == 0) { erosion_type = MORPH_RECT; }
	else if (erosion_elem == 1) { erosion_type = MORPH_CROSS; }
	else if (erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

	int dilation_type = 0;
	if (dilation_elem == 0) { dilation_type = MORPH_RECT; }
	else if (dilation_elem == 1) { dilation_type = MORPH_CROSS; }
	else if (dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

	/* Check if opening the video was a success */
	if (!Video.isOpened())
	{
		cout << "Video failed" << endl;
		return -1;
	}

	namedWindow("Select ROI", 0);
	setMouseCallback("Select ROI", MouseClicked, NULL);

	Video >> imageOne;

	if (ac != 2 || !imageOne.data) {
		cout << "No image data" << endl;
		return -1;
	}

	//downsample image
	pyrDown(imageOne, imageOne, Size(imageOne.cols / 2, imageOne.rows / 2));
	pyrDown(imageOne, imageOne, Size(imageOne.cols / 2, imageOne.rows / 2));

	imshow("Select ROI", imageOne);


	Mat elementErode = getStructuringElement(erosion_type,
											 Size(2 * erosion_size + 1, 2 * erosion_size + 1),
											 Point(erosion_size, erosion_size));


	Mat elementDilate = getStructuringElement(dilation_type,
											  Size(2 * dilation_size + 1, 2 * dilation_size + 1),
											  Point(dilation_size, dilation_size));


	for (;;)
	{
		if (Roi)
		{
			for (int k = 0; k < 20; k++)
			{
				Video >> imageTwo;
				if (imageTwo.empty())
				{
					cout << "End of Video" << endl;
					return -1;
				}
			}

			pyrDown(imageTwo, imageTwo, Size(imageTwo.cols / 2, imageTwo.rows / 2));
			pyrDown(imageTwo, imageTwo, Size(imageTwo.cols / 2, imageTwo.rows / 2));

			Mat grayImageOne, grayImageTwo;
			cvtColor(imageOne, grayImageOne, COLOR_BGR2GRAY);
			cvtColor(imageTwo, grayImageTwo, COLOR_BGR2GRAY);

			if (FirstTime)
			{
				detector->detect(grayImageOne, ImageOne_Kpts); //Find interest points
				for (int i = 0; i < ImageOne_Kpts.size(); i++)
				{
					if ((ImageOne_Kpts[i].pt.x < ROICordinates[1].x) && (ImageOne_Kpts[i].pt.x > ROICordinates[0].x) && (ImageOne_Kpts[i].pt.y < ROICordinates[1].y) && (ImageOne_Kpts[i].pt.y > ROICordinates[0].y))
					{
						ImageOne_Kpts_Roi.push_back(ImageOne_Kpts[i]);
					}
				}
				detector->compute(grayImageOne, ImageOne_Kpts_Roi, ImageOne_Desc);
				FirstTime = 0;
			}
			else {
				ImageOne_Kpts_Roi.clear();
				ImageOne_Kpts_Roi = ImageKpts;// ImageTwo_Kpts;
				detector->compute(grayImageOne, ImageOne_Kpts_Roi, ImageOne_Desc);
			}

			Mat imageOneCorner;
			imageOne.copyTo(imageOneCorner);
			drawKeypoints(imageOneCorner, ImageOne_Kpts_Roi, imageOneCorner, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_OVER_OUTIMG);

			detector->detect(grayImageTwo, ImageTwo_Kpts); //Find interest points
			detector->compute(grayImageTwo, ImageTwo_Kpts, ImageTwo_Desc); //Compute SIFT descriptors
			Mat imageTwoCorner;
			imageTwo.copyTo(imageTwoCorner);
			drawKeypoints(imageTwoCorner, ImageTwo_Kpts, imageTwoCorner, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_OVER_OUTIMG);

			imshow("Points1", imageOneCorner);
			imshow("Points2", imageTwoCorner);
#ifdef ENABLE_ALL_IMSHOW
			imshow("Points1", imageOneCorner);
			imshow("Points2", imageTwoCorner);
#endif
			if (!ImageTwo_Kpts.empty())
			{
				flann_matcher.match(ImageOne_Desc, ImageTwo_Desc, matches, Mat());

				ConvertMatchesToPoints(ImageTwo_Kpts, ImageOne_Kpts_Roi, matches, ImageTwo_Pts, ImageOne_Pts);

				ImageKpts.clear();
				for (int i = 0; i < matches.size(); i++)
				{
					const DMatch & dmatch = matches[i];
					ImageKpts.push_back(ImageTwo_Kpts[dmatch.trainIdx]);
				}

				if (matches.size() > 5)
				{
					HomoGraphy = findHomography(ImageOne_Pts, ImageTwo_Pts, RANSAC, 4, match_mask);

#ifdef ENABLE_ALL_IMSHOW
					imshow("REL", imageOneCorner);
#endif
					Mat dst, diffMat, grayDst, grayImage2;
					warpPerspective(imageOne, dst, HomoGraphy, dst.size());

					imshow("WARP", dst);
					cvtColor(dst, grayDst, CV_BGR2GRAY);
					cvtColor(imageTwo, grayImage2, CV_BGR2GRAY);

					absdiff(grayDst, grayImage2, diffMat);

					Mat binMat;
					diffMat.copyTo(binMat);
					int Count = 0;

					for (int i = 0; i < binMat.rows; i++)
					{
						for (int j = 0; j < binMat.cols; j++)
						{
							if ((int)diffMat.at<uchar>(i, j) > THRESHOLD)
							{
								Count++;
								binMat.at<uchar>(i, j) = 255;
							}

							else
							{
								binMat.at<uchar>(i, j) = 0;
							}
						}
					}

#ifdef ENABLE_ALL_IMSHOW
					imshow("DIFF", diffMat);
					imshow("BIN", binMat);
#endif

					Mat er;
					dilate(binMat, er, elementDilate);
					erode(er, er, elementErode);
					erode(er, er, elementErode);
					erode(er, er, elementErode);
					dilate(er, er, elementDilate);
					dilate(er, er, elementDilate);
#ifdef ENABLE_ALL_IMSHOW
					imshow("er before contours", er);
#endif
					// threshold specifying minimum area of a blob
					double contour_threshold = 10000;

					vector<vector<Point>> contours;
					vector<Vec4i> hierarchy;
					vector<int> small_blobs;
					double contour_area;
					Mat addweight;
					// find all contours in the binary image
					Mat dstcan, canny_edges;
					Canny(er, canny_edges, CANNY_THRESHOLD, CANNY_THRESHOLD * 2, 3);


					addWeighted(er, 1.0, canny_edges, 1.0, 0.0, addweight); // blend src image with canny image

					Mat duplicate;
					addweight.copyTo(duplicate);
					findContours(duplicate, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

					// Find indices of contours whose area is less than `threshold`
					if (!contours.empty()) {
						for (int i = 0; i < contours.size(); ++i) {
							contour_area = contourArea(contours[i]);
							if (contour_area < contour_threshold)
								small_blobs.push_back(i);
						}
					}

					Mat final_image;
					addweight.copyTo(final_image);

					// fill-in all small contours with white color
					for (size_t i = 0; i < small_blobs.size(); ++i) {
						drawContours(final_image, contours, small_blobs[i], Scalar(255),
									 CV_FILLED, 8);

					}

#ifdef ENABLE_ALL_IMSHOW
					imshow("final_image", final_image);
#endif
					// Fill blue color in planar region
					Mat DominantPlane;
					dst.copyTo(DominantPlane);
					for (int i = 0; i < final_image.rows; i++)
					{
						for (int j = 0; j < final_image.cols; j++)
						{
							if ((int)final_image.at<uchar>(i, j) == 0)
							{
								DominantPlane.at<Vec3b>(i, j)[0] = 255;
								DominantPlane.at<Vec3b>(i, j)[1] = 0;
								DominantPlane.at<Vec3b>(i, j)[2] = 0;
							}
						}
					}

					imshow("Dominant Plane", DominantPlane);

#ifdef ENABLE_ALL_IMSHOW
					Mat RGB_img = Mat(final_image.rows, final_image.cols, CV_8UC3);
									Mat R_channel = Mat::zeros(final_image.rows, final_image.cols, CV_8UC1);
									Mat B_channel = 255 - final_image;
									Mat G_channel = Mat::zeros(final_image.rows, final_image.cols, CV_8UC1);
									vector<Mat> channels;
									channels.push_back(B_channel);
									channels.push_back(G_channel);
									channels.push_back(R_channel);
									merge(channels, RGB_img);

									for (int i = 0; i < RGB_img.rows; i++)
									{
										for (int j = 0; j < RGB_img.cols; j++)
										{
											if ((RGB_img.at<Vec3b>(i, j)[0] == 0) && (RGB_img.at<Vec3b>(i, j)[1] == 0) && (RGB_img.at<Vec3b>(i, j)[2] == 0))
											{
												dst.at<Vec3b>(i, j)[0] = 255;
												dst.at<Vec3b>(i, j)[1] = 255;
												dst.at<Vec3b>(i, j)[2] = 255;

											}
										}
									}

					imshow("NON-PLANAR SURFACES", dst);
#endif
				}
				else
				{
					cout << "No Proper Matches found" << endl;
					return -1;
				}
			}
			imageTwo.copyTo(imageOne);
		}


		char s = (char)waitKey(30);
		switch (s) {
			case 'q':
				return 0;
		}

	}
	return 0;
}
