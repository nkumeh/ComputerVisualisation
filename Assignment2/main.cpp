//////
//////  main.cpp
//////  pattern_match
//////
//////  Created by Kristine  Umeh on 2/3/23.
//////
////
////
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/opencv.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/objdetect.hpp>
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//
//
//
////HOG method
//int main(int argc, char** argv)
//{
//    // Load the two images to be matched
//    Mat image1 = imread("/Users/kristineumeh/Desktop/GradSchool/CompViz/Assignment2/similar/pic.1012.jpg", IMREAD_GRAYSCALE);
//    Mat image2 = imread("/Users/kristineumeh/Desktop/GradSchool/CompViz/Assignment2/similar/pic.1011.jpg", IMREAD_GRAYSCALE);
//
//    // Check that the images were loaded correctly
//    if (image1.empty() || image2.empty())
//    {
//        cout << "Could not load the images" << endl;
//        return -1;
//    }
//
//    // Set the parameters for the HOG descriptor
//    HOGDescriptor hog;
//    hog.winSize = Size(64, 64);
//
//    // Compute the HOG features and descriptors for the two images
//    vector<float> features1, features2;
//    hog.compute(image1, features1, Size(32, 32), Size(0, 0));
//    hog.compute(image2, features2, Size(32, 32), Size(0, 0));
//
//    // Calculate the Euclidean distance between the two feature vectors
//    float distance = norm(Mat(features1), Mat(features2), NORM_L2);
//
//    // Check if the two images match based on the distance threshold
//    if (distance < 50.0)
//    {
//        cout << "The two images match!" << endl;
//    }
//    else
//    {
//        cout << "The two images do not match." << endl;
//    }
//
//    return 0;
//}
