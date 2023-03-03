//
//  filter.hpp
//  pattern_match
//
//  Created by Kristine  Umeh on 2/9/23.
//

#ifndef filter_hpp
#define filter_hpp

#include <stdio.h>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

namespace fltr {

    /*
    * Applies a 5x5 Gaussian filter to the given image.
    * param src: original color image
    * param dst: blurred color image
    * return: 0, on success
    */
    int blur5x5(cv::Mat& src, cv::Mat& dst);

    /*
    * Applies a 3x3 Sobel X filter to the given image.
    * param src: original color image
    * param dst: filtered image, typed 16S (signed short)
    * return: 0, on success
    */
    int sobelX3x3(cv::Mat& src, cv::Mat& dst);

    /*
    * Applies a 3x3 Sobel Y filter to the given image.
    * param src: original color image
    * param dst: filtered image, typed 16S (signed short)
    * return: 0, on success
    */
    int sobelY3x3(cv::Mat& src, cv::Mat& dst);

    /*
    * Generates a gradient magnitude image from the X and Y Sobel images.
    * Uses Euclidean distance for magnitude: I = sqrt( sx*sx + sy*sy )
    * param sx: sobel x image, typed 16SC3
    * param sy: sobel y image, typed 16SC3
    * param dst: gradient magnitude image, typed 8UC3
    * return: 0, on success
    */
    int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst);

    /*
    * Generates a gradient orientation image from the X and Y Sobel images.
    * Uses orientation = arctan( sy / sx ), units in degrees
    * param sx: sobel x image, typed 16SC3
    * param sy: sobel y image, typed 16SC3
    * param dst: gradient orientation image, typed 8UC3
    * return: 0, on success
    */
    int orientation(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst);

}

#endif /* filter_hpp */
