//
//  filter.hpp
//  image_filter
//
//  Created by Kristine  Umeh on 1/22/23.
//

#ifndef filter_hpp
#define filter_hpp

#include <stdio.h>

#endif /* filter_hpp */

//include function signatures

int invertFilter(cv::Mat &src, cv::Mat &dest);

int greyFilter(cv::Mat &src, cv::Mat &dest);

int blur5x5Filter(cv::Mat &src, cv::Mat &dest);

int sepiaFilter(cv::Mat &src, cv::Mat &dest);

int watermarkFilter(cv::Mat &src, int row_min, int row_max, int col_min, int col_max);

int sobelX3x3Filter(cv::Mat &src, cv::Mat3s &dest);

int sobelY3x3Filter(cv::Mat &src, cv::Mat3s &dest);

int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dest);

int blurQuantizeFilter(cv::Mat &src, cv::Mat &dest, int levels);

int cartoon(cv::Mat &src, cv::Mat &dest, int levels, int magThreshold);

int thresholdFilter(cv::Mat &src, cv::Mat &dest);
