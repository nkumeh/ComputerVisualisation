//
//  filter.cpp
//  pattern_match
//
//  Created by Kristine  Umeh on 2/9/23.
//

#include "filter.hpp"
#include <opencv2/core.hpp>
#include <filesystem>
#include <iostream>
#include <math.h>



namespace fltr {

    constexpr auto PI = 3.14159265;

    // CONSTANTS
    static char FLTR_GAUSSIAN_DATA[5] = { 1, 2, 4, 2, 1 };
    static const cv::Mat FLTR_GAUSSIAN_V(5, 1, CV_8SC1, FLTR_GAUSSIAN_DATA);
    static const cv::Mat FLTR_GAUSSIAN_H(1, 5, CV_8SC1, FLTR_GAUSSIAN_DATA);

    static char FLTR_SOBEL_DIF_DATA[3] = { 1, 0, -1 };
    static char FLTR_SOBEL_AVG_DATA[3] = { 1, 2, 1 };
    static const cv::Mat FLTR_SOBEL_X_V(3, 1, CV_8SC1, FLTR_SOBEL_AVG_DATA);
    static const cv::Mat FLTR_SOBEL_X_H(1, 3, CV_8SC1, FLTR_SOBEL_DIF_DATA);
    static const cv::Mat FLTR_SOBEL_Y_V(3, 1, CV_8SC1, FLTR_SOBEL_DIF_DATA);
    static const cv::Mat FLTR_SOBEL_Y_H(1, 3, CV_8SC1, FLTR_SOBEL_AVG_DATA);

    // HELPERS

    /*
    * Convolves an image with a filter.
    * param im: image
    * param fltr: filter, required to be type char and no bigger than image
    * param res: result image, required to has same dimensions and type as im
    * param type: image type
    */
    void convolve(const cv::Mat& im, const cv::Mat& fltr, cv::Mat& res, int type) {

        if (im.rows != res.rows || im.cols != res.cols) {
            throw std::exception("Convolution failed: Miscv::Matched dimensions");
        }

        // initialize res cv::Matrix
        im.copyTo(res);

        // saved calculations
        int fltr_buf_r = fltr.rows / 2;
        int fltr_buf_c = fltr.cols / 2;
        int normFactor = sum(max(fltr, 0))[0];  // sum of positive values

        // loop over im where filter fits
        for (int r = fltr_buf_r; r < im.rows - fltr_buf_r; r++) {
            for (int c = fltr_buf_c; c < im.cols - fltr_buf_c; c++) {

                // line up and apply filter to compute pixel value
                cv::Vec3f sum(0, 0, 0);
                for (int sub_r = r - fltr_buf_r, fltr_r = 0; sub_r <= r + fltr_buf_r; sub_r++, fltr_r++) {
                    for (int sub_c = c - fltr_buf_c, fltr_c = 0; sub_c <= c + fltr_buf_c; sub_c++, fltr_c++) {
                        char fltrVal = fltr.at<char>(fltr_r, fltr_c);
                        cv::Vec3f fltrVec(fltrVal, fltrVal, fltrVal);
                        switch (type)
                        {
                        case CV_8UC3:
                            sum = sum + (fltrVec.mul(im.at<cv::Vec3b>(sub_r, sub_c)));
                            break;
                        case CV_16SC3:
                            sum = sum + (fltrVec.mul(im.at<cv::Vec3s>(sub_r, sub_c)));
                            break;
                        }
                    }
                }

                // normalize
                sum[0] = sum[0] / normFactor;
                sum[1] = sum[1] / normFactor;
                sum[2] = sum[2] / normFactor;

                // set the res pixel
                switch (type)
                {
                case CV_8UC3:
                    res.at<cv::Vec3b>(r, c) = sum;
                    break;
                case CV_16SC3:
                    res.at<cv::Vec3s>(r, c) = sum;
                    break;
                }
            }
        }
    }

    // PRIMARY FUNCTIONS

    /*
    * Applies a 5x5 Gaussian filter to the given image.
    * Implemented using separable filters.
    * param src: original color image, each pixel is a cv::Vec3b
    * param dst: blurred color image, each pixel is a cv::Vec3b
    * return: 0, on success
    */
    int blur5x5(cv::Mat& src, cv::Mat& dst) {
        try {
            // apply horizontal filter (1x5)
            cv::Mat temp(src.size(), CV_8UC3);
            convolve(src, FLTR_GAUSSIAN_H, temp, CV_8UC3);
            // apply vertical filter (5x1)
            convolve(temp, FLTR_GAUSSIAN_V, dst, CV_8UC3);
        }
        catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            return -1;
        }
        return 0;
    }

    /*
    * Applies a 3x3 Sobel X filter to the given image.
    * param src: original color image
    * param dst: filtered image, typed 16S (signed short)
    * return: 0, on success
    */
    int sobelX3x3(cv::Mat& src, cv::Mat& dst) {
        try {
            // set up src copy with proper type
            cv::Mat srcCopy(src.size(), CV_16SC3);
            src.convertTo(srcCopy, CV_16SC3);
            // apply horizontal filter (1x3)
            cv::Mat temp(srcCopy.size(), CV_16SC3);
            convolve(srcCopy, FLTR_SOBEL_X_H, temp, CV_16SC3);
            // apply vertical filter (3x1)
            convolve(temp, FLTR_SOBEL_X_V, dst, CV_16SC3);
        }
        catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            return -1;
        }
        return 0;
    }

    /*
    * Applies a 3x3 Sobel Y filter to the given image.
    * param src: original color image
    * param dst: filtered image, typed 16S (signed short)
    * return: 0, on success
    */
    int sobelY3x3(cv::Mat& src, cv::Mat& dst) {
        try {
            // set up src copy with proper type
            cv::Mat srcCopy(src.size(), CV_16SC3);
            src.convertTo(srcCopy, CV_16SC3);
            // apply horizontal filter (1x3)
            cv::Mat temp(srcCopy.size(), CV_16SC3);
            convolve(srcCopy, FLTR_SOBEL_Y_H, temp, CV_16SC3);
            // apply vertical filter (3x1)
            convolve(temp, FLTR_SOBEL_Y_V, dst, CV_16SC3);
        }
        catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            return -1;
        }
        return 0;
    }

    /*
    * Generates a gradient magnitude image from the X and Y Sobel images.
    * Uses Euclidean distance for magnitude: I = sqrt( sx*sx + sy*sy )
    * param sx: sobel x image, typed 16SC3
    * param sy: sobel y image, typed 16SC3
    * param dst: gradient magnitude image, typed 8UC3
    * return: 0, on success
    */
    int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst) {
        try {
            cv::Mat mag(sx.size(), CV_32FC3);
            cv::Mat sx2 = sx.mul(sx);
            cv::Mat sy2 = sy.mul(sy);
            cv::Mat sum = sx2 + sy2;
            sum.convertTo(mag, CV_32F);
            sqrt(mag, mag);
            mag.convertTo(dst, CV_8U);
        }
        catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            return -1;
        }
        return 0;
    }

    /*
    * Generates a gradient orientation image from the X and Y Sobel images.
    * Uses orientation = arctan( sy / sx ), units in degrees
    * param sx: sobel x image, typed 16SC3
    * param sy: sobel y image, typed 16SC3
    * param dst: gradient orientation image, converted from [-180, 180] to [0, 255], typed 8UC3
    * return: 0, on success
    */
    int orientation(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst) {
        try {
            // intermediate calculations stored as float
            cv::Mat orient(sx.size(), CV_32FC3);
            for (int r = 0; r < sx.rows; r++) {
                for (int c = 0; c < sx.cols; c++) {
                    for (int ch = 0; ch < sx.channels(); ch++) {
                        // compute orientation as arctan, range is [-180, 180] per documentation
                        short x = sx.at<cv::Vec3s>(r, c)[ch];
                        short y = sy.at<cv::Vec3s>(r, c)[ch];
                        orient.at<cv::Vec3f>(r, c)[ch] = (float)(atan2(y, x) * 180 / PI);
                    }
                }
            }
            // convert the [-180, 180] range so that it fits in uchar
            // formula: X' = ((X + 180) * (255)) / (360)
            orient = (orient + 180) * 255 / 360;
            // save to dst
            orient.convertTo(dst, CV_8UC3);
        }
        catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            return -1;
        }
        return 0;
    }

}
