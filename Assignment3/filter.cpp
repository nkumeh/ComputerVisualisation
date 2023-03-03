//////
//////  filter.cpp
//////  image_filter
//////
//////  Created by Kristine  Umeh on 1/22/23.
//////
//#include <iostream>
//#include <cstdio>
//#include <cstring>
//#include <opencv2/opencv.hpp>
//#include <math.h>
////
//#include "filter.hpp"
////
//
//int invertFilter(cv::Mat &src, cv::Mat &dest){
//
//    //    actual way to copy and allocate data of the same size
//    dest = src.clone();
//
//    printf("The size of the image is: %d %d\n", src.rows, src.cols);
//
//    //negation of image
//    for (int i=0; i < dest.rows; i++){
//
//        for(int j=0; j < dest.cols; j++){
//
//            cv::Vec3b &dptr = dest.at<cv::Vec3b>(i,j);
//
//             dptr[0] = 255 - dptr[0];
//             dptr[1] = 255 - dptr[1];
//             dptr[2] = 255 - dptr[2];
//            
//            
//        }
//
//    }
//    return 0;
//}
//
//int greyFilter(cv::Mat &src, cv::Mat &dest){
//    
//    //    actual way to copy and allocate data of the same size
//    dest = src.clone();
//    
//    printf("The size of the image is: %d %d\n", src.rows, src.cols);
//    
//    //negation of image
//    for (int i=0; i < dest.rows; i++){
//        
//        for(int j=0; j < dest.cols; j++){
//            
//            cv::Vec3b &dptr = dest.at<cv::Vec3b>(i,j);
//
////            take away the pigment by averaging
////            BGR
//            int b = dptr[0]; // blue
//            int g = dptr[1]; //green
//            int r = dptr[2]; // red
//            
//            int avg = (b+g+r)/3;
//            
//            dptr[0] = avg;
//            dptr[1] = avg;
//            dptr[2] = avg;
//            
//        }
//        
//    }
//    watermarkFilter(dest, 50, 100, 50, 100);
//
//    return 0;
//}
//
//
//// custom filter 
//int sepiaFilter(cv::Mat &src, cv::Mat &dest){
//    //    actual way to copy and allocate data of the same size
//    dest = src.clone();
//
//    printf("The size of the image is: %d %d\n", src.rows, src.cols);
//
//    for (int i=0; i < src.rows; i++){
//
//        for(int j=0; j < src.cols; j++){
//
//            cv::Vec3b &dptr = dest.at<cv::Vec3b>(i,j);
//
//            
//            dptr[0] = src.at<cv::Vec3b>(i,j)[0] * 0.189 + src.at<cv::Vec3b>(i,j)[1] * 0.534 + src.at<cv::Vec3b>(i,j)[0] * 0.272;
//            dptr[1] = src.at<cv::Vec3b>(i,j)[0] * 0.168 + src.at<cv::Vec3b>(i,j)[1] * 0.686 + src.at<cv::Vec3b>(i,j)[0] * 0.349;
//            dptr[2] = src.at<cv::Vec3b>(i,j)[0] * 0.131 + src.at<cv::Vec3b>(i,j)[1] * 0.769 + src.at<cv::Vec3b>(i,j)[0] * 0.393;
//
////            dptr[0] = dptr[0] * 0.189 + dptr[1] * 0.534 + dptr[2] * 0.272;
////            dptr[1] = dptr[0] * 0.168 + dptr[1] * 0.686 + dptr[2] * 0.349;
////            dptr[2] = dptr[0] * 0.131 + dptr[1] * 0.769 + dptr[2] * 0.393;
//
//        }
//
//    }
//    return 0;
//}
//
//int watermarkFilter(cv::Mat &src, int row_min, int row_max, int col_min, int col_max){
//    cv::Mat dest = src.clone();
//
//    printf("The size of the image is: %d %d\n", src.rows, src.cols);
//
//    for (int i=row_min; i < row_max; i++){
//
//        for(int j=col_min; j < col_max; j++){
//
//            cv::Vec3b &dptr = dest.at<cv::Vec3b>(i,j);
//            
//            dptr[0] = 0;
//            dptr[1] = 0;
//            dptr[2] = 0;
//
//        }
//
//    }
//    
//    return 0;
//}
//
//
//int blur5x5Filter(cv::Mat &src, cv::Mat &dest){
//    //    actual way to copy and allocate data of the same size
//    cv::Mat temp;
//    
//    temp = src.clone();
//
//    printf("The size of the image is: %d %d\n", src.rows, src.cols);
//    
////  gaussian coeff .... is this blur on?
//    int coeff = 1 + 2 + 4 + 2 + 1;
//
//    for (int i=2; i < src.rows-2; i++){
//        
//        for(int j=0; j < src.cols; j++){
//
//            cv::Vec3b &dptr = temp.at<cv::Vec3b>(i,j);
//
//            dptr[0] = (1 * src.at<cv::Vec3b>(i-2,j)[0] + 2 * src.at<cv::Vec3b>(i-1,j)[0] + 4 * src.at<cv::Vec3b>(i,j)[0] + 2 * src.at<cv::Vec3b>(i+1,j)[0] + 1 * src.at<cv::Vec3b>(i+2,j)[0])/coeff;
//
//            dptr[1] = (1 * src.at<cv::Vec3b>(i-2,j)[1] + 2 * src.at<cv::Vec3b>(i-1,j)[1] + 4 * src.at<cv::Vec3b>(i,j)[1] + 2 * src.at<cv::Vec3b>(i+1,j)[1] + 1 * src.at<cv::Vec3b>(i+2,j)[1])/coeff;
//
//            dptr[2] = (1 * src.at<cv::Vec3b>(i-2,j)[2] + 2 * src.at<cv::Vec3b>(i-1,j)[2] + 4 * src.at<cv::Vec3b>(i,j)[2] + 2 * src.at<cv::Vec3b>(i+1,j)[2] + 1 * src.at<cv::Vec3b>(i+2,j)[2])/coeff;
//
//        }
//    }
//    
//    for (int i=0; i < src.rows; i++){
//        
//        for(int j=2; j < src.cols-2; j++){
//
//            cv::Vec3b &dptr = temp.at<cv::Vec3b>(i,j);
//
//            dptr[0] = (1 * src.at<cv::Vec3b>(i,j-2)[0] + 2 * src.at<cv::Vec3b>(i,j-1)[0] + 4 * src.at<cv::Vec3b>(i,j)[0] + 2 * src.at<cv::Vec3b>(i,j+1)[0] + 1 * src.at<cv::Vec3b>(i,j+2)[0])/coeff;
//
//            dptr[1] = (1 * src.at<cv::Vec3b>(i,j-2)[1] + 2 * src.at<cv::Vec3b>(i,j-1)[1] + 4 * src.at<cv::Vec3b>(i,j)[1] + 2 * src.at<cv::Vec3b>(i,j+1)[1] + 1 * src.at<cv::Vec3b>(i,j+2)[1])/coeff;
//
//            dptr[2] = (1 * src.at<cv::Vec3b>(i,j-2)[2] + 2 * src.at<cv::Vec3b>(i,j-1)[2] + 4 * src.at<cv::Vec3b>(i,j)[2] + 2 * src.at<cv::Vec3b>(i,j+1)[2] + 1 * src.at<cv::Vec3b>(i,j+2)[2])/coeff;
//
//        }
//    }
////    dest = temp.clone();
//    
//    temp.copyTo(dest);
//    
//    return 0;
//}
//
//
//int sobelX3x3Filter(cv::Mat &src, cv::Mat3s &dest){
//    //    actual way to copy and allocate data of the same size
//    cv::Mat3s temp;
//    temp = src.clone();
//    
//    printf("The size of the image is: %d %d\n", src.rows, src.cols);
//    
//    for (int i=1; i < src.rows-1; i++){
//        
//        for(int j=0; j < src.cols; j++){
//            
//            cv::Vec3s &dptr = temp.at<cv::Vec3s>(i,j);
//            
//            dptr[0] = ((1) * src.at<cv::Vec3b>(i-1,j)[0] + (2) * src.at<cv::Vec3b>(i,j)[0] + (1) * src.at<cv::Vec3b>(i+1,j)[0]);
//            
//            dptr[1] = ((1) * src.at<cv::Vec3b>(i-1,j)[1] + (2) * src.at<cv::Vec3b>(i,j)[1] + (1) * src.at<cv::Vec3b>(i+1,j)[1]);
//            
//            dptr[2] = ((1) * src.at<cv::Vec3b>(i-1,j)[2] + (2) * src.at<cv::Vec3b>(i,j)[2] + (1) * src.at<cv::Vec3b>(i+1,j)[2]);
//            
//        }
//    }
//    
//    for (int i=0; i < src.rows; i++){
//        
//        for(int j=1; j < src.cols-1; j++){
//            
//            cv::Vec3s &dptr = dest.at<cv::Vec3s>(i,j);
//            
//            dptr[0] = ((1) * temp.at<cv::Vec3s>(i,j-1)[0] + (0) * temp.at<cv::Vec3s>(i,j)[0] + (-1) * temp.at<cv::Vec3s>(i,j+1)[0]);
//            
//            dptr[1] = ((1) * temp.at<cv::Vec3s>(i,j-1)[1] + (0) * temp.at<cv::Vec3s>(i,j)[1] + (-1) * temp.at<cv::Vec3s>(i,j+1)[1]);
//            
//            dptr[2] = ((1) * temp.at<cv::Vec3s>(i,j-1)[2] + (0) * temp.at<cv::Vec3s>(i,j)[2] + (-1) * temp.at<cv::Vec3s>(i,j+1)[2]);
//            
//        }
//    }
////    temp.copyTo(dest);
//    
//    return 0;
//    
//}
//
//
//
//int sobelY3x3Filter(cv::Mat &src, cv::Mat3s &dest){
//    //    actual way to copy and allocate data of the same size
//    cv::Mat3s temp;
//    
//    temp = src.clone();
//
//    printf("The size of the image is: %d %d\n", src.rows, src.cols);
//    
//    for (int i=1; i < src.rows-1; i++){
//        
//        for(int j=0; j < src.cols; j++){
//            
//            cv::Vec3s &dptr = temp.at<cv::Vec3s>(i,j);
//            
//            dptr[0] = ((1) * src.at<cv::Vec3b>(i-1,j)[0] + (0) * src.at<cv::Vec3b>(i,j)[0] + (-1) * src.at<cv::Vec3b>(i+1,j)[0]);
//            
//            dptr[1] = ((1) * src.at<cv::Vec3b>(i-1,j)[1] + (0) * src.at<cv::Vec3b>(i,j)[1] + (-1) * src.at<cv::Vec3b>(i+1,j)[1]);
//            
//            dptr[2] = ((1) * src.at<cv::Vec3b>(i-1,j)[2] + (0) * src.at<cv::Vec3b>(i,j)[2] + (-1) * src.at<cv::Vec3b>(i+1,j)[2]);
//            
//        }
//    }
//    
//    for (int i=0; i < src.rows; i++){
//        
//        for(int j=1; j < src.cols-1; j++){
//            
//            cv::Vec3s &dptr = dest.at<cv::Vec3s>(i,j);
//            
//            dptr[0] = ((1) * temp.at<cv::Vec3s>(i,j-1)[0] + (2) * temp.at<cv::Vec3s>(i,j)[0] + (1) * temp.at<cv::Vec3s>(i,j+1)[0]);
//            
//            dptr[1] = ((1) * temp.at<cv::Vec3s>(i,j-1)[1] + (2) * temp.at<cv::Vec3s>(i,j)[1] + (1) * temp.at<cv::Vec3s>(i,j+1)[1]);
//            
//            dptr[2] = ((1) * temp.at<cv::Vec3s>(i,j-1)[2] + (2) * temp.at<cv::Vec3s>(i,j)[2] + (1) * temp.at<cv::Vec3s>(i,j+1)[2]);
//            
//        }
//    }
//    
//    return 0;
//  
//}
//
//int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dest){
//
//    for (int i=0; i < sx.rows; i++){
//
//        for(int j=0; j < sy.cols; j++){
//            
//            cv::Vec3b &dptr = dest.at<cv::Vec3b>(i,j);
//
//            dptr[0] = sqrt((pow(sx.at<cv::Vec3b>(i,j)[0], 2) + pow(sy.at<cv::Vec3b>(i,j)[0], 2)));
//
//            dptr[1] = sqrt((pow(sx.at<cv::Vec3b>(i,j)[1], 2) + pow(sy.at<cv::Vec3b>(i,j)[1], 2)));
//
//            dptr[2] = sqrt((pow(sx.at<cv::Vec3b>(i,j)[2], 2) + pow(sy.at<cv::Vec3b>(i,j)[2], 2)));
//
//        }
//
//    }
//
//    return 0;
//}
//
//
//    
//
//int blurQuantizeFilter(cv::Mat &src, cv::Mat &dest, int levels){
//    //    actual way to copy and allocate data of the same size
////    cv::Mat temp;
////    temp = src.clone();
//    
//    cv::Mat temp(src.size(), CV_8UC3);
//
//    int b = 255/levels;
//    
//    printf("The size of the image is: %d %d\n", src.rows, src.cols);
//    
//    blur5x5Filter(src, temp);
//    
////    implementing this
//    cv::Vec3b dptr1;
//    cv::Vec3b dptr2;
//    
//    for (int i=0; i < src.rows; i++){
//        
//        for(int j=0; j < src.cols; j++){
//            
//            dptr1[0] = temp.at<cv::Vec3b>(i,j)[0] / b;
//            dptr1[1] = temp.at<cv::Vec3b>(i,j)[1] / b;
//            dptr1[2] = temp.at<cv::Vec3b>(i,j)[2] / b;
//            
//            dptr2[0] = dptr1[0] * b;
//            dptr2[1] = dptr1[1] * b;
//            dptr2[2] = dptr1[2] * b;
//            
//            dest.at<cv::Vec3b>(i,j)[0] = dptr2[0];
//            dest.at<cv::Vec3b>(i,j)[1] = dptr2[1];
//            dest.at<cv::Vec3b>(i,j)[2] = dptr2[2];
//            
//        }
//
//    }
//
//    return 0;
//}
//
//
//int cartoon(cv::Mat &src, cv::Mat &dest, int levels, int magThreshold){
//    //    actual way to copy and allocate data of the same size
//    cv::Mat temp;
//    temp = src.clone();
//        
//    magThreshold = 15;
//    
//    printf("The size of the image is: %d %d\n", src.rows, src.cols);
//
//    for (int i=1; i < dest.rows-1; i++){
//        
//        for(int j=1; j < dest.cols-1; j++){
//            
//            cv::Vec3b dptr = dest.at<cv::Vec3b>(i,j);
//            if (dptr[0] > magThreshold) {
//                dptr[0] = 0;
//            }
//            if (dptr[1] > magThreshold) {
//                dptr[1] = 0;
//            }
//            if (dptr[2] > magThreshold) {
//                dptr[2] = 0;
//            }
//        }
//    }
//    
//    temp.copyTo(dest);
//    
//    return 0;
//}
//
//
//int thresholdFilter(cv::Mat &src, cv::Mat &dest){
//
//    //    actual way to copy and allocate data of the same size
//    cv::Mat temp, temp2;
//    temp = src.clone();
//    
//    int magThreshold = 127;
//
//    printf("The size of the image is: %d %d\n", src.rows, src.cols);
//    
//    blur5x5Filter(src, temp);
//    greyFilter(temp, temp2);
//
//    for (int i=0; i < temp.rows; i++){
//        
//        for(int j=0; j < temp.cols; j++){
//            
//            cv::Vec3b &dptr = temp2.at<cv::Vec3b>(i,j);
//            
//            for (int k=0; k < 3; k++) {
//                if (dptr[k] > 127){
//                    dptr[k] = 255;
//                }
//                else {
//                    dptr[k] = 0;
//                };
//            }
//        }
//    }
//
//    temp2.copyTo(dest);
//    
//    return 0;
//}
//
//void grassfire(cv::Mat& im, cv::Mat& out, int bgVal, bool useEightConn = false) {
//        if (im.channels() != 1) {
//            return;
//        }
//        if (out.type() != CV_32SC1) {
//            return;
//        }
//        int oobVal = (bgVal == 0) ? 0 : std::max(im.rows, im.cols);
//        out = 0;
//    
//        // 1
//    
//        for (int r = 0; r < im.rows; r++) {
//            for (int c = 0; c < im.cols; c++) {
//                int px = im.at<uchar>(r, c);
//                if (px != bgVal) {
//                    int nbrUp = oobVal, nbrLt = oobVal;
//                    int rUp = r - 1, cLt = c - 1;
//                    if (rUp >= 0) nbrUp = out.at<int>(rUp, c);
//                    if (cLt >= 0) nbrLt = out.at<int>(r, cLt);
//                    int nbrMin = std::min(nbrUp, nbrLt);
//                    if (useEightConn)
//                    {
//                        int nbrUpLt = oobVal, nbrUpRt = oobVal;
//                        int cRt = c + 1;
//                        if (rUp >= 0 && cLt >= 0) nbrUpLt = out.at<int>(rUp, cLt);
//                        if (rUp >= 0 && cRt < out.cols) nbrUpRt = out.at<int>(rUp, cRt);
//                        nbrMin = std::min(nbrMin, nbrUpLt);
//                        nbrMin = std::min(nbrMin, nbrUpRt);
//                    }
//                    out.at<int>(r, c) = 1 + nbrMin;
//                }
//            }
//        }
//    
//        // 2
//        
//        for (int r = out.rows - 1; r > -1; r--) {
//            for (int c = out.cols - 1; c > -1; c--) {
//                int px = out.at<int>(r, c);
//                if (px != bgVal) {
//                    int nbrDn = oobVal, nbrRt = oobVal;
//                    int rDn = r + 1, cRt = c + 1;
//                    if (rDn < out.rows) nbrDn = out.at<int>(rDn, c);
//                    if (cRt < out.cols) nbrRt = out.at<int>(r, cRt);
//                    int nbrMin = std::min(nbrDn, nbrRt);
//                    if (useEightConn)
//                    {
//                        int nbrDnRt = oobVal, nbrDnLt = oobVal;
//                        int cLt = c - 1;
//                        if (rDn < out.rows && cRt < out.cols) nbrDnRt = out.at<int>(rDn, cRt);
//                        if (rDn < out.rows && cLt >= 0) nbrDnLt = out.at<int>(rDn, cLt);
//                        nbrMin = std::min(nbrMin, nbrDnRt);
//                        nbrMin = std::min(nbrMin, nbrDnLt);
//                    }
//                    out.at<int>(r, c) = std::min(px, 1 + nbrMin);
//                }
//            }
//        }
//    }
//    void grow(cv::Mat& im, cv::Mat& out, int steps, bool useEightConn = false) {
//        cv::Mat im_grass(im.size(), CV_32SC1, cv::Scalar(0));
//        grassfire(im, im_grass, 255, useEightConn);
//        // DEBUG
//        //cv::Mat db;
//        //cv::normalize(im_grass, db, 0, 255, cv::NORM_MINMAX);
//        //db.convertTo(db, CV_8U);
//        //namedWindow("dilate grassfire", cv::WINDOW_KEEPRATIO);
//        //cv::imshow("dilate grassfire", db);
//        out = im_grass <= steps;
//    }
//    void shrink(cv::Mat& im, cv::Mat& out, int steps, bool useEightConn = false) {
//        cv::Mat im_grass(im.size(), CV_32SC1, cv::Scalar(0));
//        grassfire(im, im_grass, 0, useEightConn);
//        // DEBUG
//        //cv::Mat db;
//        //cv::normalize(im_grass, db, 0, 255, cv::NORM_MINMAX);
//        //db.convertTo(db, CV_8U);
//        //namedWindow("dilate grassfire", cv::WINDOW_KEEPRATIO);
//        //cv::imshow("dilate grassfire", db);
//        out = im_grass > steps;
//    }
//
//
////int connectedComponents(cv::Mat &src, cv::Mat &dest){
////    // Connected components analysis
////            cv::Mat labels;
////            int n_regions = cv::connectedComponents(dilated, labels, 8, CV_32S);
////            std::vector<cv::Vec3b> colors(n_regions);
////            colors[0] = cv::Vec3b(0, 0, 0); // background
////            for (int i = 1; i < n_regions; i++)
////            {
////                colors[i] = cv::Vec3b(i * 50 % 256, i * 100 % 256, i * 150 % 256);
////            }
////            cv::Mat regions = cv::Mat::zeros(dilated.size(), CV_8UC3);
////            for (int y = 0; y < dilated.rows; y++)
////            {
////                for (int x = 0; x < dilated.cols; x++)
////                {
////                    int label = labels.at<int>(y, x);
////                    cv::Vec3b &pixel = regions.at<cv::Vec3b>(y, x);
////                    pixel = colors[label];
////                }
////            }
////            std::vector<std::vector<cv::Point>> contours;
////            for (int i = 1; i < n_regions; i++)
////            {
////                std::vector<cv::Point> contour;
////                for (int y = 0; y < cv::dilate.rows; y++)
////                {
////                    for (int x = 0; x < cv::dilate.cols; x++)
////                    {
////                        if (labels.at<int>(y, x) == i)
////                        {
////                            contour.push_back(cv::Point(x, y));
////                        }
////                    }
////                }
////                if (contour.size() > 500)
////                {
////                    cv::drawContours(frame, std::vector<std::vector<cv::Point>>{contour}, 0, cv::Scalar(0, 255, 0), 2);
////                }
////            }
////    return 0;
////}
//
//void computeFeatures(cv::Mat regionMap, int regionID) {
//    // Extract binary mask for specified region
////    creating new region map mask with region ID in region and 0 otherwise
//    cv::Mat regionMaskBinary = (regionMap == regionID);
//
//    cv::Mat white(regionMap.size(), CV_8UC1, cv::Scalar(255));
//    
//    cv::Mat regionMask;
//    cv::bitwise_and(white, white, regionMask, regionMaskBinary);
//
//    // Compute central moments and normalize
//    cv::Moments moments = cv::moments(regionMask, true);
////    store moments in new mat which is a 64-bit floating-point matrix
//    cv::Mat centralMoments(1, 4, CV_64F);
//
////    taking the moments about various points area, position, etc
//    centralMoments.at<double>(0, 0) = moments.mu20;
//    centralMoments.at<double>(0, 1) = moments.mu11;
//    centralMoments.at<double>(0, 2) = moments.mu02;
//    centralMoments.at<double>(0, 3) = moments.mu30;
////    centralMoments.at<double>(0, 4) = moments.mu21;
////    centralMoments.at<double>(0, 5) = moments.mu12;
//
////    cv::NORM_L2, which specifies Euclidean normalization (i.e., dividing by the L2 norm)
//    cv::normalize(centralMoments, centralMoments, 1, 0, cv::NORM_L2, -1, cv::Mat());
//
//    // Compute oriented bounding box about the axis of least moment
//    // Find contours of region
//    std::vector<std::vector<cv::Point>> contours;
//    cv::findContours(regionMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
//
//    // Compute minimum area rectangle
//    cv::RotatedRect rect = cv::minAreaRect(contours[0]);
//
////    regionMask.convertTo(regionMask, CV_8UC1);
////    cv::RotatedRect rect = cv::minAreaRect(cv::findNonZero(regionMask));
//
//    // Compute aspect ratio of bounding box
//    float aspectRatio = rect.size.width / rect.size.height;
//
//    // Display aspect ratio in real time on video output
//    cv::namedWindow("Region Features", cv::WINDOW_NORMAL);
//    while (true) {
//        cv::Mat frame;
//        // Capture frame from camera or video stream
//        // ...
//        cv::Scalar color(0, 255, 0); // Green
//        cv::putText(frame, "Aspect ratio: " + std::to_string(aspectRatio), cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
//        cv::imshow("Region Features", frame);
//        if (cv::waitKey(1) >= 0) break;
//    }
//}
//
