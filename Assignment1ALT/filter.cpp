////
////  filter.cpp
////  image_filter
////
////  Created by Kristine  Umeh on 1/22/23.
////
#include <iostream>
#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <math.h>
//
#include "filter.hpp"
//

int invertFilter(cv::Mat &src, cv::Mat &dest){

    //    actual way to copy and allocate data of the same size
    dest = src.clone();

    printf("The size of the image is: %d %d\n", src.rows, src.cols);

    //negation of image
    for (int i=0; i < dest.rows; i++){

        for(int j=0; j < dest.cols; j++){

            cv::Vec3b &dptr = dest.at<cv::Vec3b>(i,j);

             dptr[0] = 255 - dptr[0];
             dptr[1] = 255 - dptr[1];
             dptr[2] = 255 - dptr[2];
            
            
        }

    }
    return 0;
}

int greyFilter(cv::Mat &src, cv::Mat &dest){
    
    //    actual way to copy and allocate data of the same size
    dest = src.clone();
    
    printf("The size of the image is: %d %d\n", src.rows, src.cols);
    
    //negation of image
    for (int i=0; i < dest.rows; i++){
        
        for(int j=0; j < dest.cols; j++){
            
            cv::Vec3b &dptr = dest.at<cv::Vec3b>(i,j);

//            take away the pigment by averaging
//            BGR
            int b = dptr[0]; // blue
            int g = dptr[1]; //green
            int r = dptr[2]; // red
            
            int avg = (b+g+r)/3;
            
            dptr[0] = avg;
            dptr[1] = avg;
            dptr[2] = avg;
            
        }
        
    }
    watermarkFilter(dest, 50, 100, 50, 100);

    return 0;
}


// custom filter 
int sepiaFilter(cv::Mat &src, cv::Mat &dest){
    //    actual way to copy and allocate data of the same size
    dest = src.clone();

    printf("The size of the image is: %d %d\n", src.rows, src.cols);

    for (int i=0; i < src.rows; i++){

        for(int j=0; j < src.cols; j++){

            cv::Vec3b &dptr = dest.at<cv::Vec3b>(i,j);

            
            dptr[0] = src.at<cv::Vec3b>(i,j)[0] * 0.189 + src.at<cv::Vec3b>(i,j)[1] * 0.534 + src.at<cv::Vec3b>(i,j)[0] * 0.272;
            dptr[1] = src.at<cv::Vec3b>(i,j)[0] * 0.168 + src.at<cv::Vec3b>(i,j)[1] * 0.686 + src.at<cv::Vec3b>(i,j)[0] * 0.349;
            dptr[2] = src.at<cv::Vec3b>(i,j)[0] * 0.131 + src.at<cv::Vec3b>(i,j)[1] * 0.769 + src.at<cv::Vec3b>(i,j)[0] * 0.393;

//            dptr[0] = dptr[0] * 0.189 + dptr[1] * 0.534 + dptr[2] * 0.272;
//            dptr[1] = dptr[0] * 0.168 + dptr[1] * 0.686 + dptr[2] * 0.349;
//            dptr[2] = dptr[0] * 0.131 + dptr[1] * 0.769 + dptr[2] * 0.393;

        }

    }
    return 0;
}

int watermarkFilter(cv::Mat &src, int row_min, int row_max, int col_min, int col_max){
    cv::Mat dest = src.clone();

    printf("The size of the image is: %d %d\n", src.rows, src.cols);

    for (int i=row_min; i < row_max; i++){

        for(int j=col_min; j < col_max; j++){

            cv::Vec3b &dptr = dest.at<cv::Vec3b>(i,j);
            
            dptr[0] = 0;
            dptr[1] = 0;
            dptr[2] = 0;

        }

    }
    
    return 0;
}


int blur5x5Filter(cv::Mat &src, cv::Mat &dest){
    //    actual way to copy and allocate data of the same size
    cv::Mat temp;
    
    temp = src.clone();

    printf("The size of the image is: %d %d\n", src.rows, src.cols);
    
//  gaussian coeff .... is this blur on?
    int coeff = 1 + 2 + 4 + 2 + 1;

    for (int i=2; i < src.rows-2; i++){
        
        for(int j=0; j < src.cols; j++){

            cv::Vec3b &dptr = temp.at<cv::Vec3b>(i,j);

            dptr[0] = (1 * src.at<cv::Vec3b>(i-2,j)[0] + 2 * src.at<cv::Vec3b>(i-1,j)[0] + 4 * src.at<cv::Vec3b>(i,j)[0] + 2 * src.at<cv::Vec3b>(i+1,j)[0] + 1 * src.at<cv::Vec3b>(i+2,j)[0])/coeff;

            dptr[1] = (1 * src.at<cv::Vec3b>(i-2,j)[1] + 2 * src.at<cv::Vec3b>(i-1,j)[1] + 4 * src.at<cv::Vec3b>(i,j)[1] + 2 * src.at<cv::Vec3b>(i+1,j)[1] + 1 * src.at<cv::Vec3b>(i+2,j)[1])/coeff;

            dptr[2] = (1 * src.at<cv::Vec3b>(i-2,j)[2] + 2 * src.at<cv::Vec3b>(i-1,j)[2] + 4 * src.at<cv::Vec3b>(i,j)[2] + 2 * src.at<cv::Vec3b>(i+1,j)[2] + 1 * src.at<cv::Vec3b>(i+2,j)[2])/coeff;

        }
    }
    
    for (int i=0; i < src.rows; i++){
        
        for(int j=2; j < src.cols-2; j++){

            cv::Vec3b &dptr = temp.at<cv::Vec3b>(i,j);

            dptr[0] = (1 * src.at<cv::Vec3b>(i,j-2)[0] + 2 * src.at<cv::Vec3b>(i,j-1)[0] + 4 * src.at<cv::Vec3b>(i,j)[0] + 2 * src.at<cv::Vec3b>(i,j+1)[0] + 1 * src.at<cv::Vec3b>(i,j+2)[0])/coeff;

            dptr[1] = (1 * src.at<cv::Vec3b>(i,j-2)[1] + 2 * src.at<cv::Vec3b>(i,j-1)[1] + 4 * src.at<cv::Vec3b>(i,j)[1] + 2 * src.at<cv::Vec3b>(i,j+1)[1] + 1 * src.at<cv::Vec3b>(i,j+2)[1])/coeff;

            dptr[2] = (1 * src.at<cv::Vec3b>(i,j-2)[2] + 2 * src.at<cv::Vec3b>(i,j-1)[2] + 4 * src.at<cv::Vec3b>(i,j)[2] + 2 * src.at<cv::Vec3b>(i,j+1)[2] + 1 * src.at<cv::Vec3b>(i,j+2)[2])/coeff;

        }
    }
//    dest = temp.clone();
    
    temp.copyTo(dest);
    
    return 0;
}


int sobelX3x3Filter(cv::Mat &src, cv::Mat3s &dest){
    //    actual way to copy and allocate data of the same size
    cv::Mat3s temp;
    temp = src.clone();
    
    printf("The size of the image is: %d %d\n", src.rows, src.cols);
    
    for (int i=1; i < src.rows-1; i++){
        
        for(int j=0; j < src.cols; j++){
            
            cv::Vec3s &dptr = temp.at<cv::Vec3s>(i,j);
            
            dptr[0] = ((1) * src.at<cv::Vec3b>(i-1,j)[0] + (2) * src.at<cv::Vec3b>(i,j)[0] + (1) * src.at<cv::Vec3b>(i+1,j)[0]);
            
            dptr[1] = ((1) * src.at<cv::Vec3b>(i-1,j)[1] + (2) * src.at<cv::Vec3b>(i,j)[1] + (1) * src.at<cv::Vec3b>(i+1,j)[1]);
            
            dptr[2] = ((1) * src.at<cv::Vec3b>(i-1,j)[2] + (2) * src.at<cv::Vec3b>(i,j)[2] + (1) * src.at<cv::Vec3b>(i+1,j)[2]);
            
        }
    }
    
    for (int i=0; i < src.rows; i++){
        
        for(int j=1; j < src.cols-1; j++){
            
            cv::Vec3s &dptr = dest.at<cv::Vec3s>(i,j);
            
            dptr[0] = ((1) * temp.at<cv::Vec3s>(i,j-1)[0] + (0) * temp.at<cv::Vec3s>(i,j)[0] + (-1) * temp.at<cv::Vec3s>(i,j+1)[0]);
            
            dptr[1] = ((1) * temp.at<cv::Vec3s>(i,j-1)[1] + (0) * temp.at<cv::Vec3s>(i,j)[1] + (-1) * temp.at<cv::Vec3s>(i,j+1)[1]);
            
            dptr[2] = ((1) * temp.at<cv::Vec3s>(i,j-1)[2] + (0) * temp.at<cv::Vec3s>(i,j)[2] + (-1) * temp.at<cv::Vec3s>(i,j+1)[2]);
            
        }
    }
//    temp.copyTo(dest);
    
    return 0;
    
}



int sobelY3x3Filter(cv::Mat &src, cv::Mat3s &dest){
    //    actual way to copy and allocate data of the same size
    cv::Mat3s temp;
    
    temp = src.clone();

    printf("The size of the image is: %d %d\n", src.rows, src.cols);
    
    for (int i=1; i < src.rows-1; i++){
        
        for(int j=0; j < src.cols; j++){
            
            cv::Vec3s &dptr = temp.at<cv::Vec3s>(i,j);
            
            dptr[0] = ((1) * src.at<cv::Vec3b>(i-1,j)[0] + (0) * src.at<cv::Vec3b>(i,j)[0] + (-1) * src.at<cv::Vec3b>(i+1,j)[0]);
            
            dptr[1] = ((1) * src.at<cv::Vec3b>(i-1,j)[1] + (0) * src.at<cv::Vec3b>(i,j)[1] + (-1) * src.at<cv::Vec3b>(i+1,j)[1]);
            
            dptr[2] = ((1) * src.at<cv::Vec3b>(i-1,j)[2] + (0) * src.at<cv::Vec3b>(i,j)[2] + (-1) * src.at<cv::Vec3b>(i+1,j)[2]);
            
        }
    }
    
    for (int i=0; i < src.rows; i++){
        
        for(int j=1; j < src.cols-1; j++){
            
            cv::Vec3s &dptr = dest.at<cv::Vec3s>(i,j);
            
            dptr[0] = ((1) * temp.at<cv::Vec3s>(i,j-1)[0] + (2) * temp.at<cv::Vec3s>(i,j)[0] + (1) * temp.at<cv::Vec3s>(i,j+1)[0]);
            
            dptr[1] = ((1) * temp.at<cv::Vec3s>(i,j-1)[1] + (2) * temp.at<cv::Vec3s>(i,j)[1] + (1) * temp.at<cv::Vec3s>(i,j+1)[1]);
            
            dptr[2] = ((1) * temp.at<cv::Vec3s>(i,j-1)[2] + (2) * temp.at<cv::Vec3s>(i,j)[2] + (1) * temp.at<cv::Vec3s>(i,j+1)[2]);
            
        }
    }
    
    return 0;
  
}

int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dest){

    for (int i=0; i < sx.rows; i++){

        for(int j=0; j < sy.cols; j++){
            
            cv::Vec3b &dptr = dest.at<cv::Vec3b>(i,j);

            dptr[0] = sqrt((pow(sx.at<cv::Vec3b>(i,j)[0], 2) + pow(sy.at<cv::Vec3b>(i,j)[0], 2)));

            dptr[1] = sqrt((pow(sx.at<cv::Vec3b>(i,j)[1], 2) + pow(sy.at<cv::Vec3b>(i,j)[1], 2)));

            dptr[2] = sqrt((pow(sx.at<cv::Vec3b>(i,j)[2], 2) + pow(sy.at<cv::Vec3b>(i,j)[2], 2)));

        }

    }

    return 0;
}


    

int blurQuantizeFilter(cv::Mat &src, cv::Mat &dest, int levels){
    //    actual way to copy and allocate data of the same size
//    cv::Mat temp;
//    temp = src.clone();
    
    cv::Mat temp(src.size(), CV_8UC3);

    int b = 255/levels;
    
    printf("The size of the image is: %d %d\n", src.rows, src.cols);
    
    blur5x5Filter(src, temp);
    
//    implementing this
    cv::Vec3b dptr1;
    cv::Vec3b dptr2;
    
    for (int i=0; i < src.rows; i++){
        
        for(int j=0; j < src.cols; j++){
            
            dptr1[0] = temp.at<cv::Vec3b>(i,j)[0] / b;
            dptr1[1] = temp.at<cv::Vec3b>(i,j)[1] / b;
            dptr1[2] = temp.at<cv::Vec3b>(i,j)[2] / b;
            
            dptr2[0] = dptr1[0] * b;
            dptr2[1] = dptr1[1] * b;
            dptr2[2] = dptr1[2] * b;
            
            dest.at<cv::Vec3b>(i,j)[0] = dptr2[0];
            dest.at<cv::Vec3b>(i,j)[1] = dptr2[1];
            dest.at<cv::Vec3b>(i,j)[2] = dptr2[2];
            
        }

    }

    return 0;
}


int cartoon(cv::Mat &src, cv::Mat &dest, int levels, int magThreshold){
    //    actual way to copy and allocate data of the same size
    cv::Mat temp;
    temp = src.clone();
        
    magThreshold = 15;
    
    printf("The size of the image is: %d %d\n", src.rows, src.cols);

    for (int i=1; i < dest.rows-1; i++){
        
        for(int j=1; j < dest.cols-1; j++){
            
            cv::Vec3b dptr = dest.at<cv::Vec3b>(i,j);
            if (dptr[0] > magThreshold) {
                dptr[0] = 0;
            }
            if (dptr[1] > magThreshold) {
                dptr[1] = 0;
            }
            if (dptr[2] > magThreshold) {
                dptr[2] = 0;
            }
        }
    }
    
    temp.copyTo(dest);
    
    return 0;
}


int thresholdFilter(cv::Mat &src, cv::Mat &dest){

    //    actual way to copy and allocate data of the same size
    dest = src.clone();

    printf("The size of the image is: %d %d\n", src.rows, src.cols);

    //negation of image
    for (int i=0; i < dest.rows; i++){

        for(int j=0; j < dest.cols; j++){

            cv::Vec3b &dptr = dest.at<cv::Vec3b>(i,j);

             dptr[0] = 255 - dptr[0];
             dptr[1] = 255 - dptr[1];
             dptr[2] = 255 - dptr[2];
            
            
        }

    }
    return 0;
}
