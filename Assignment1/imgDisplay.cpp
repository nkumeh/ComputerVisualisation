////
////  imgDisplay.cpp
////  image_filter
////
////  Created by Kristine  Umeh on 1/22/23.
////
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//#include <iostream>
//#include "imgDisplay.hpp"
//#include "filter.hpp"
//
////#include <opencv2/opencv.hpp>
//using namespace cv;
//using namespace std;
//
//
//int main(int argc, char *argv[]) {
//    cv::Mat src;
//    char filename[256];
//
//    if(argc < 2) {
//        printf("Usage %s < image filename\n", argv[0]);
//        exit(-1);
//    }
//    
//    strcpy(filename, argv[1]);
//    src = cv::imread(filename); // reads image to file and allocates space accordingly
//    
//    if(src.data == NULL) {
//        printf("Unable to read image %s\n", filename);
//        exit(-2);
//    }
//    
//    cv::namedWindow(filename, 1);
//    cv::Mat img, img_output;
//    
//    
//    int flag = 1;
//
//        for(;;) {
//
////          copy source to image variable
//            src.copyTo(img);
//
//            if(img.empty() ) {
//              printf("frame is empty\n");
//              break;
//            }
//
//            // see if there is a waiting keystroke
//            char key = cv::waitKey(10);
//
//            if (flag == 2) {
//                printf("inside grey\n");
//                cvtColor(img, img_output, COLOR_BGR2GRAY);
//
//            }
//
//            else if (flag == 3){
//                cv::Mat imgNeg;
//                invertFilter(img, imgNeg);
//                imgNeg.copyTo(img_output);
//
//            }
//
//            else if (flag == 4){
//                cv::Mat customGrey;
//                greyFilter(img, customGrey);
//                customGrey.copyTo(img_output);
//
//            }
//
//            else if (flag == 5){
//                cv::Mat imgBlur;
//                blur5x5Filter(img, imgBlur);
//                imgBlur.copyTo(img_output);
//
//            }
//
//            else if (flag == 6){
//                cv::Mat3s imgXSob;
//                imgXSob = img.clone();
//                sobelX3x3Filter(img, imgXSob);
//                convertScaleAbs(imgXSob, img_output);
//
//            }
//
//            else if (flag == 7){
//                cv::Mat3s imgYSob;
//                imgYSob = img.clone();
//                sobelY3x3Filter(img, imgYSob);
//                convertScaleAbs(imgYSob, img_output);
//
//            }
//
//
//            else if (flag == 8){
//                cv::Mat customFilt;
//                sepiaFilter(img, customFilt);
//                customFilt.copyTo(img_output);
//
//            }
//
//            else if (flag == 9){
//                cv::Mat3s frameX = img.clone();
//                cv::Mat3s frameY = img.clone();
//                cv::Mat frameX_output;
//                cv::Mat frameY_output;
//
//                sobelX3x3Filter(img, frameX);
//                convertScaleAbs(frameX, frameX_output);
//
//                sobelY3x3Filter(img, frameY);
//                convertScaleAbs(frameY, frameY_output);
//
//                magnitude(frameX_output, frameY_output, img_output);
//
//            }
//            
//            else if (flag == 10){
//                cv::Mat Qblur;
//                Qblur = img.clone();
//                blurQuantizeFilter(Qblur, img_output, 15);
//
//            }
//
//
//            else if (flag == 1){
//                printf("copying original frame\n");
//                img.copyTo(img_output);
//            }
//
//            cv::imshow("Image", img_output);
//
//            if (key == 'g'){
//                printf("flag change 2");
//                flag = 2;
//
//            }
//            else if (key == 'n'){
//                printf("flag change 3");
//                flag = 3;
//
//            }
//            else if (key == 'd'){
//                printf("flag change 4");
//                flag = 4;
//
//            }
//            else if (key == 'b'){
//                printf("flag change 5");
//                flag = 5;
//
//            }
//            else if (key == 'x'){
//                printf("flag change 6");
//                flag = 6;
//
//            }
//            else if (key == 'y'){
//                printf("flag change 7");
//                flag = 7;
//
//            }
//            else if (key == 'c'){
//                printf("flag change 8");
//                flag = 8;
//
//            }
//            else if (key == 'm'){
//                printf("flag change 9");
//                flag = 9;
//
//            }
//            else if (key == 'i'){
//                printf("flag change 10");
//                flag = 10;
//
//            }
//            else if (key == ' '){
//                printf("flag change 1");
//                flag = 1;
//                
//            }
//
////          save image
//            if( key == 's') {
//                string saved_img;
//                cout << "Name your Video:\n";
//                cin >> saved_img;
//
//                cv::imwrite(filename, src);
//                imwrite(saved_img, img_output);
//
//                cout << "Vid saved as: " << saved_img << endl;
//
//                }
//        
//
////          quit program
//            else if( key == 'q') {
//                break;
//            }
//        }
//        destroyWindow(filename);
//        printf("TERMINATE!!!\n");
//    
//        return(0);
//
//}
//
//    
