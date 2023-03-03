//
//  vidDisplay.cpp
//  image_filter
//
//  Created by Kristine  Umeh on 1/22/23.
//
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#include "filter.hpp"

//#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;


int main(int argc, char *argv[]) {
        cv::VideoCapture *capdev;

        // open the video device
        capdev = new cv::VideoCapture(0); // using camera
//        capdev = new cv::VideoCapture(1); // using phine cam

        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }

        // get some properties of the image
        cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                       (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
//        cv::Size refS(300,300);

        printf("Expected size: %d %d\n", refS.width, refS.height);

        cv::namedWindow("Video", WINDOW_AUTOSIZE); // identifies a window
        cv::Mat frame, frame_output;

//    initialise a flag to support operations.. doesnt slow it down so much
        int flag = 1;

        for(;;) {

            *capdev >> frame; // get a new frame from the camera, treat as a stream

            if( frame.empty() ) {
              printf("frame is empty\n");
              break;
            }

            // see if there is a waiting keystroke

            char key = cv::waitKey(10);

            if (flag == 2) {
                printf("inside grey\n");
                cvtColor(frame, frame_output, COLOR_BGR2GRAY);

            }

            else if (flag == 3){
                printf("inside invert filter");

                cv::Mat frameNeg;
                invertFilter(frame, frameNeg);
                frameNeg.copyTo(frame_output);

                printf("outside invert filter");

            }

            else if (flag == 4){
                printf("inside customised grey filter");

                cv::Mat customGrey;
                greyFilter(frame, customGrey);
                customGrey.copyTo(frame_output);

                printf("outside customised grey filter");

            }

            else if (flag == 5){
                printf("inside blur filter");
                cv::Mat frameBlur;

                blur5x5Filter(frame, frameBlur);
                frameBlur.copyTo(frame_output);

                printf("outside blur filter");

            }

            else if (flag == 6){
                printf("inside sobel x filter");

                cv::Mat3s frameXSob;
                frameXSob = frame.clone();

                sobelX3x3Filter(frame, frameXSob);

                convertScaleAbs(frameXSob, frame_output);
//                frameXSob.copyTo(frame_output);

                printf("outside sobel x filter");

            }

            else if (flag == 7){
                printf("inside sobel y filter");

                cv::Mat3s frameYSob;
                frameYSob = frame.clone();

                sobelY3x3Filter(frame, frameYSob);

                convertScaleAbs(frameYSob, frame_output);

                printf("outside sobel y filter");

            }


            else if (flag == 8){
                printf("inside customised filter");

                cv::Mat customFilt;
                sepiaFilter(frame, customFilt);
                customFilt.copyTo(frame_output);

                printf("outside customised filter");

            }

            else if (flag == 9){
                printf("inside magnitude filter");
                cv::Mat3s frameX = frame.clone();
                cv::Mat3s frameY = frame.clone();

                cv::Mat frameX_output;
                cv::Mat frameY_output;

                sobelX3x3Filter(frame, frameX);
                convertScaleAbs(frameX, frameX_output);

                sobelY3x3Filter(frame, frameY);
                convertScaleAbs(frameY, frameY_output);

                magnitude(frameX_output, frameY_output, frame_output);

                printf("outside magnitude filter");
            }

            else if (flag == 10){
                printf("inside blurQ filter");

                cv::Mat Qblur;
                Qblur = frame.clone();
                blurQuantizeFilter(Qblur, frame_output, 15);

                printf("outside blurQ filter");

            }
            else if (flag == 11){
                printf("inside cartoon filter");
                
                cv::Mat3s frameX = frame.clone();
                cv::Mat3s frameY = frame.clone();

                cv::Mat frameX_output;
                cv::Mat frameY_output;
                cv::Mat mag_output;
                cv::Mat Qblur;
                cv::Mat bq_output;

                sobelX3x3Filter(frame, frameX);
                convertScaleAbs(frameX, frameX_output);

                sobelY3x3Filter(frame, frameY);
                convertScaleAbs(frameY, frameY_output);

                magnitude(frameX_output, frameY_output, mag_output);
                
                
                Qblur = mag_output.clone();
                blurQuantizeFilter(Qblur, bq_output, 15);

                cv::Mat catoonMat;
                catoonMat = bq_output.clone();
                cartoon(catoonMat, frame_output, 10, 15);

                printf("outside cartoon filter");

            }

            else if (flag == 1){
                printf("copying original frame\n");
                frame.copyTo(frame_output);
            }

            printf("start showing frame\n");

            cv::imshow("video", frame_output);

            printf("end showing frame\n");

            if (key == 'g'){
                printf("flag change 2");
                flag = 2;

            }
            else if (key == 'n'){
                printf("flag change 3");
                flag = 3;

            }
            else if (key == 'd'){
                printf("flag change 4");
                flag = 4;

            }
            else if (key == 'b'){
                printf("flag change 5");
                flag = 5;

            }
            else if (key == 'x'){
                printf("flag change 6");
                flag = 6;

            }
            else if (key == 'y'){
                printf("flag change 7");
                flag = 7;

            }
            else if (key == 'w'){
                printf("flag change 8");
                flag = 8;

            }
            else if (key == 'm'){
                printf("flag change 9");
                flag = 9;

            }
            else if (key == 'i'){
                printf("flag change 10");
                flag = 10;

            }
            else if (key == 'c'){
                printf("flag change 11");
                flag = 11;

            }

            else if (key == ' '){
                printf("flag change 1");
                flag = 1;

            }

//          save image
            if( key == 's') {
                string saved_vid;
                cout << "Name your Video:\n";
                cin >> saved_vid;

                imwrite(saved_vid, frame_output);

                cout << "Vid saved as: " << saved_vid << endl;

                }

//          quit program
            else if( key == 'q') {
                break;
            }
        }
        delete capdev;
        return(0);
}
