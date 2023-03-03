////
////  main.cpp
////  real_time_2D
////
////  Created by Kristine  Umeh on 2/15/23.
////
//
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//#include <iostream>
//#include <opencv2/opencv.hpp>
//
//#include "filter.hpp"
//
//using namespace cv;
//using namespace std;
//
//int main(int argc, char *argv[]) {
//        cv::VideoCapture *capdev;
//
//        // open the video device
//        capdev = new cv::VideoCapture(0); // using camera
////        capdev = new cv::VideoCapture(1); // using phone cam
//
//
//        if( !capdev->isOpened() ) {
//                printf("Unable to open video device\n");
//                return(-1);
//        }
//
//        // get some properties of the image
//        cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
//                       (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
////        cv::Size refS(300,300);
//
//        printf("Expected size: %d %d\n", refS.width, refS.height);
//
//        cv::namedWindow("OriginalVideo", WINDOW_AUTOSIZE);
//        cv::namedWindow("ThresholdedVideo", WINDOW_NORMAL); // identifies a window
//
//        cv::Mat frame, frame_output, blurred_output, threshed_output;
//
////    initialise a flag to support operations.. doesnt slow it down so much
//        int flag = 1;
//
//    for(;;) {
//
//        *capdev >> frame; // get a new frame from the camera, treat as a stream
//
//        if( frame.empty() ) {
//            printf("frame is empty\n");
//            break;
//        }
//
//        // see if there is a waiting keystroke
//        char key = cv::waitKey(10);
//
////    blur at this step? and add saturation .....image preprocesssing
//
//        if (flag == 3) {
//            printf("inside feature detection\n");
//            thresholdFilter(frame_output, threshed_output);
//
//            computeFeatures(threshed_output, 1);
//        }
//
//        if (flag == 2) {
//            printf("inside blur\n");
//            GaussianBlur(frame, blurred_output, Size(5,5), 0);
//            cvtColor(blurred_output, frame_output, COLOR_BGR2GRAY);
//
//    //       apply thresholding
////            threshold(frame_output, threshed_output, 100, 255, THRESH_BINARY);
//
//            thresholdFilter(frame_output, threshed_output);
//
//            cv::imshow("ThresholdedVideo", threshed_output);
//
//        }
//
//        else if (flag == 1){
//            printf("copying original frame\n");
//            frame.copyTo(frame_output);
//        }
//
//        printf("start showing frame\n");
//        cv::imshow("OriginalVideo", frame);
//
//        printf("end showing frame\n");
//
//        if (key == 'f'){
//            printf("flag change 2");
//            flag = 3;
//        }
//
//        if (key == 't'){
//            printf("flag change 2");
//            flag = 2;
//        }
//
//        else if (key == ' '){
//            printf("flag change 1");
//            flag = 1;
//        }
//
////      save image
//        if( key == 's') {
//            string saved_vid;
//            cout << "Name your Video:\n";
//            cin >> saved_vid;
//
//            imwrite(saved_vid, frame_output);
//
//            cout << "Vid saved as: " << saved_vid << endl;
//
//            }
//
////      quit program
//        else if( key == 'q') {
//            break;
//        }
//    }
//    delete capdev;
//    return(0);
//}


