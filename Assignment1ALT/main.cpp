// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/highgui.hpp>
// #include <opencv2/imgproc.hpp>
// #include <iostream>
//
//
//deep fried overspill
//            dptr[0] *= dptr[2];
//            dptr[1] *= dptr[1];
//            dptr[2] *= dptr[0];
           
           
           

//            static filter... salt and pepper like noise
//            dptr[0] = dptr[-1] + dptr[0];
//            dptr[1] = dptr[0] + dptr[1];
//            dptr[2] = dptr[1] + dptr[2];
           
//            std::cout << static_cast<int>(dest.at<cv::Vec3b>(i,j)[0] ) << "\n";
//
//            std::cout << static_cast<int>(dest.at<cv::Vec3b>(i,j)[1] )<< "\n";
//
//            std::cout << static_cast<int>(dest.at<cv::Vec3b>(i,j)[2] )<< "\n";
////#include <opencv2/opencv.hpp>
//using namespace cv;
//
//int main()
//{
//    // Load image
//    Mat image = imread("class_image.jpeg", IMREAD_COLOR);
//    
//    // Convert to grayscale
//    Mat gray;
//    
//    cvtColor(image, gray, COLOR_BGR2GRAY);
//    
//    // Apply median blur
//    medianBlur(gray, gray, 5);
//    
//    // Apply Canny edge detection
//    Mat edges;
//    Canny(gray, edges, 50, 150);
//    
//    // Threshold the edges
//    Mat bw;
//    threshold(edges, bw, 128, 255, THRESH_BINARY);
//    
//    // Invert the edges
//    Mat bw_inv;
//    bitwise_not(bw, bw_inv);
//    
//    // Dilate the edges
//    Mat dilated;
//    Mat kernel = Mat::ones(3, 3, CV_8U);
//    dilate(bw_inv, dilated, kernel);
//    
//    // Color the edges
//    Mat color_edges;
//    image.copyTo(color_edges, dilated);
//    
//
//
//    
//    // Save the output
////    imwrite("cartoon.jpg", color_edges);
//    
//    imshow("cartoon.jpg", color_edges);
//    
//    imshow("edges", edges);
//    
//    imshow("bw", bw);
//    
//    imshow("bw_inv", bw_inv);
//    
//    imshow("dilated", dilated);
//    
//    imshow("OG", image);
//    
//    waitKey(0);
//    
//    return 0;
//}
//
//
//
//
//
//
//// using namespace cv;
//// using namespace std;
////
////
////
////
////int main() {
////
//////   string path = "class_image.jpeg";
//////   Mat img = imread(path);
//////    variable declarations
//////   Mat imgGray, imgBlur, imgCanny, imgDil, imgErode;
////
////    string path = "Resources/test_video.mp4";
////    VideoCapture cap(path);
////    Mat img, imgGray;
////
////
////    while (true) {
////        cap.read(img);
////
////        cvtColor(img, imgGray, COLOR_BGR2GRAY);
////
////        imshow("Vid", img);
////
////        imshow("VidGrey", imgGray);
////
////
////         waitKey(1);
////     }
////
//////    converting images to different colour scales
//////   cvtColor(img, imgGray, COLOR_BGR2RGB);
////
//////   GaussianBlur(img, imgBlur, Size(3, 3), 3, 0);
//////
//////    Canny(imgBlur, imgCanny, 50, 150);
//////
//////
//////
//////   Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3)); // for dilation
//////
//////    dilate(imgCanny, imgDil, kernel);
//////    erode(imgDil, imgErode, kernel);
////
//////   imshow("Image", img);
//////   imshow("Image Gray", imgGray);
//////
//////   imshow("Image Blur", imgBlur);
//////   imshow("Image Canny", imgCanny);
//////   imshow("Image Dilation", imgDil);
//////   imshow("Image Erode", imgErode);
//////   waitKey(0);
////
////
////    return 0;
////
////}
//
//
/////////////////// Webcam //////////////////////
/////
/////
////int main() {
////
//////    string path = "Resources/test_video.mp4";
////
////    VideoCapture cap(0);
////    Mat img;
////
////    while (true) {
////        cap.read(img);
////
////        imshow("Image", img);
////        waitKey(1);
////    }
////
////  return 0;
////
//// }
//
//
//
//
//
// ///////////////// Video //////////////////////
//
//// int main() {
////
////     string path = "Resources/test_video.mp4";
////     VideoCapture cap(path);
////     Mat img;
////
////     while (true) {
////         cap.read(img);
////
////         imshow("Image", img);
////         waitKey(1);
////     }
////
////   return 0;
////
////  }
//
//
////
////#include <opencv2/imgcodecs.hpp>
//// #include <opencv2/highgui.hpp>
//// #include <opencv2/imgproc.hpp>
//// #include <iostream>
////
//// using namespace cv;
//// using namespace std;
////
////
//// /////////////////  Images  //////////////////////
////
////// int main() {
//////
//////     string path = "/Users/kristineumeh/Desktop/class_image.jpeg";
//////     cv::Mat img = imread(path);
//////     cv::imshow("Image", img);
//////     cv::waitKey(0);
//////     return 0;
////// }
////
//////
//////  main.cpp
//////  RealTimeFiltering
//////
//////  Created by Kristine  Umeh on 1/14/23.
//////
//////
////#include <iostream>
////#include <cstdio>
////#include <cstring>
////#include <opencv2/opencv.hpp>
////
//////start out with main.. always int
////int main(int argc, const char * argv[]) {
////    cv::Mat src;
////
////    char filename[256];
////    // insert code here...
////
////    if(argc < 2) {
////        printf("Usage %s < image filename\n", argv[0]);
////        exit(-1);
////    }
////    strcpy(filename, argv[1]);
////
////    src = cv::imread(filename); // reads image to file and allocates space accordingly
////    if(src.data == NULL) {
////        printf("Unable to read image %s\n", filename);
////        exit(-2);
////    }
//////    open cv allocation: imread, create, copyTo
//////    deallocation: when eveything that has to do with that data is done.. then its done also
////
//////    aliasing
////    cv::Mat src2 = src;
////
//////    actual way to copy and allocate data of the same size
////    src.copyTo(src2);
////
//////    for create, (row, col, type)
////    src2.create(100, 200, CV_16SC3);
////
////    src2 = cv::Mat::zeros(100, 200, CV_16SC3);
////
////    src2.create(src.size(), src.type()); // new image same size
////
////    printf("The size of the image is: %d %d\n", src.rows, src.cols);
////
////
////
//////    
////
////    cv::namedWindow(filename, 1);
////
////    cv::imshow(filename, src);
////
////    cv::waitKey(0);
////
////    cv::destroyWindow(filename);
////
////    cv::imwrite("/Users/kristineumeh/Desktop/GradSchool/ComputerVisualisation/RealTimeFilterMain/Resources/class_image.jpeg", src);
////
////    printf("TERMINATE!!!\n");
////
////     return (0);
////   }
////
////
