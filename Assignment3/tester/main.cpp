// main.cpp : This file contains the 'main' function. Program execution begins and ends there.
// author: Elizabeth Witten
//

#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <experimental/filesystem>
#include <unordered_map>
#include <numeric> // iota
#include <fstream> //ofstream, ifstream

namespace fs = std::filesystem;

struct ImageData{
    cv::Mat im;             // 8UC3
    cv::Mat overlay;        // 8UC3
    cv::Mat bin;            // 8UC1
    cv::Mat visual_feats;   // 8UC3
    cv::Mat regions;        // 32SC1
    int n_regions;
    std::vector<int> region_sizes;
};


// Constants:
static const int MODE = 1; // 0 = train, 1 = classify

static const char QUIT = 'q';
static const char ACTION = ' ';

static const int N_DILATE = 25;
static const int N_ERODE = 30;
static const int MAG_THRESH = 127;
static const float MIN_CONTOUR_RATIO = 0.05;
static const int K_KNN = 3;

static const int CAMERA_ID = 1;
static const cv::String WIN_ORIG = "Camera view";
static const cv::String WIN_FEAT = "Feature view";
static const cv::String OUT_FOLDER = "train";
static const cv::String CSV_NAME = "feat.csv";


// Helpers:
int thresholdFilter(cv::Mat& src, cv::Mat& dest) {
    //  actual way to copy and allocate data of the same size
    cv::Mat temp, temp2;
    temp = src.clone();

    //printf("The size of the image is: %d %d\n", src.rows, src.cols);
    //blur5x5Filter(src, temp);
    //greyFilter(temp, temp2);

    cv::GaussianBlur(src, temp, cv::Size(5, 5), 0, 0);
    cv::cvtColor(temp, temp2, cv::COLOR_BGR2GRAY);
    for (int i = 0; i < temp2.rows; i++) {
        for (int j = 0; j < temp2.cols; j++) {
            uchar* dptr = temp2.data + temp2.step[0] * i + temp2.step[1] * j;
            if (*dptr > MAG_THRESH) *dptr = 255;
            else *dptr = 0;
            //cv::Vec3b& dptr = temp2.at<cv::Vec3b>(i, j);
            //for (int k = 0; k < 3; k++) {
            //  if (dptr[k] > 127) {
            //      dptr[k] = 255;
            //  }
            //  else {
            //      dptr[k] = 0;
            //  };
            //}

        }
    }
    temp2.copyTo(dest);
    return 0;
}
void grassfire(cv::Mat& im, cv::Mat& out, int bgVal, bool useEightConn = false) {
    if (im.channels() != 1) {
        return;
    }
    if (out.type() != CV_32SC1) {
        return;
    }
    int oobVal = std::max(im.rows, im.cols);
    out = 0;
    // 1
    for (int r = 0; r < im.rows; r++) {
        for (int c = 0; c < im.cols; c++) {
            int px = im.at<uchar>(r, c);
            if (px != bgVal) {
                int nbrUp = oobVal, nbrLt = oobVal;
                int rUp = r - 1, cLt = c - 1;
                if (rUp >= 0) nbrUp = out.at<int>(rUp, c);
                if (cLt >= 0) nbrLt = out.at<int>(r, cLt);
                int nbrMin = std::min(nbrUp, nbrLt);
                if (useEightConn)
                {
                    int nbrUpLt = oobVal, nbrUpRt = oobVal;
                    int cRt = c + 1;
                    if (rUp >= 0 && cLt >= 0) nbrUpLt = out.at<int>(rUp, cLt);
                    if (rUp >= 0 && cRt < out.cols) nbrUpRt = out.at<int>(rUp, cRt);
                    nbrMin = std::min(nbrMin, nbrUpLt);
                    nbrMin = std::min(nbrMin, nbrUpRt);
                }
                out.at<int>(r, c) = 1 + nbrMin;
            }
        }
    }
    // 2
    for (int r = out.rows - 1; r > -1; r--) {
        for (int c = out.cols - 1; c > -1; c--) {
            int px = out.at<int>(r, c);
            if (px != bgVal) {
                int nbrDn = oobVal, nbrRt = oobVal;
                int rDn = r + 1, cRt = c + 1;
                if (rDn < out.rows) nbrDn = out.at<int>(rDn, c);
                if (cRt < out.cols) nbrRt = out.at<int>(r, cRt);
                int nbrMin = std::min(nbrDn, nbrRt);
                if (useEightConn)
                {
                    int nbrDnRt = oobVal, nbrDnLt = oobVal;
                    int cLt = c - 1;
                    if (rDn < out.rows && cRt < out.cols) nbrDnRt = out.at<int>(rDn, cRt);
                    if (rDn < out.rows && cLt >= 0) nbrDnLt = out.at<int>(rDn, cLt);
                    nbrMin = std::min(nbrMin, nbrDnRt);
                    nbrMin = std::min(nbrMin, nbrDnLt);
                }
                out.at<int>(r, c) = std::min(px, 1 + nbrMin);
            }
        }
    }
}

void dilate(cv::Mat& im, cv::Mat& out, int steps, bool useEightConn = false) {
    //DEBUG
    //cv::imshow("Before Dilate", im);

    cv::Mat im_grass(im.size(), CV_32SC1, cv::Scalar(0));
    grassfire(im, im_grass, 255, useEightConn);

    // DEBUG

    //cv::Mat db;
    //cv::normalize(im_grass, db, 0, 255, cv::NORM_MINMAX);
    //db.convertTo(db, CV_8U);
    //namedWindow("dilate grassfire", cv::WINDOW_KEEPRATIO);
    //cv::imshow("dilate grassfire", db);

    out = im_grass <= steps;

    //DEBUG
    //cv::imshow("After", out);

}

void erode(cv::Mat& im, cv::Mat& out, int steps, bool useEightConn = false) {
    //DEBUG

    //cv::imshow("Before Erode", im);

    cv::Mat im_grass(im.size(), CV_32SC1, cv::Scalar(0));
    grassfire(im, im_grass, 0, useEightConn);

    // DEBUG

    //cv::Mat db;
    //cv::normalize(im_grass, db, 0, 255, cv::NORM_MINMAX);
    //db.convertTo(db, CV_8U);
    //namedWindow("dilate grassfire", cv::WINDOW_KEEPRATIO);
    //cv::imshow("dilate grassfire", db);

    out = im_grass > steps;

    //DEBUG
    //cv::imshow("After Erode", out);

}

int getComponents(cv::Mat& src, cv::Mat& overlay, cv::Mat& labels, std::vector<int>& region_sizes, int* n_regions) {

    *n_regions = cv::connectedComponents(src, labels, 8, CV_32S);
    
    // count region sizes
    region_sizes.assign(*n_regions, 0);
    for (int y = 0; y < labels.rows; y++) {
        for (int x = 0; x < labels.cols; x++) {
            int id = labels.at<int>(y, x);
            region_sizes[id]++;
        }
    }
    // filter by size and remap the region ids
    int min_contour_size = MIN_CONTOUR_RATIO * (src.rows * src.cols);
    std::vector<int> regionMap(*n_regions);
    regionMap[0] = 0;
    int hi = 0;
    for (int i = 1; i < *n_regions; i++) {
        int newId = (region_sizes[i] < min_contour_size) ? 0 : ++hi;
        regionMap[i] = newId;
        if (newId != 0) region_sizes[newId] = region_sizes[i];
    }
    *n_regions = hi + 1;
    while (region_sizes.size() != *n_regions) region_sizes.pop_back();

    // colors
    std::vector<cv::Vec3b> colors(*n_regions);
    colors[0] = cv::Vec3b(0, 0, 0); // background
    for (int i = 1; i < *n_regions; i++)
    {
        colors[i] = cv::Vec3b(i * 50 % 256, i * 100 % 256, i * 150 % 256);
    }

    // update and color the labels
    for (int y = 0; y < labels.rows; y++)
    {
        for (int x = 0; x < labels.cols; x++)
        {
            int& id = labels.at<int>(y, x);
            id = regionMap[id];
            if (id != 0) overlay.at<cv::Vec3b>(y, x) = colors[id];
        }
    }
    return 0;
}

void computeFeatures(cv::Mat& src, cv::Mat& overlay, cv::Mat& regionMap, int regionID, std::vector<float>& features) {
    // Extract binary mask for specified region
    // Creating new region map mask with region ID in region and 0 otherwise

    cv::Mat regionMaskBinary = (regionMap == regionID);
    cv::Mat white(regionMap.size(), CV_8UC1, cv::Scalar(255));
    cv::Mat regionMask;
    cv::bitwise_and(white, white, regionMask, regionMaskBinary);

    // Compute hu moments - translation, scale, and rotation invariant
    cv::Moments moments = cv::moments(regionMask, true);
    double hu_moments[7];
    cv::HuMoments(moments, hu_moments);

    // Compute oriented bounding box about the axis of least moment
    // Find contours of region
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(regionMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Compute minimum area rectangle
    cv::RotatedRect rect = cv::minAreaRect(contours[0]);

    // Compute aspect ratio of bounding box
    float side1 = std::max(rect.size.width, rect.size.height);
    float side2 = std::min(rect.size.width, rect.size.height);
    float aspectRatio = side1 / side2;

    // Compute percent filled of bounding box
    float filledRatio = cv::countNonZero(regionMask) / rect.size.area();
    cv::Scalar featVisualColor(255, 255, 0);

    // Display rectangle
    cv::Point2f rectVertices[4];
    rect.points(rectVertices);
    for (int i = 0; i < 4; i++)
    {
        cv::line(overlay, rectVertices[i], rectVertices[(i + 1) % 4], featVisualColor, 2);
    }

    // Display axis
    cv::Point2d centroid(moments.m10 / moments.m00, moments.m01 / moments.m00);
    cv::circle(overlay, centroid, 5, featVisualColor);

    // Display aspect ratio
    cv::putText(overlay, "Aspect ratio: " + std::to_string(aspectRatio), rect.center - cv::Point2f(0, 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, featVisualColor, 2);
    // Display aspect ratio
    cv::putText(overlay, "Filled: " + std::to_string(filledRatio), rect.center + cv::Point2f(0, 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, featVisualColor, 2);
    // Output feature vector
    features = {
        (float)hu_moments[0],
        (float)hu_moments[1],
        (float)hu_moments[2],
        (float)hu_moments[3],
        (float)hu_moments[4],
        (float)hu_moments[5],
        (float)hu_moments[6],
        aspectRatio,
        filledRatio
    };
}

/** Add an overlay onto the source image.
 * param src: original color image, typed 8UC3
 * param overlay: overlay image a with black background, typed 8UC3
 * param dst: combination of source image and overlay, typed 8UC3
 * return: 0, on success
 **/

int addOverlay(cv::Mat& src, cv::Mat& overlay, cv::Mat& dst) {
    try {
        // create a mask of the overlay
        cv::Mat overlayGray, mask;
        cvtColor(overlay, overlayGray, cv::COLOR_BGR2GRAY);
        threshold(overlayGray, mask, 1, 255, cv::THRESH_BINARY);

        // apply mask to overlay
        cv::Mat overlayCrop;
        bitwise_and(overlay, overlay, overlayCrop, mask);

        // apply mask to src
        cv::Mat srcCrop;
        bitwise_and(src, src, srcCrop, ~mask);

        // combine
        dst = overlayCrop + srcCrop;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return -1;
    }
    return 0;
}

int train(std::vector<float>& features, std::ofstream& file) {
    // prompt for label
    std::string label;
    std::cout << "Label:\t";
    std::cin >> label;

    // Write to file
    if (!label.empty())
    {
        file << label << ",";
        for (int i = 0; i < features.size(); i++) { file << features[i] << " "; }
        file << std::endl;
    }
    return 0;
}

/**Calculate the distance between two feature vectors (of the same size) as the sum of squared errors
 * param v1: first vector
 * param v2: second vector
 * returns: distance
 **/

double calcSSE(std::vector<float>& v1, std::vector<float>& v2) {
    int s1 = v1.size();
    int s2 = v2.size();
    if (v1.size() != v2.size()) {
        throw std::exception("SSE: Vectors must be the same size");
    }
    double sum = 0;
    for (int i = 0; i < v1.size(); i++) {
        double diff = v1.at(i) - v2.at(i);
        double diff2 = diff * diff;
        if (sum + diff2 < 0) return DBL_MAX;
        else sum += diff2;
    }
    return sum;
}
std::string classifyKNN(std::vector<float> target, std::unordered_map<std::string, std::vector<std::vector<float>>> db) {
    // for each category, compute sum of k closest distances
    std::unordered_map<std::string, double> dists;
    for (std::pair<std::string, std::vector<std::vector<float>>> entry : db) {
        std::string label = entry.first;
        std::vector<std::vector<float>> feats = entry.second;
        std::vector<double> distances(feats.size());
        for (int i = 0; i < feats.size(); i++) {
            distances[i] = calcSSE(feats[i], target);
        }
        std::sort(distances.begin(), distances.end());
        double sum = 0;
        for (int k = 0; k < K_KNN; k++) {
            sum += distances[k];
        }
        dists[label] = sum;
    }

    // find the best label
std::string bestLabel;
    double minDist = DBL_MAX;
    for (std::pair<std::string, double> entry : dists) {
        if (entry.second < minDist) {
            minDist = entry.second;
            bestLabel = entry.first;
        }
    }

    return bestLabel;
}
/** Main function.
 *  Displays live video based on user input.
 */

int main()
{
    // set up:
    cv::VideoCapture* capdev; // video capture device
    ImageData frame; // current frame

    const fs::path dir{ OUT_FOLDER };
    std::ofstream featureFileOut;
    std::unordered_map<std::string, std::vector<std::vector<float>>> db;

    // open video device
    capdev = new cv::VideoCapture(CAMERA_ID);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return -1;
    }

    // initialize overlay to be used for showing features
    frame.overlay = cv::Mat(capdev->get(cv::CAP_PROP_FRAME_HEIGHT),
                            capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                            CV_8UC3, cv::Scalar(0, 0, 0));

    // prepare display window(s)
    cv::namedWindow(WIN_ORIG, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(WIN_FEAT, cv::WINDOW_AUTOSIZE);

    // if mode is classify, load up the database
    if (MODE == 1) {
        if (!fs::exists(dir / CSV_NAME)) {
            std::cout << "No feature file available" << std::endl;
            return -1;
        }
        std::ifstream featureFileIn(dir / CSV_NAME);
        if (!featureFileIn.is_open())
        {
            std::cout << "Failed to open feature vector file" << std::endl;
            return -1;
        }
        // Read the image names and feature vectors from the file
        std::string line;
        getline(featureFileIn, line); // throw away header
        while (getline(featureFileIn, line))
        {
            // create string stream from line
            std::stringstream ss(line);

            // parse label
            std::string label;
            std::getline(ss, label, ',');

            // parse feature vector
            std::vector<float> feats;
            float val;
            while (ss >> val) {
                feats.push_back(val);
            }
            // store this sample's data
            db[label].push_back(feats);
        }
    }

    // main loop
    bool go = true;
    while (go) {

        // see if the window was closed
        if (!cv::getWindowProperty(WIN_ORIG, cv::WND_PROP_VISIBLE) || !cv::getWindowProperty(WIN_FEAT, cv::WND_PROP_VISIBLE)) {
            go = false;
            break;
        }

        // read the next frame
        frame.overlay = 0;
        *capdev >> frame.im;
        if (frame.im.empty()) {
            go = false;
            printf("Frame is empty!\n");
            break;
        }

        // threshold and clean up
        thresholdFilter(frame.im, frame.bin);
        frame.bin = ~frame.bin;
        dilate(frame.bin, frame.bin, N_DILATE, true);
        erode(frame.bin, frame.bin, N_ERODE);

        // get components
        getComponents(frame.bin, frame.overlay, frame.regions, frame.region_sizes, &frame.n_regions);

        // Choose the largest region
        int bestId = 0;
        for (int i = 0; i < frame.n_regions; i++) {
            if (bestId < frame.region_sizes[i]) bestId = i;
        }

        // Compute features
        std::vector<float> features;
        computeFeatures(frame.im, frame.overlay, frame.regions, bestId, features);

        // apply feature overlay
        cv::Mat bin_ch[] = { frame.bin, frame.bin, frame.bin };
        cv::merge(bin_ch, 3, frame.visual_feats);
        addOverlay(frame.visual_feats, frame.overlay, frame.visual_feats);

        // display the image(s)
        cv::imshow(WIN_ORIG, frame.im);
        cv::imshow(WIN_FEAT, frame.visual_feats);

        // see if there is a waiting keystroke and perform appropriate action
        switch (cv::waitKey(10)) {

        case QUIT:
            go = false;
            break;

        case ACTION:
            switch (MODE)
            {
            case 0:  // train

                    // open file if needed
                    if (!featureFileOut.is_open()) {
                        try { fs::create_directory(dir); }
                        catch (const std::exception& e) { std::cerr << e.what() << "\n"; }

                        bool isNew = !fs::exists(dir / CSV_NAME);
                        featureFileOut.open(dir / CSV_NAME, std::ios::app);
                        if (isNew) featureFileOut << "label,features\n";
                }
                train(features, featureFileOut);
                break;

            case 1:  // classify

            {
                std::string label = classifyKNN(features, db);
                std::cout << label << std::endl;
                break;
            }
            default:
                break;
            }
        }
    }
    // cleanup
    delete capdev;
    featureFileOut.close();
    if (cv::getWindowProperty(WIN_ORIG, cv::WND_PROP_VISIBLE)) cv::destroyWindow(WIN_ORIG);
    if (cv::getWindowProperty(WIN_FEAT, cv::WND_PROP_VISIBLE)) cv::destroyWindow(WIN_FEAT);
    return 0;
}
