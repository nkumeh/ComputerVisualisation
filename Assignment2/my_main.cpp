//
//  my_main.cpp
//  pattern_match
//
//  Created by Kristine  Umeh on 2/9/23.
//

#include "my_main.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <regex>
#include <sstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "filter.hpp"

namespace fs = std::filesystem;

// HISTOGRAMS

const int MAX_HISTOGRAM_DIMS = 5;
const int MAX_HISTOGRAM_CHANNELS = 5;

/*
* Calculate a histogram.
* param srcArr: array of source images.
*    All source images must have the same number of rows and columns
*    Type: uchar
* param maxVals: array of max values corresponding to source images
*    Used to compute bin size
* param dims: number of histogram dimensions
*    Equal to number of source images
*    Must be greater than zero and less than or equal to MAX_HISTOGRAM_DIMS
* param histogram: Mat with [dims + 1] dimensions
*    The size of each dimension corresponds to number of bins
*   The final dimension corresponds to number of channels
*    Number of channels must match the source image with the greatest number of channels
*    e.g. 3-D histogram with 8/16/4 bins from source images with max 3 channels has dimensions 8x16x4x3
*    Type: int
*/
int calcHist(cv::Mat srcArr[], int maxVals[], int dims, cv::Mat& hist) {

    try {

        // validations

//        if (dims <= 0 || dims > MAX_HISTOGRAM_DIMS) {
//            throw std::exception(
//                "Error in calcHist: Invalid number of dimensions."
//            );
//        }
//
//        if (hist.dims != (dims + 1)) {
//            throw std::exception(
//                "Error in calcHist: Unexpected number of dimensions in hist matrix."
//            );
//        }
//
//        if (hist.channels() != 1) {
//            throw std::exception(
//                "Error in calcHist: Histogram must be single-channel. Instead, set the last dimension to source channel count."
//            );
//        }

        const int srcRows = srcArr[0].rows;
        const int srcCols = srcArr[0].cols;
        for (int i = 1; i < dims; i++) {
//            if (srcRows != srcArr[i].rows || srcCols != srcArr[i].cols) {
//                throw std::exception(
//                    "Error in calcHist: Source images must have the same rows and columns."
//                );
            }
        }

        // compute bin sizes
        float bSizes[MAX_HISTOGRAM_DIMS] = {};
        cv::MatSize bCounts = hist.size;
        for (int d = 0; d < dims; d++) {
            bSizes[d] = (float)(maxVals[d] + 1) / bCounts[d];
        }

        // loop over image(s) and count up the values
        hist = 0;
        for (int r = 0; r < srcRows; r++) {                    // ROWS
            for (int c = 0; c < srcCols; c++) {                // COLS

                // vector(s) to access histogram, one vector per channel
                cv::Vec<int, MAX_HISTOGRAM_DIMS + 1> hPtrs[MAX_HISTOGRAM_CHANNELS] = {};
                for (int ch = 0; ch < MAX_HISTOGRAM_CHANNELS; ch++) {
                    hPtrs[ch] = cv::Vec<int, MAX_HISTOGRAM_DIMS + 1>(0);
                    hPtrs[ch][dims] = ch;  // last dimension is the channel number
                }

                // fill the vectors, one dimension per image
                int maxChannels = 0;
                for (int d = 0; d < dims; d++) {            // DIMS (src imgs)
                    cv::Mat src = srcArr[d];
                    int bsize = bSizes[d];

                    // switch datatype (# channels)
                    int channels = src.channels();
                    if (channels > maxChannels) maxChannels = channels;
                    switch (channels)
                    {
                    case 1:
                    {
                        uchar binId = src.at<uchar>(r, c) / bsize;
                        hPtrs[0][d] = binId;
                    }
                    break;
                    case 2:
                    {
                        cv::Vec2b px = src.at<cv::Vec2b>(r, c);
                        for (int ch = 0; ch < channels; ch++) {
                            uchar binId = px[ch] / bsize;
                            hPtrs[ch][d] = binId;
                        }
                    }
                    break;
                    case 3:
                    {
                        cv::Vec3b px = src.at<cv::Vec3b>(r, c);
                        for (int ch = 0; ch < channels; ch++) {
                            uchar binId = px[ch] / bsize;
                            hPtrs[ch][d] = binId;
                        }
                    }
                    break;
                    case 4:
                    {
                        cv::Vec4b px = src.at<cv::Vec4b>(r, c);
                        for (int ch = 0; ch < channels; ch++) {
                            uchar binId = px[ch] / bsize;
                            hPtrs[ch][d] = binId;
                        }
                    }
                    break;
//                    default:
//                        throw std::exception("Error in calcHist: Unsupported number of channels");
//                    }
                }

                // increment histogram counter
                for (int ch = 0; ch < maxChannels; ch++) {
                    hist.at<int>(hPtrs[ch])++;
                }
            }
        }

    }
//    catch (std::exception& ex) {
//        std::cout << ex.what() << "\n";
//        return -1;
//    }

    return 0;
}



// DISTANCE METRICS

const int BINS = 16;

/*
* Calculate the distance between two feature vectors (of the same size)
* param v1: first vector
* param v2: second vector
* returns: distance
*/
typedef double(*DistanceMetric) (std::vector<int>& v1, std::vector<int>& v2);

/*
* Calculate the distance between two feature vectors (of the same size) as the sum of squared errors
* param v1: first vector
* param v2: second vector
* returns: distance
*/
double calcSSE(std::vector<int>& v1, std::vector<int>& v2)
{
    int s1 = v1.size();
    int s2 = v2.size();
//    if (v1.size() != v2.size()) {
//        throw std::exception("SSE: Vectors must be the same size");
//    }

    double sum = 0;
    for (int i = 0; i < v1.size(); i++) {
        int diff = v1.at(i) - v2.at(i);
        sum += (diff * diff);
    }

    return sqrt(sum);
}

double calcIntersection(std::vector<int>& v1, std::vector<int>& v2)
{
//    if (v1.size() != v2.size()) {
//        throw std::exception("SSE: Vectors must be the same size");
//    }

    double total = 0;
    double sum = 0;
    for (int i = 0; i < v1.size(); i++) {
        total += v1.at(i);
        int intersection = std::min(v1.at(i), v2.at(i));
        sum += intersection;
    }

    return total - sum;
}



// FEATURE EXTRACTION

/*
* Extract a feature vector from an image.
* param src: image to compute features for
*    Type: uchar
* param features: returned feature vector
* return: 0, on success
*/
typedef int(*FeatureExtractor) (cv::Mat& src, std::vector<int>& features);

/*
* Flattens a matrix into a vector.
* param m: source matrix
*    Channels: 1
*    Type: int
* param flat: stores result
*    Type: int
* returns: 0, on success
*/
int flattenMatrix(cv::Mat& m, std::vector<int>& flat) {
    try {
        int* datastart = m.ptr<int>(0);
        int* dataend = datastart + m.total();
        flat.assign(datastart, dataend);
    }
    catch (std::exception& ex) {
        std::cout << ex.what() << "\n";
        return -1;
    }
    return 0;
}

/*
* Gets a feature vector of the 9x9 center pixels.
* param src: image to compute features for
*    Image must be larger than 9x9
* param features: returned feature vector
* return: 0 on success
*/
int getCenter9x9(cv::Mat& src, std::vector<int>& features) {
    try {

        // copy to correct datatype
        cv::Mat srcInt;
        src.convertTo(srcInt, CV_32S);

        // extract the 9x9 square in the middle of the image
        cv::Mat centerRect = srcInt(cv::Rect(srcInt.cols / 2 - 4, srcInt.rows / 2 - 4, 9, 9));
        centerRect = centerRect.reshape(1);

        // save result
        flattenMatrix(centerRect, features);

    }
    catch (std::exception& ex) {
        std::cout << ex.what() << "\n";
        return -1;
    }
    return 0;
}

int getRGB(cv::Mat& src, std::vector<int>& features) {
    try {

//        if (src.channels() != 3) {
//            throw std::exception("3-channel image required to calculate RGB histogram.");
//        }

        // split RGB channels
        cv::Mat channels[3] = {};
        cv::split(src, channels);

        // compute 3D histogram
        int maxVals[] = { 255, 255, 255 };
        int ndims = 3;
        int dims[] = { 8, 8, 8, 1 };
        cv::Mat hist(ndims + 1, dims, CV_32SC1);
        calcHist(channels, maxVals, ndims, hist);

        // save return value
        flattenMatrix(hist, features);

    }
    catch (std::exception& ex) {
        std::cout << ex.what() << "\n";
        return -1;
    }
    return 0;
}

/*
* Gets a feature vector representing texture.
* Texture metric: sobel magnitude image
* param src: image to compute features for
* param features: returned feature vector
* return: 0 on success
*/
int getTexture(cv::Mat& src, std::vector<int>& features) {

    try {

        // get magnitude image
        cv::Mat sobelX(src.size(), CV_16SC3),
            sobelY(src.size(), CV_16SC3),
            mag(src.size(), CV_8UC3),
            orient(src.size(), CV_8UC3);
        fltr::sobelX3x3(src, sobelX);
        fltr::sobelY3x3(src, sobelY);
        fltr::magnitude(sobelX, sobelY, mag);
        fltr::orientation(sobelX, sobelY, orient);

        // compute 2D histogram
        cv::Mat srcArr[] = { mag, orient };
        int maxVals[] = { 255, 255 };
        int ndims = 2;
        int dims[] = { BINS, BINS, 3 };
        cv::Mat hist(ndims + 1, dims, CV_32SC1);
        calcHist(srcArr, maxVals, ndims, hist);

        // save return value
        flattenMatrix(hist, features);

    }
    catch (std::exception& ex) {
        std::cout << ex.what() << "\n";
        return -1;
    }

    return 0;

}



// MAIN FUNCTIONALITY

/*
* Offline phase.
* Extract features for all database images and save to a CSV file.
* param folder: path to folder of database images
* param csvpath: path to save output
*   CSV format: image_name, features
*       image_name: file name of image
*       features: space-delimited feature vector
* param extract: feature extract to use
* return: 0, on success
*/
int extractFeatures(cv::String folder, cv::String csvpath, FeatureExtractor extract) {

    try {

        // Load the image directory
        std::vector<cv::String> filenames;
        cv::glob(folder, filenames, "*.jpg");

        // Create an output file stream to save the feature vectors
        std::ofstream featureFile;
        featureFile.open(csvpath);
        featureFile << "image_name,features\n";

        // Loop through all the images in the directory
        for (int i = 0; i < filenames.size(); i++)
        {

            // Check if the file is not jpg
            if (std::regex_match(filenames[i], std::regex(".*[.]jpg")) == false)
            {
                continue;
            }
            // Load the image
            cv::Mat image = cv::imread(filenames[i]);

            // Check if the image was loaded correctly
            if (image.empty())
            {
                std::cout << "Failed to load image " << filenames[i] << std::endl;
                continue;
            }

            // Extract features
            std::vector<int> feats;
            extract(image, feats);

            // Write the image name to file
            std::string fileName = fs::path(filenames[i]).stem().string();
            featureFile << fileName << ",";

            // Write the feature vector to file
            for (int i = 0; i < feats.size(); i++) {
                featureFile << feats[i] << " ";
            }
            featureFile << std::endl;

            // Progress output
            if (i % 5 == 0) std::cout << "Processed " << i + 1 << " files" << std::endl;

        }

        featureFile.close();

    }
    catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return -1;
    }
    return 0;
}

int match(cv::Mat& target, DistanceMetric dist,
    std::vector<FeatureExtractor> extractors, std::vector<cv::String> csvs, std::vector<float> weights,
    int n, std::vector<cv::String>& matches) {

    try {
        // Check if the image was loaded correctly
        if (target.empty())
        {
            std::cout << "Failed to load target image" << std::endl;
            return -1;
        }

        // Extract the feature vector(s)
        std::vector<std::vector<int>> targetFeatures;
        for (FeatureExtractor extract : extractors) {
            std::vector<int> feat;
            extract(target, feat);
            targetFeatures.push_back(feat);
        }

        // Maps image name to distance
        std::unordered_map<cv::String, double> distancemap;

        for (int csvid = 0; csvid < csvs.size(); csvid++)
        {
            cv::String csvpath = csvs[csvid];
            float csvweight = weights[csvid];
            std::vector<int> targetVector = targetFeatures[csvid];

            // Load the feature vector file
            std::ifstream featureFile(csvpath);

            // Check if the feature vector file was opened correctly
            if (!featureFile.is_open())
            {
                std::cout << "Failed to open feature vector file" << std::endl;
                return -1;
            }

            // Read the image names and feature vectors from the file
            std::vector<std::string> imageNames;
            std::vector<std::vector<int>> imageVectors;
            std::string line;
            getline(featureFile, line); // throw away header
            while (getline(featureFile, line))
            {
                // create string stream from line
                std::stringstream ss(line);
                // parse image name
                std::string imageName;
                std::getline(ss, imageName, ',');
                // parse feature vector
                std::vector<int> feats;
                int val;
                while (ss >> val) {
                    feats.push_back(val);
                }
                // store this image's data
                imageNames.push_back(imageName);
                imageVectors.push_back(feats);
            }

            // Compute the distances between the target feature and all other features
            for (int i = 0; i < imageVectors.size(); i++)
            {
                distancemap[imageNames[i]] += csvweight * dist(imageVectors[i], targetVector);
            }

        }

        // Unzip the distancemap
        std::vector<std::string> imageNames;
        std::vector<double> distances;
        for (auto it = distancemap.begin(); it != distancemap.end(); it++) {
            imageNames.push_back(it->first);
            distances.push_back(it->second);
        }

        // Find the indices of the N most similar images
        std::vector<int> indices(distances.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int i, int j)
            { return distances[i] < distances[j]; });

        // Save results
        matches.clear();
        matches.resize(n);
        for (int i = 0; i < n; i++)
        {
            matches[i] = imageNames[indices[i]];

            // DEBUGGING
            std::cout << "Distance " << i << ": " << distances[indices[i]] << std::endl;
        }
    }
    catch (std::exception& ex) {
        std::cout << ex.what() << "\n";
        return -1;
    }

    return 0;
}

static const std::unordered_map<cv::String, FeatureExtractor> FEATURE_MAP = {
    {"BASELINE_9X9", getCenter9x9},
    {"HIST_RGB", getRGB},
    {"HIST_GRADIENT_2D", getTexture},
};
static const std::unordered_map<cv::String, DistanceMetric> DISTANCE_MAP = {
    {"HIST_INTERSECT", calcIntersection},
    {"SSE", calcSSE},
};

int main(int argc, char* argv[]) {

    if (argc == 1) {
        std::cout << "Usage:\n"
            << "\t-E dbpath csvpath featureset\n"
            << "\t-M targetpath distancemetric N featureset1 csvpath1 [w1 featureset2 csvpath2 w2...]\n";
        return 0;
    }


    // feature extraction or matching
    cv::String mode = argv[1];

    if (mode.compare("-E") == 0) {
        if (argc != 5) {
            std::cout << "Usage:\n"
                << "\t-E dbpath csvpath featureset\n"
                << "\t-M targetpath distancemetric N featureset1 csvpath1 [w1 featureset2 csvpath2 w2...]\n";
            return 0;
        }

        cv::String dbpath = argv[2];
        cv::String csvpath = argv[3];
        cv::String featureset = argv[4];

        FeatureExtractor extractor;
        if (auto search = FEATURE_MAP.find(featureset); search != FEATURE_MAP.end())
        {
            extractor = search->second;
        }
        else {
            std::cout << featureset << " is not an available featureset.\n";
            return -1;
        }

        extractFeatures(dbpath, csvpath, extractor);
    }


    else if (mode.compare("-M") == 0) {

        if (argc < 7) {
            std::cout << "Usage:\n"
                << "\t-E dbpath csvpath featureset\n"
                << "\t-M targetpath distancemetric N featureset1 csvpath1 [w1 featureset2 csvpath2 w2...]\n";
            return 0;
        }

        cv::String targetpath = argv[2];

        cv::String distancemetric = argv[3];
        DistanceMetric dist;
        if (auto search = DISTANCE_MAP.find(distancemetric); search != DISTANCE_MAP.end())
        {
            dist = search->second;
        }
        else {
            std::cout << distancemetric << " is not an available distancemetric.\n";
            return -1;
        }

        int n = atoi(argv[4]);

        int featureargs = argc - 5;
        if (featureargs != 2 && featureargs % 3 != 0) {
            std::cout << "Multiple features specified, but weights are mismatched.\n"
                << "Usage:\n"
                << "\t-E dbpath csvpath featureset\n"
                << "\t-M targetpath distancemetric N featureset1 csvpath1 [w1 featureset2 csvpath2 w2...]\n";
            return 0;
        }

        std::vector<FeatureExtractor> extractors;
        std::vector<cv::String> csvs;
        std::vector<float> weights;
        for (int i = 0; i < featureargs; i += 3) {
            cv::String featureset = argv[5 + i];
            if (auto search = FEATURE_MAP.find(featureset); search != FEATURE_MAP.end())
            {
                extractors.push_back(search->second);
            }
            else {
                std::cout << featureset << " is not an available featureset.\n";
                return -1;
            }
            cv::String path = argv[6 + i];
            csvs.push_back(path);
            if (featureargs == 2) {
                weights.push_back(1.0);
            }
            else {
                cv::String weightstr = argv[7 + i];
                try {
                    weights.push_back(std::stof(weightstr));
                }
                catch (std::exception& ex) {
                    std::cout << "Invalid weight argument: " << weightstr << std::endl;
                    return -1;
                }
            }
        }

        // Load the target image
        cv::Mat target = cv::imread(targetpath);

        // Check if the image was loaded correctly
        if (target.empty())
        {
            std::cout << "Failed to load target image " << targetpath << std::endl;
            return -1;
        }

        std::vector<cv::String> matches;
        match(target, dist, extractors, csvs, weights, n, matches);

        // Print the names of the N most similar images
        //std::string dbpath = "D:/media/photos/db/olympus/";  // DEBUGGING
        std::cout << "TARGET:\t" << targetpath << std::endl;
        std::cout << "RESULTS:" << std::endl;
        for (int i = 0; i < n; i++)
        {
            cv::String match = matches[i];
            //cv::String matchpath = dbpath + match + ".jpg";
            std::cout << i << "\t" << match << std::endl;

            //cv::Mat matchimg = cv::imread(matchpath);
            //cv::imshow(match, matchimg);

            // DEBUGGING
            //cv::Mat centerRect = matchimg(cv::Rect(matchimg.cols / 2 - 4, matchimg.rows / 2 - 4, 9, 9));
            //cv::imshow(match + "_centered", centerRect);
        }

    }


    else {
        std::cout << "Usage:\n"
            << "\t-E dbpath csvpath featureset\n"
            << "\t-M targetpath distancemetric N featureset1 csvpath1 [w1 featureset2 csvpath2 w2...]\n";
        return 0;
    }

    cv::waitKey(0);
    return 0;
}
