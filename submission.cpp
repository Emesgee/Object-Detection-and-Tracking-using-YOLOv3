#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/tracking.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

// initialize the parameters for object detection
double objectnessThreshold = 0.5;    // Objectness threshold
float confThreshold = 0.5;          // Confidence threshold
double nmsThreshold = 0.2;           // Non-maximum suppression threshold
int inpWidth = 416;                  // Width of network's input image
int inpHeight = 416;                 // Height of network's input image

// array of class names
vector<string> classes;

// remove bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, vector<Mat>& outs, map<int, Ptr<Tracker>>& trackers, map<int, bool>& tracked);

// draw the predicted bounding box
void drawPred(int classId, double conf, int left, int top, int right, int bottom, Mat& frame, bool tracked);

// get the names if of the output layers 
vector<String> getOutputsNames(const Net& net);


int main() {

    // load names classes
    ifstream ifs("../../models/coco.names");
    string line;
    while (getline(ifs, line)) {
        cout << "Read class: " << line << endl;
        classes.push_back(line);
    }

    // Load the DNN
    Net net = readNetFromDarknet("../../models/yolov3.cfg", "../../models/yolov3.weights");

    // load video
    VideoCapture cap("soccer-ball.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file.\n";
        return -1;
    }
    Mat frame, blob;
    map<int, Ptr<Tracker>> trackers;
    map<int, bool> tracked;

    // initiate videowriter
    VideoWriter writer;
    int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
    double fps = 25.0;  
    string filename = "output%d.avi";
    writer.open(filename, codec, fps, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));

    while (cap.read(frame)) {

        // Create a 4D blob from a frame.
        blobFromImage(frame, blob, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

        //Sets the input to the network
        net.setInput(blob);

        // Runs the forward pass to get output of the output layers
        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));

        // Remove bounding boxes with low confidence using non-maximum suppression
        postprocess(frame, outs, trackers, tracked);

        // put effeciency information the function getPerProfile returns the overall time for inference(t)
        // and the timings for each of the layers (in layersTimes)
        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        string label = format("Inference time for a frame : %.2f ms", t);
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 225));

        imshow("Object Detection and Tracking", frame);
        

        writer.write(frame);

        // Display the frame with bounding boxes or tracker traces
       
        int delay = 1000 / fps;
        // Check for ESC key press to exit
        if (cv::waitKey(delay) == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

// Remove bounding boxes with low confidence using non-maximum suppression
void postprocess(Mat& frame, vector<Mat>& outs, map<int, Ptr<Tracker>>& trackers, map<int, bool>& tracked)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i) {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold) {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                // Ensure bounding box coordinates are within image boundaries
                left = max(left, 0);
                top = max(top, 0);
                width = min(width, frame.cols - left);
                height = min(height, frame.rows - top);

                Rect box(left, top, width, height);
                int classId = classIdPoint.x;
                if (classes[classId] == "sports ball") // Check if the detected class is "sports ball"
                {
                    classIds.push_back(classId);
                    confidences.push_back((float)confidence);
                    boxes.push_back(box);
                    // Check if tracker for this object already exists
                    if (trackers.find(classId) == trackers.end()) {
                        // If not, initialize a new tracker
                        trackers[classId] = TrackerKCF::create();
                        trackers[classId]->init(frame, box);
                        tracked[classId] = true; // Set to true if we have tracking information
                    }
                    else {
                        // If tracker exists, update it
                        if (trackers[classId]->update(frame, box)) {
                            tracked[classId] = true;
                            string labelTracking = format("Ball tracking running...");
                            putText(frame, labelTracking, Point(0, 45), FONT_HERSHEY_PLAIN, 1.5, Scalar(0, 255, 0));
                        }
                        else {
                            tracked[classId] = false;
                            string labelDetecting = format("Ball detection running...");
                            putText(frame, labelDetecting, Point(0, 45), FONT_HERSHEY_PLAIN, 1.5, Scalar(255,0,0));
                        }


                    }

                }
            }
        }
    }

    // Check if there are no bounding boxes detected
    if (boxes.empty()) {
        // Display "Ball Lost running..." as no bounding boxes were found in the frame
        string labelLost = format("Ball Lost running...");
        putText(frame, labelLost, Point(0, 45), FONT_HERSHEY_PLAIN, 1.5, Scalar(0, 0, 255));
    }


    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences.
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
            box.x + box.width, box.y + box.height, frame, tracked[classIds[idx]]);
    }
}

// Draw the predicted bounding box for detection
void drawPred(int classId, double conf, int left, int top, int right, int bottom, Mat& frame, bool tracked) {
    // Draw a rectangle around the detected object
    Scalar color = tracked ? Scalar(0, 255, 0) : Scalar(255, 0, 0);
    rectangle(frame, Point(left, top), Point(right, bottom), color, 3);

    // Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty()) {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    // Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
        Point(left + labelSize.width, top + baseLine),
        Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top),
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net) {
    static vector<String> names;
    if (names.empty()) {
        // Get the indices of the output layers, i.e., the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();

        // Get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i) {
            names[i] = layersNames[outLayers[i] - 1];
        }
    }
    return names;
}
