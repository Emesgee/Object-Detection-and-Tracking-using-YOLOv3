
Object Detection and Tracking using YOLOv3
This C++ program demonstrates real-time object detection and tracking using the YOLOv3 model. It utilizes OpenCV libraries for both object detection and tracking.

Requirements
OpenCV (4.x or higher)
C++ compiler supporting C++11
CMake (for building)

Build the executable using CMake and insert the input image in the Release folder.
In order to build the executable please follow these steps:
1: Unzip the file
2: Create a build folder inside the unzipped folder
3: In the command line type: cmake -G "Visual Studio 16 2019" ..
You can choose your own VS version in my case it was:  cmake -G "Visual Studio 17 2022" .. '
4: Now, to run the executable: ..\build\Release\submission.exe

How to Use
1: Ensure you have OpenCV installed on your system.
2:Compile the program using any C++ compiler that supports C++11.
3:Place the soccer-ball.mp4 video file in the same directory as the executable or provide the correct path to the video file.
4:Place the YOLOv3 configuration file (yolov3.cfg), YOLOv3 weights file (yolov3.weights), and COCO class names file (coco.names) in the models directory within the project directory.
5:Run the compiled executable.

Description
- The program reads a video file frame by frame and performs object detection using the YOLOv3 model.
- Detected objects of interest, particularly sports balls, are tracked using the KCF tracker.
- The program outputs the processed video with bounding boxes drawn around detected objects and tracking information overlayed.

Parameters
- objectnessThreshold: Threshold for objectness in YOLOv3.
- confThreshold: Confidence threshold for detected objects.
- nmsThreshold: Non-maximum suppression threshold.
- inpWidth and inpHeight: Width and height of the input image to the neural network.

Functions
- postprocess: Removes bounding boxes with low confidence using non-maximum suppression and tracks detected objects.
- drawPred: Draws predicted bounding boxes around detected objects.
- getOutputsNames: Retrieves the names of the output layers from the neural network.

Files
- soccer-ball.mp4: Sample video file for object detection and tracking.
- yolov3.cfg: YOLOv3 configuration file.
- yolov3.weights: Pre-trained YOLOv3 weights.
- coco.names: File containing COCO class names.

Notes
- Ensure that the paths to the video file, YOLOv3 configuration file, YOLOv3 weights file, and COCO class names file are correctly specified.
- The program expects the sports ball class to be present in the COCO class names file.

Credits
This application was developed by Mohammad Ghadban.

Cheers!