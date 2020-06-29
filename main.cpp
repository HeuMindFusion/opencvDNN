#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>  
#include <opencv2/videoio.hpp>
#include <iostream>
#include <string>
#include "config.hpp"

//using namespace cv;
//using namespace std;

using namespace cv::dnn;
int main()
{
    std::cout << "Built with OpenCV " << CV_VERSION << std::endl;
    cv::Mat frame;
    cv::VideoCapture capture(0);
    if (!capture.isOpened())
    {
        std::cout << "open camera failed. " << std::endl;
        return -1;
    }
    Net net = readNetFromCaffe(ROOT_DIR"/WeightFile/MobileNetSSD_deploy.prototxt",
        ROOT_DIR"/WeightFile/MobileNetSSD_deploy.caffemodel");
    while (true)
    {
        cv::Mat frame;
        capture >> frame;
        if (!frame.empty())
        {
           cv::Mat inputBlob = blobFromImage(frame, inScaleFactor,
                cv::Size(inWidth, inHeight), meanVal, false);
            net.setInput(inputBlob, "data");
            cv::Mat detection = net.forward("detection_out");

            cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

            float confidenceThreshold = 0.7f;
            for (int i = 0; i < detectionMat.rows; i++)
            {
                float confidence = detectionMat.at<float>(i, 2);

                if (confidence > confidenceThreshold)
                {
                    size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

                    int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                    int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                    int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                    int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

                    cv::Rect object((int)xLeftBottom, (int)yLeftBottom,
                        (int)(xRightTop - xLeftBottom),
                        (int)(yRightTop - yLeftBottom));
                    std::ostringstream ss;
                    ss << classNames[objectClass] << " ";

                    cv::String conf(ss.str());

                    rectangle(frame, object, cv::Scalar(0, 255, 0));
                    int baseLine = 0;
                    cv::Size labelSize = cv::getTextSize(ss.str(), cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseLine);

                    auto center = (object.br() + object.tl()) * 0.5;
                    center.x = center.x - labelSize.width / 2;

                    rectangle(frame, cv::Rect(cv::Point(center.x, center.y - labelSize.height),
                        cv::Size(labelSize.width, labelSize.height + baseLine)),
                        cv::Scalar(255, 255, 255), -1);
                    putText(frame, ss.str(), center,
                        3, 0.7, cv::Scalar(0, 0, 255));
                }

                cv::imshow("camera", frame);

            }
            if (cv::waitKey(30) > 0)
            {
                break;
            }

        }

    }
    return 0;
}
