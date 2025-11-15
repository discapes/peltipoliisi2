#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;

int main(int argc, char **argv) {
  if (argc != 1) {
    std::cout << argv[0] << " takes no arguments.\n";
    return 1;
  }
  cout << "initializing...\n";

  // Create an empty 400x400 blue image
  cv::Mat img(400, 400, CV_8UC3, cv::Scalar(255, 0, 0));

  // Create a window
  cv::namedWindow("MyWindow", cv::WINDOW_AUTOSIZE);

  // Show the image
  cv::imshow("MyWindow", img);

  // Wait until a key is pressed
  cv::waitKey(0);

  return 0;
}
