//
// Created by michael on 6/9/23.
//

#ifndef BBEXTRACTION_IMUTILS_H
#define BBEXTRACTION_IMUTILS_H

#include <opencv2/opencv.hpp>
#include <functional>

using namespace std::placeholders;

enum ARMOR_SIZE {
  LARGE, SMALL
};

enum ARMOR_COLOR {
  BLUE_ARMOR, RED_ARMOR
};

namespace cv_constants {
constexpr double MIN_LIGHTBAR_HEIGHT_RATIO = 0.005;
constexpr double MAX_VERT_ASPECT_RATIO = 1.15;
constexpr double MIN_LIGHTBAR_BOX_RATIO = 0.0005;
constexpr auto inWhiteRange = [](auto &&PH1, auto &&PH2) {
  return cv::inRange(std::forward<decltype(PH1)>(PH1),
                     std::vector<int>{0, 0, 70},
                     std::vector<int>{255, 80, 255},
                     std::forward<decltype(PH2)>(PH2));
};
constexpr auto inBlueRange = [](auto &&PH1, auto &&PH2) {
  return cv::inRange(std::forward<decltype(PH1)>(PH1),
                     std::vector<int>{80, 90, 140},
                     std::vector<int>{110, 255, 255},
                     std::forward<decltype(PH2)>(PH2));
};
constexpr auto inRedRange = [](auto &&PH1, auto &&PH2) {
  return cv::inRange(std::forward<decltype(PH1)>(PH1),
                     std::vector<int>{100, 100, 128},
                     std::vector<int>{120, 255, 255},
                     std::forward<decltype(PH2)>(PH2));
};

}

class Coords {
 public:
  int32_t x;
  int32_t y;

  Coords operator+(const Coords &) const;
  Coords operator*(double) const;
  Coords operator/(double) const;
  Coords operator-(const Coords &) const;
  explicit Coords(int32_t x, int32_t y) : x{x}, y{y} {}
  explicit Coords(std::vector<int>);
  cv::Point2f toopencvpoint();
};

class LightBar {
 public:
  Coords top;
  Coords bottom;
};

class ArmorPanel {
 public:
  LightBar left;
  LightBar right;
};

class OpResult {
 public:
  cv::Mat3b image;
  std::function<Coords(Coords)> translator;
};

class CropResult : public OpResult {
};

class SplitResult : public OpResult {
};

ArmorPanel get_key_points(const cv::Mat3b& image, Coords topleft, Coords bottomright, ARMOR_COLOR color);

double get_panel_coeff(ArmorPanel a, const cv::Mat3b &image);

bool valid_armorpanel(ArmorPanel a, const cv::Mat &image);

bool valid_bb(Coords topleft, Coords bottomright);

std::optional<std::pair<ArmorPanel, ARMOR_SIZE>> getArmorPanelStats(Coords topleft,
                                                                    Coords bottomright,
                                                                    const cv::Mat3b &image, ARMOR_COLOR color);

cv::Mat3b clahe_enhanced_image(const cv::Mat3b& image);

#endif //BBEXTRACTION_IMUTILS_H
