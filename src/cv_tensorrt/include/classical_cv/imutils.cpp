//
// Created by michael on 6/9/23.
//

#include "imutils.h"
#include <cmath>
#include "InvalidLightbar.h"

using std::vector;
using namespace cv;

Coords Coords::operator+(const Coords &other) const {
  return Coords{this->x + other.x, this->y + other.y};
}

Coords Coords::operator*(double d) const {
  return Coords{static_cast<int32_t>(this->x * d), static_cast<int32_t>(this->y * d)};
}

Coords Coords::operator/(double d) const {
  return *this * (1 / d);
}

Coords Coords::operator-(const Coords &other) const {
  return Coords{this->x - other.x, this->y - other.y};
}

Coords::Coords(std::vector<int> vec) {
  x = vec[0];
  y = vec[1];
}
cv::Point2f Coords::toopencvpoint() {
  return cv::Point2f{static_cast<float>(x), static_cast<float>(y)};
}

CropResult crop_interest_region(const Mat3b &image, Coords topleft, Coords bottomright) {
  auto translator = [topleft](Coords incrop) -> Coords {
    return incrop + topleft;
  };

  return CropResult{image(Range(topleft.y, bottomright.y), Range(topleft.x, bottomright.x)), translator};
}

std::pair<SplitResult, SplitResult> split_down_middle(const Mat3b &image) {
  int32_t width = image.cols;
  int32_t halfwidth = image.cols / 2;
  auto lefttranslator = [](Coords incrop) -> Coords {
    return incrop;
  };
  auto rightranslator = [halfwidth](Coords incrop) -> Coords {
    return Coords{incrop.x + halfwidth, incrop.y};
  };

  return std::pair{SplitResult{image(Range::all(), Range(0, halfwidth - 1)), lefttranslator},
                   SplitResult{image(Range::all(), Range(halfwidth, width)), rightranslator}};
}

Mat1b get_mask(const Mat3b &image, ARMOR_COLOR color) {
  Mat3b hsv_image;
  Mat mask;
  if (color == BLUE) {
    cvtColor(image, hsv_image, COLOR_BGR2HSV);
    cv_constants::inBlueRange(hsv_image, mask);
  } else {
    cvtColor(image, hsv_image, COLOR_RGB2HSV);
    cv_constants::inRedRange(hsv_image, mask);
  }
  cv::Mat3b switched;
  cvtColor(image, switched, COLOR_BGR2RGB);
//  imshow("Display window 2", switched);
//  cv::waitKey(0);
//  imshow("Display window 2", mask);
//  cv::waitKey(0);
  Mat kernel{Size(5, 5), CV_64FC1, Scalar(1.0)};
  Mat reshaped_mask;

  erode(mask, reshaped_mask, kernel);
  dilate(reshaped_mask, reshaped_mask, kernel);
  return static_cast<Mat1b>(reshaped_mask);
}

LightBar box_points_to_lightbar(const vector<Coords> &coordsvec) {
  vector<double> dist{};
  for (int i = 1; i <= 2; ++i) {
    Coords delta = coordsvec[i] - coordsvec[i - 1];
    dist.push_back(pow(delta.x, 2.0) + pow(delta.y, 2.0));
  }
  if (dist[0] < dist[1]) {
    Coords midpt1 = (coordsvec[0] + coordsvec[1]) / 2;
    Coords midpt2 = (coordsvec[2] + coordsvec[3]) / 2;
    return midpt1.y < midpt2.y ? LightBar{midpt1, midpt2} : LightBar{midpt2, midpt1};
  } else {
    Coords midpt1 = (coordsvec[1] + coordsvec[2]) / 2;
    Coords midpt2 = (coordsvec[0] + coordsvec[3]) / 2;
    return midpt1.y < midpt2.y ? LightBar{midpt1, midpt2} : LightBar{midpt2, midpt1};
  }
}

vector<Coords> box_points_to_coords(Point2f *coords2dvec) {
  vector<Coords> coords{};
  for (int i = 0; i < 4; ++i) {
    coords.emplace_back(coords2dvec[i].x, coords2dvec[i].y);
  }
  return coords;
}

LightBar get_lightbar_in_split(const SplitResult &splitResult, ARMOR_COLOR color) {
  Mat1b mask{get_mask(splitResult.image, color)};
  Rect rect = cv::boundingRect(mask);
  if (static_cast<double>(rect.area()) / (mask.rows * mask.cols) < cv_constants::MIN_LIGHTBAR_BOX_RATIO) {
    throw InvalidLightbar{};
  }
  vector<Coords> coords{};
  coords.emplace_back(rect.x, rect.y);
  coords.emplace_back(rect.x + rect.width, rect.y);
  coords.emplace_back(rect.x + rect.width, rect.y + rect.height);
  coords.emplace_back(rect.x, rect.y + rect.height);
  return box_points_to_lightbar(coords);
}

ArmorPanel get_key_points(cv::Mat3b image, Coords topleft, Coords bottomright, ARMOR_COLOR color) {
  CropResult cropResult{crop_interest_region(image, topleft, bottomright)};
  auto [leftSplit, rightSplit] = split_down_middle(cropResult.image);
  LightBar leftlb = get_lightbar_in_split(leftSplit, color);
  LightBar rightlb = get_lightbar_in_split(rightSplit, color);
  return ArmorPanel{LightBar{cropResult.translator(leftSplit.translator(leftlb.top)),
                             cropResult.translator(leftSplit.translator(leftlb.bottom)),},
                    LightBar{cropResult.translator(rightSplit.translator(rightlb.top)),
                             cropResult.translator(rightSplit.translator(rightlb.bottom))}};
}

bool valid_bb(Coords topleft, Coords bottomright) {
  double height = bottomright.y - topleft.y;
  double width = bottomright.x - topleft.x;
  if (height > width) {
    return (height / width) > cv_constants::MAX_VERT_ASPECT_RATIO;
  }
  return true;
}

bool valid_armorpanel(ArmorPanel a, const cv::Mat &image) {
  Coords leftdelta = a.left.top - a.left.bottom;
  double leftdist = sqrt(pow(leftdelta.x, 2.0) + pow(leftdelta.y, 2.0));
  Coords rightdelta = a.right.top - a.right.bottom;
  double rightdist = sqrt(pow(rightdelta.x, 2.0) + pow(rightdelta.y, 2.0));
  double minimagedim = min(image.rows, image.cols);
  return (min(leftdist, rightdist) / minimagedim) > cv_constants::MIN_LIGHTBAR_HEIGHT_RATIO;
}

double get_panel_coeff(ArmorPanel a, const cv::Mat3b &image) {
  Range armorpanelys{min(a.left.top.y, a.right.top.y), max(a.left.bottom.y, a.right.bottom.y)};
  Range armorpanelxs{min(a.left.top.x, a.left.bottom.x), max(a.right.top.x, a.left.top.x)};
  cv::Mat3b crop = image(armorpanelys, armorpanelxs);
  cv::Mat3b hsv_image;
  cv::Mat1b mask;
  cvtColor(crop, hsv_image, COLOR_BGR2HSV);
  cv_constants::inWhiteRange(hsv_image, mask);

  cv::Mat out;
  cv::merge(std::vector{mask, mask, mask}, out);
//  cv::rectangle(out, rect, cv::Scalar{255, 0, 0}, 5);
//  cv::imshow("Display window 2", out);

  vector<vector<Point>> contours{};
  findContours(mask, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
  size_t maxcontourindex = 0;
  if (contours.empty()) {
    throw InvalidLightbar{};
  }
  for (int i = 1; i < contours.size(); ++i) {
    if (cv::contourArea(contours[i]) > cv::contourArea(contours[maxcontourindex])) {
      maxcontourindex = i;
    }
  }

  auto rect = cv::boundingRect(contours[maxcontourindex]);
  cv::waitKey(0);

  return static_cast<double>(armorpanelxs.size() - rect.width) / armorpanelxs.size();
}

std::optional<std::pair<ArmorPanel, ARMOR_SIZE>> getArmorPanelStats(Coords topleft,
                                                                    Coords bottomright,
                                                                    const cv::Mat3b &image, ARMOR_COLOR color) {
  auto empty = std::optional<std::pair<ArmorPanel, ARMOR_SIZE>>{};
  if (!valid_bb(topleft, bottomright)) {
    return empty;
  }
  try {
    ArmorPanel keypoints = get_key_points(image, topleft, bottomright, color);
    if (!valid_armorpanel(keypoints, image)) {
      return empty;
    }
    double widthcoeff = get_panel_coeff(keypoints, image);
    ARMOR_SIZE size = SMALL;
    if (widthcoeff < .6) {
      size = SMALL;
    } else {
      size = LARGE;
    }
    return std::optional{std::pair{keypoints, size}};
  } catch (InvalidLightbar &i) {
    return empty;
  } catch (cv::Exception &c) {
    return empty;
  }
}