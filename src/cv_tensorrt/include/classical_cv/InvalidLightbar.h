//
// Created by michael on 6/28/23.
//

#ifndef BBEXTRACTION__INVALIDLIGHTBAR_H_
#define BBEXTRACTION__INVALIDLIGHTBAR_H_

#include <exception>

class InvalidLightbar : std::exception {
 public:
  explicit InvalidLightbar() {

  }
};

#endif //BBEXTRACTION__INVALIDLIGHTBAR_H_
