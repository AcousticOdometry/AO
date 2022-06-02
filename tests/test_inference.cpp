#include <gtest/gtest.h>

#include "common.hpp"

TEST(TestInference, LibTorch) {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
}