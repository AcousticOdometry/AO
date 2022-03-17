#include "extractor.hpp"

// #include <gammatone/filter.hpp>
#include <gtest/gtest.h>

#include <iostream>
#include <algorithm>
#include <ctime>

TEST(TestExtractor, GammatoneFilterBank) {
    auto extractor = ao::extractor::GammatoneFilterbank<int>();
    std::cout << extractor.num_samples << std::endl;
    // Random input
    std::srand(unsigned(std::time(nullptr)));
    std::vector<int> input(extractor.num_samples);
    std::generate(input.begin(), input.end(), std::rand);
    // for (auto& i : input) {
    //     std::cout << i << " ";
    // }
    // std::cout << std::endl;
    // Allocate output and compute
    std::vector<int> output(extractor.num_features);
    extractor.compute(input, output);
    for (auto& o : output) {
        std::cout << o << " ";
    }
    std::cout << std::endl;
}