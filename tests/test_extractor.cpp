#include "extractor.hpp"

// #include <gammatone/filter.hpp>
#include <gtest/gtest.h>

#include <iostream>

// TEST(Libgammatone, Minimal) {
//     // Init a gammatone filter sampled at 44.1 kHz, centered at 1 kHz
//     gammatone::filter<double> filter(44100, 1000);

//     // An input signal (here 1000 times zero, silly but minimal)
//     std::vector<double> input(1000, 0.0);

//     // Init an output buffer to store the filter response
//     std::vector<double> output(1000);

//     // Compute the output signal from input
//     filter.compute_range(input.begin(), input.end(), output.begin());

//     std::cout << "Output: " << std::endl;
//     for (auto& o : output) {
//         std::cout << o << " ";
//     }
//     std::cout << std::endl;
// }

TEST(TestExtractor, GammatoneFilterBank) {
    auto extractor = ao::extractor::GammatoneFilterbank<int>();
    std::cout << extractor.num_samples << std::endl;
}