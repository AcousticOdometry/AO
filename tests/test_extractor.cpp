#include "extractor.hpp"

// #include <gammatone/filter.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <ctime>
#include <iostream>

std::vector<float> example_input{ 582, 571, 561, 557, 555, 539, 536, 527, 502,
    467, 421, 380, 337, 296, 257, 211, 172, 127, 80, 35, -7, -66, -115, -160,
    -212, -256, -316, -366, -419, -468, -514, -558, -599, -631, -658, -688,
    -708, -738, -766, -783, -810, -832, -857, -885, -912, -920, -921, -912,
    -908, -900, -896, -889, -882, -882, -872, -877, -868, -850, -836, -820,
    -816, -801, -778, -753, -729, -705, -690, -675, -655, -634, -622, -613,
    -602, -597, -580, -568, -557, -541, -531, -527, -511, -503, -505, -498,
    -493, -482, -477, -462, -456, -449, -448, -446, -452, -452, -448, -447,
    -439, -433, -431, -425, -429, -422, -401, -380, -356, -333, -318, -307,
    -297, -288, -279, -264, -238, -210, -188, -161, -140, -119, -92, -72, -54,
    -40, -8, 16, 42, 72, 107, 138, 174, 211, 243, 276, 309, 348, 375, 393, 403,
    421, 432, 448, 475, 498, 530, 561, 595, 617, 637, 648, 663, 674, 681, 694,
    705, 718, 726, 744, 755, 767, 769, 782, 781, 768, 764, 737, 705, 672, 634,
    591, 540, 485, 427, 380, 354, 329, 300, 249, 184, 124, 62, -8, -83, -168,
    -233, -287, -328, -384, -445, -495, -544, -579, -611, -654, -690, -734,
    -770, -806, -839, -874, -900, -923, -934, -930, -921, -909, -902, -903,
    -899, -890, -892, -905, -903, -891, -873, -844, -816, -812, -803, -794,
    -762, -730, -704, -684, -675, -671, -673, -669, -667, -654, -632, -610,
    -579, -553, -534, -518, -516, -521, -523, -514, -503, -487, -479, -477,
    -483, -487, -495, -489, -486, -481, -457, -452 };

TEST(TestExtractor, GammatoneFilterBank) {
    auto extractor = ao::extractor::GammatoneFilterbank<float>(
        /* num_samples */ 250,
        /* num_features */ 64,
        /* sample_rate */ 25000,
        /* low_Hz */ 50,
        /* high_Hz */ 8000);
    std::cout << "Center frequencies: ";
    for (auto& filter : extractor.filters) {
        std::cout << filter.cf << " ";
    }
    std::cout << std::endl;
    // Example input
    std::vector<float> input = example_input;
    // for (auto& i : input) {
    //     std::cout << i << " ";
    // }
    // std::cout << std::endl;
    // Allocate output and compute
    std::vector<float> output  = extractor.compute(input);
    std::transform(output.begin(), output.end(), output.begin(),
        [](auto& o) { return std::log10(o); });
    for (auto& o : output) {
        std::cout << o << " ";
    }
    std::cout << std::endl;
}

/**
 * @brief Extractors overload the feature "compute" method with several
 * signatures. Test that all of them work and return the same values.
 * 
 */
TEST(TestExtractor, ComputeOverloads) {
    // TODO parametrize with different extractors
    auto extractor = ao::extractor::GammatoneFilterbank<float>(
        /* num_samples */ 250,
        /* num_features */ 64,
        /* sample_rate */ 25000,
        /* low_Hz */ 50,
        /* high_Hz */ 8000);
    // TODO parametrize input
    std::vector<float> input = example_input;
    std::vector<float> output(extractor.num_features);
    // compute(const std::vector<T>& input, std::vector<T>& output)
    ASSERT_NO_THROW(extractor.compute(input, output));
    // compute(const std::vector<T>& input)
    EXPECT_EQ(output, extractor.compute(input));
    // operator()
    EXPECT_EQ(output, extractor(input));
}