#include "extractor.cpp"
#include "extractor.hpp"

// #include <gammatone/filter.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <complex>
#include <ctime>
#include <iostream>

// TODO propper fixtures
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

std::vector<float> invalid_input{ 2559, 2728, 2890, 2935, 3037, 3135, 3295,
    3382, 3388, 3551, 3553, 3456, 3308, 3094, 2916, 2772, 2608, 2290, 2132,
    2120, 2263, 2374, 2268, 2007, 1554, 1294, 1002, 763, 596, 464, 591, 328,
    -21, -521, -1096, -1288, -1339, -1260, -1419, -1543, -1467, -1476, -1384,
    -1480, -1649, -2024, -2177, -1926, -1756, -1634, -1581, -1839, -2124, -2109,
    -1885, -1707, -1584, -1506, -1412, -1238, -929, -610, -616, -833, -830,
    -648, -385, -105, 15, -50, -166, -153, -61, -41, 14, 163, 279, 372, 474,
    535, 527, 459, 419, 391, 375, 409, 422, 395, 248, 128, 54, -64, -168, -225,
    -213, -207, -187, -200, -339, -470, -515, -499, -526, -588, -632, -703,
    -749, -782, -861, -960, -1066, -1077, -1045, -1019, -989, -1022, -1066,
    -1158, -1142, -1075, -1021, -940, -945, -951, -998, -978, -961, -1073,
    -1045, -1031, -1055, -975, -912, -860, -843, -745, -678, -735, -695, -672,
    -605, -565, -525, -518, -645, -606, -598, -605, -573, -624, -617, -608,
    -519, -433, -460, -462, -482, -510, -500, -446, -444, -517, -555, -604,
    -650, -659, -624, -626, -636, -574, -495, -469, -453, -386, -379, -365,
    -295, -252, -205, -143, -13, 60, 101, 178, 264, 370, 493, 632, 730, 808,
    972, 1111, 1255, 1421, 1546, 1656, 1788, 1954, 2076, 2233, 2396, 2500, 2647,
    2824, 3008, 3135, 3287, 3411, 3509, 3619, 3771, 3906, 4002, 4149, 4281,
    4390, 4518, 4591, 4598 };

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
    std::vector<float> output = extractor.compute(input);
    EXPECT_EQ(output, extractor.compute(input)); // Test second execution
    // std::transform(output.begin(), output.end(), output.begin(),
    //     [](auto& o) { return std::log10(o); });
    // for (auto& o : output) {
    //     std::cout << o << " ";
    // }
    // Invalid input
    EXPECT_THROW(extractor.compute(invalid_input), std::runtime_error);
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