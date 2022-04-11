#include "extractor.cpp"
#include "extractor.hpp"

#include <gtest/gtest.h>
#include <sndfile.h>

#include <algorithm>
#include <complex>
#include <ctime>
#include <filesystem>
#include <iostream>

// Audio fixture
std::filesystem::path AUDIO_PATH =
    std::filesystem::path(__FILE__).parent_path() / "data" / "audio0.wav";

TEST(TestWav, WavIntoVector) {
    SF_INFO file_info;
    SNDFILE* file  = sf_open(AUDIO_PATH.string().c_str(), SFM_READ, &file_info);
    int samplerate = file_info.samplerate;
    if (sf_error(file) || samplerate < 0) {
        throw std::runtime_error(sf_strerror(file));
        sf_close(file);
    }
    std::vector<float> empty_data(250), audio_data(250);
    sf_read_float(file, audio_data.data(), audio_data.size());
    ASSERT_NE(audio_data, empty_data);
}

TEST(TestExtractor, GammatoneFilterBank) {
    // Example input
    int num_samples = 250;
    SF_INFO file_info;
    SNDFILE* file = sf_open(AUDIO_PATH.string().c_str(), SFM_READ, &file_info);
    std::vector<float> input(num_samples);
    sf_read_float(file, input.data(), input.size());

    // Construct extractor
    auto extractor = ao::extractor::GammatoneFilterbank<float>(
        /* num_samples */ num_samples,
        /* num_features */ 64,
        /* sample_rate */ file_info.samplerate,
        /* low_Hz */ 50,
        /* high_Hz */ 8000);
    std::cout << "Center frequencies: ";
    for (auto& filter : extractor.filters) {
        std::cout << filter.cf << " ";
    }
    std::cout << std::endl;

    // Execute extractor
    std::vector<float> output = extractor.compute(input);
    EXPECT_EQ(output, extractor.compute(input)); // Test second execution

    // Invalid input (too short)
    std::vector<float> invalid_input(num_samples-1);
    sf_read_float(file, invalid_input.data(), invalid_input.size());
    EXPECT_THROW(extractor.compute(invalid_input), std::invalid_argument);
}

/**
 * @brief Extractors overload the feature "compute" method with several
 * signatures. Test that all of them work and return the same values.
 *
 */
TEST(TestExtractor, ComputeOverloads) {
    // TODO parametrize input
    int num_samples = 250;
    SF_INFO file_info;
    SNDFILE* file = sf_open(AUDIO_PATH.string().c_str(), SFM_READ, &file_info);
    std::vector<float> input(num_samples);
    sf_read_float(file, input.data(), input.size());

    // TODO parametrize with different extractors
    auto extractor = ao::extractor::GammatoneFilterbank<float>(
        /* num_samples */ num_samples,
        /* num_features */ 64,
        /* sample_rate */ file_info.samplerate,
        /* low_Hz */ 50,
        /* high_Hz */ 8000);
    std::vector<float> output = extractor.compute(input);
    // operator()
    EXPECT_EQ(output, extractor(input));
}