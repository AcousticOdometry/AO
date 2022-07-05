#include "ao/extractor.hpp"

#include <eigen3/Eigen/Core>
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
    std::vector<float> signal(num_samples * file_info.channels);
    sf_readf_float(file, signal.data(), num_samples);

    // Construct extractor
    auto extractor = ao::extractor::GammatoneFilterbank<float>(
        /* num_samples */ num_samples,
        /* num_features */ 64,
        /* sample_rate */ file_info.samplerate,
        /* transform */ static_cast<float (*)(float)>(std::log10), // Not used
        /* on_channel */ 0, // Not using Matrices, this parameter is useless
        /* low_Hz */ 50,
        /* high_Hz */ 8000);
    std::cout << "Center frequencies: ";
    for (auto& filter : extractor.filters) {
        std::cout << filter.cf << " ";
    }
    std::cout << std::endl;

    std::vector<float> input(num_samples);
    for (int channel = 0; channel < file_info.channels; channel++) {
        for (std::size_t index = 0; index < num_samples; index++) {
            input[index] = signal[index * file_info.channels + channel];
        }
        // Execute extractor
        std::vector<float> output = extractor.compute(input);
        EXPECT_EQ(output, extractor.compute(input)); // Test second execution

        // Invalid input (too short)
        EXPECT_THROW(
            extractor.compute(
                std::vector<float>(input.begin(), input.end() - 1)),
            std::invalid_argument);
    }
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
    std::vector<float> signal(num_samples * file_info.channels);
    sf_readf_float(file, signal.data(), num_samples);

    // TODO parametrize with different extractors
    auto extractor = ao::extractor::GammatoneFilterbank<float>(
        /* num_samples */ num_samples,
        /* num_features */ 64,
        /* sample_rate */ file_info.samplerate,
        /* transform */ static_cast<float (*)(float)>(std::log10),
        /* on_channel */ 0, // Not using Matrices, this parameter is useless
        /* low_Hz */ 50,
        /* high_Hz */ 8000);

    std::vector<float> input(num_samples);
    for (int channel = 0; channel < file_info.channels; channel++) {
        for (std::size_t index = 0; index < num_samples; index++) {
            input[index] = signal[index * file_info.channels + channel];
        }
        // Compute only won't transform the output features
        std::vector<float> output = extractor.compute(input);
        std::transform(
            output.begin(), output.end(), output.begin(), extractor.transform);
        // operator()
        EXPECT_EQ(output, extractor(input));
    }
}

// TODO test that the extractor averages the channels depending on the
// on_channel value
TEST(TestExtractor, OnChannel) {
    // TODO parametrize input
    int num_samples = 250;
    SF_INFO file_info;
    SNDFILE* file = sf_open(AUDIO_PATH.string().c_str(), SFM_READ, &file_info);
    std::vector<float> signal(num_samples * file_info.channels);
    sf_readf_float(file, signal.data(), num_samples);

    // Build a matrix from the signal
    Eigen::MatrixXf signal_matrix(file_info.channels, num_samples);
    for (int channel = 0; channel < file_info.channels; channel++) {
        for (std::size_t index = 0; index < num_samples; index++) {
            signal_matrix(channel, index) =
                signal[index * file_info.channels + channel];
        }
    }

    // Test averaging extractor
    auto extract_average = ao::extractor::GammatoneFilterbank<float>(
        /* num_samples */ num_samples,
        /* num_features */ 64,
        /* sample_rate */ file_info.samplerate,
        /* transform */ static_cast<float (*)(float)>(std::log10),
        /* on_channel */ -1);

    std::vector<float> input(num_samples);
    // Average channels from signal
    for (std::size_t index = 0; index < num_samples; index++) {
        input[index] = 0;
        for (int channel = 0; channel < file_info.channels; channel++) {
            input[index] += signal[index * file_info.channels + channel];
        }
        input[index] /= file_info.channels;
    }
    EXPECT_EQ(extract_average(signal_matrix), extract_average(input));

    // Test row extractor
    for (int channel = 0; channel < file_info.channels; channel++) {
        auto extract_row = ao::extractor::GammatoneFilterbank<float>(
            /* num_samples */ num_samples,
            /* num_features */ 64,
            /* sample_rate */ file_info.samplerate,
            /* transform */ static_cast<float (*)(float)>(std::log10),
            /* on_channel */ channel);

        // Extract channel from signal
        for (std::size_t index = 0; index < num_samples; index++) {
            input[index] = signal[index * file_info.channels + channel];
        }

        EXPECT_EQ(extract_row(signal_matrix), extract_row(input));
    }
}