#include "ao.hpp"

#include <gtest/gtest.h>
#include <sndfile.h>

#include <iostream>

// TODO Audio fixture
// class TestAudioFiles : public testing::TestWithParam<int> {};
std::filesystem::path AUDIO_PATH2 =
    std::filesystem::path(__FILE__).parent_path() / "data" / "audio0.wav";

// TODO parametrize num_samples
TEST(TestInference, LibTorch) {
    auto model_path = std::filesystem::path(
        "/home/esdandreu/AO/models/"
        "torch-script;name_numpy-arrays;date_2022-05-23;time_13-39-14.pt");
    int num_samples = 250;
    std::vector<ao::extractor::Extractor<float>*> extractors;
    // ! Hardcoded number of features
    auto extractor = ao::extractor::GammatoneFilterbank<float>(
        /* num_samples */ num_samples,
        /* num_features */ 256,
        /* sample_rate */ 44100,
        /* low_Hz */ 50,
        /* high_Hz */ 8000);
    extractors.push_back(&extractor);
    // ! Hardcoded number of frames
    auto model = ao::AO<float>(model_path, extractors, 120);
    // Load file
    SF_INFO file_info;
    SNDFILE* file = sf_open(AUDIO_PATH2.string().c_str(), SFM_READ, &file_info);
    std::vector<float> input(num_samples);
    sf_count_t sample_n = 0;
    int count           = 0;
    while (sample_n < file_info.frames) {
        sf_read_float(file, input.data(), input.size());
        auto pre_features = model.features.clone();
        model.predict(input);
        // Assert that the features have shifted by one with a new introduction
        // at the end
        ASSERT_TRUE(pre_features
                        .index({
                            0,
                            torch::indexing::Slice(),
                            torch::indexing::Slice(),
                            torch::indexing::Slice(1, torch::indexing::None),
                        })
                        .equal(model.features.index({
                            0,
                            torch::indexing::Slice(),
                            torch::indexing::Slice(),
                            torch::indexing::Slice(0, -1),
                        })));
        std::cout << model.prediction << std::endl;
        sample_n += input.size();
        if (count++ > 3) {
            break;
        }
    }
}