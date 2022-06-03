#include "ao.hpp"

#include <gtest/gtest.h>

#include <iostream>

TEST(TestInference, LibTorch) {
    auto model_path = std::filesystem::path(
        "/home/esdandreu/AO/models/"
        "torch-script;name_numpy-arrays;date_2022-05-23;time_13-39-14.pt");
    std::vector<ao::extractor::Extractor<float>*> extractors;
    auto extractor = ao::extractor::GammatoneFilterbank<float>(
        /* num_samples */ 256,
        /* num_features */ 256,
        /* sample_rate */ 44100,
        /* low_Hz */ 50,
        /* high_Hz */ 8000);
    extractors.push_back(&extractor);
    auto acoustic_odometry = ao::AO<float>(model_path, extractors);
    std::cout << "ok\n";
}