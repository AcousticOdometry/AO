#pragma once

#include "extractor.hpp"

#include <torch/script.h>

#include <filesystem>

namespace ao {

/**
 * @brief Acoustic Odometry class
 *
 * @tparam T Input signal type
 */
template <typename T> class AO {
    public:
    // ! Don't know if it will be destroyed or not
    const std::vector<extractor::Extractor<T>*> extractors;
    const size_t num_frames;
    const size_t num_features;
    const size_t num_samples;
    const torch::Tensor& prediction;

    protected:
    torch::jit::script::Module model;

    private:
    torch::Tensor _input;
    torch::Tensor _prediction;
    size_t frame_n = 0;

    public:
    /**
     * @brief Construct a new AO object
     *
     */
    AO(const std::filesystem::path& model_path,
       const std::vector<extractor::Extractor<T>*>& extractors,
       // TODO remove default values?
       const size_t& num_frames         = 120,
       const std::string& device_string = "cpu")
    : extractors(extractors),
      _prediction(torch::zeros({6})),
      prediction(_prediction),
      num_frames(num_frames),
      num_features(extractors[0]->num_features),
      num_samples(extractors[0]->num_samples) {
        // Assert all extractors have the same number of frames and samples
        for (auto& extractor : extractors) {
            if (extractor->num_samples != num_samples) {
                throw std::runtime_error(fmt::format(
                    "Provided extractors with different number of input "
                    "samples: {} != {}",
                    extractor->num_features,
                    num_samples));
            }
            if (extractor->num_features != num_features) {
                throw std::runtime_error(fmt::format(
                    "Provided extractors with different number of output "
                    "features: {} != {}",
                    extractor->num_features,
                    num_samples));
            }
        }
        // TODO test load on GPU
        model = torch::jit::load(model_path, torch::Device(device_string));
        // ? tensor with type T ?
        _input = torch::empty(
            {1,
             static_cast<long int>(extractors.size()),
             static_cast<long int>(num_features),
             static_cast<long int>(num_frames)});
        std::cout << _input << std::endl;
    }

    // What about several channels ? vector of vectors ?
    const torch::Tensor& predict(const std::vector<T>& input) {
        _input.index({
            0,
            0,
        });
        // TODO build input tensor
        // TODO predict
        return prediction;
    }
};

} // namespace ao