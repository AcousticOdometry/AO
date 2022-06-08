#pragma once

// The python API will use pytorch directly instead of the C++ frontend
#ifdef WITH_LIBTORCH

#include "ao/extractor.hpp"

#include <fmt/core.h>
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
    const torch::Tensor& features;   // Read only model input tensor
    const torch::Tensor& prediction; // Read only model output tensor
    const torch::Device& device;

    protected:
    torch::jit::script::Module model;

    private:
    std::vector<T> _temp_features; // Temporary vector of features used to
                                   // update _features
    torch::Tensor _features;       // Model input tensor
    torch::Tensor _prediction;     // Model output tensor

    public:
    /**
     * @brief Construct a new AO object
     *
     * @param model_path Path to the model file
     * @param extractors Vector of extractor pointers that will be called with
     * the input signal to extract features as channels for the model input.
     * @param num_frames Number of frames to keep rolling in the model input.
     * @param device Device to run the model on. Defaults to cpu.
     */
    AO(const std::filesystem::path& model_path,
       const std::vector<extractor::Extractor<T>*>& extractors,
       const size_t& num_frames,
       const torch::Device& device = torch::Device("cpu"))
    : extractors(extractors),
      // TODO hardcoded prediction size
      _prediction(torch::zeros({1})),
      prediction(_prediction),
      device(device),
      features(_features),
      num_frames(num_frames),
      // TODO this crashes if extractors is empty
      num_features(extractors[0]->num_features),
      num_samples(extractors[0]->num_samples) {
        // Assert all extractors have the same number of frames and samples
        for (auto& extractor : extractors) {
            if (extractor->num_samples != num_samples) {
                throw std::runtime_error(fmt::format(
                    "Provided extractors with different number of input "
                    "samples: {} != {}",
                    extractor->num_samples,
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
        // Load model
        // TODO test load on GPU
        model = torch::jit::load(model_path.string(), this->device);
        // Allocate memory for the model input tensor
        // ? tensor with type T ?
        _features = torch::empty({
            1,
            static_cast<long int>(extractors.size()),
            static_cast<long int>(num_features),
            static_cast<long int>(num_frames),
        });
        // Allocate memory for a temporary vector of features
        this->_temp_features.resize(num_features);
    }

    // ! What about several channels ? vector of vectors ?
    /**
     * @brief Update the model input features by extracting new features from
     * unique channel of input signal samples.
     *
     * @param samples A vector of samples that compose the input signal.
     * @return const torch::Tensor& The updated model input tensor.
     */
    const torch::Tensor& update(const std::vector<T>& samples) {
        // Compute features and insert them on the model input tensor
        for (int i = 0; i < this->extractors.size(); i++) {
            this->extractors[i]->compute(samples, this->_temp_features);
            // Replace the first column with new features
            this->_features.index_put_(
                {0, i, torch::indexing::Slice(), 0},
                torch::from_blob(
                    this->_temp_features.data(),
                    {static_cast<long int>(this->num_features)}));
            // Roll the tensor in order to have the first column as last
            this->_features = this->_features.roll(-1, 3);
        }
        return this->features;
    }

    /**
     * @brief Uses the model to predict with the stored features.
     *
     * @return const torch::Tensor& Prediction.
     */
    const torch::Tensor& predict() {
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(this->_features);
        this->_prediction = this->model.forward(inputs).toTensor();
        return this->prediction;
    }

    /**
     * @brief Updates the model input features and uses them to predict with
     * the model.
     *
     * @param samples Vector of samples that compose the input signal.
     * @return const torch::Tensor& Prediction.
     */
    const torch::Tensor& predict(const std::vector<T>& samples) {
        this->update(samples);
        return this->predict();
    }

    /**
     * @brief Wrapper for predict.
     *
     * @param samples Vector of samples that compose the input signal.
     * @return const torch::Tensor& Prediction.
     */
    const torch::Tensor& operator()(const std::vector<T>& samples) {
        return this->predict(samples);
    }
};

} // namespace ao

#endif