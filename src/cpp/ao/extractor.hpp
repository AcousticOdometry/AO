#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#pragma message("Constant M_PI was not defined, check your `cmath` imports")
#define M_PI 3.14159265358979323846
#endif

#include <fmt/core.h>

#include <functional>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <vector>
#include <array>

namespace ao {
namespace extractor {

/**
 * @brief Base abstract class for Extractors
 *
 * @tparam T Input signal type
 */
template <typename T> class Extractor {
    public:
    const size_t num_samples;  // Number of samples in the input signal
    const size_t num_features; // Number of features to extract
    const int sample_rate;     // Samples per second of the input signal [Hz]
    std::function<T(T)> transform;

    /**
     * @brief Construct a new Extractor object.
     *
     * @param num_samples Number of samples needed to extract a vector of
     * features.
     * @param num_features Number of features per vector.
     * @param sample_rate Samples per second of the input signal.
     * @param on_channel If the input samples are a 2D vector, on_channel
     * defines which
     */
    Extractor(
        const size_t& num_samples  = 1024,
        const size_t& num_features = 12,
        const int& sample_rate     = 44100,
        const std::function<T(T)>& transform = [](T x) { return x; }
        // TODO const size_t& on_channel = 0
        )
    : num_samples(num_samples),
      num_features(num_features),
      sample_rate(sample_rate),
      transform(transform) {
        static_assert(
            std::is_arithmetic<T>::value, "T must be an arithmetic type");
    }

    /**
     * @brief Destroy the Extractor object.
     *
     */
    virtual ~Extractor() {}

    /**
     * @brief Compute the feature extraction of an input signal into `output`
     * argument. This method must be implemented by each derived extractor.
     *
     * @param input Input signal, its size must be equal to `num_samples`.
     * @param features Output feature vector, its size must be equal to
     * `num_features`.
     */
    virtual void
    compute(const std::vector<T>& input, std::vector<T>& features) const = 0;

    /**
     * @brief Computes a vector of feature from the input signal. It ensures
     * that input and output have appropiate size.
     *
     * @param input A vector of samples that compose a signal.
     * @throw std::invalid_argument if the input signal size is different than
     * `num_samples`.
     * @return std::vector<T> Vector of features with size `num_features`.
     */
    virtual std::vector<T> compute(const std::vector<T>& input) const {
        if (input.size() != this->num_samples) {
            throw std::invalid_argument(fmt::format(
                "Input signal must be of length {}. Instead it is "
                "of length {}.",
                this->num_samples,
                input.size()));
        }
        std::vector<T> features(this->num_features);
        this->compute(input, features);
        return features;
    }

    /**
     * @brief Extract features from the input signal and transform them if
     * necessary. Does not check the input signal size.
     *
     * @param input A vector of samples that compose a signal.
     * @param features Output feature vector, its size must be equal to
     * `num_features`.
     */
    virtual void
    operator()(const std::vector<T>& input, std::vector<T>& features) const {
        this->compute(input, features);
        std::transform(
            features.begin(),
            features.end(),
            features.begin(),
            this->transform);
        return features;
    }

    /**
     * @brief Extract features from the input signal and transform them if
     * necessary. Validates the input signal size.
     *
     * @param input A vector of samples that compose a signal.
     * @throw std::invalid_argument if the input signal size is different than
     * `num_samples`.
     * @return std::vector<T> Vector of features with size `num_features`.
     */
    virtual std::vector<T> operator()(const std::vector<T>& input) const {
        auto features = this->compute(input);
        std::transform(
            features.begin(),
            features.end(),
            features.begin(),
            this->transform);
        return features;
    }
};

} // namespace extractor
} // namespace ao

// Include subclasses
// ! Don't know if it is the best way
#include "extractor/GammatoneFilterbank.hpp"