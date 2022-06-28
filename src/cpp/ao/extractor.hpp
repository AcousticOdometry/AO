#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#pragma message("Constant M_PI was not defined, check your `cmath` imports")
#define M_PI 3.14159265358979323846
#endif

#include <eigen3/Eigen/Core>
#include <fmt/core.h>

#include <algorithm>
#include <array>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

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
    const int on_channel; // If a multi-channel signal is provided, this is the
                          // channel to use for the extraction
    const std::function<T(T)> transform;

    using RefMatrix =
        Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;

    /**
     * @brief Construct a new Extractor object.
     *
     * @param num_samples Number of samples needed to extract a vector of
     *      features.
     * @param num_features Number of features per vector.
     * @param sample_rate Samples per second of the input signal.
     * @param transform Function to transform the output features.
     * @param on_channel If the input samples are a 2D vector, on_channel
     *      defines which channel to use. If the value is negative, all
     *      channels will be averaged.
     */
    Extractor(
        const size_t& num_samples            = 1024,
        const size_t& num_features           = 12,
        const int& sample_rate               = 44100,
        const std::function<T(T)>& transform = [](T x) { return x; },
        const int& on_channel                = -1)
    : num_samples(num_samples),
      num_features(num_features),
      sample_rate(sample_rate),
      transform(transform),
      on_channel(on_channel),
      matrix_to_vector(matrix_to_vector_factory(on_channel)) {
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

    /**
     * @brief Extract features from a multi-channel input signal provided as a
     * a matrix. It makes use of the `matrix_to_vector` function initialized
     * with the `on_channel` parameter.
     * 
     * @param input Multi-channel input signal.
     * @return std::vector<T> Vector of features with size `num_features`.
     */
    virtual std::vector<T> operator()(RefMatrix input) const {
        return this->operator()(this->matrix_to_vector(input));
    }

    // TODO
    // virtual std::vector<T>
    // operator()(RefMatrix input, std::vector<T>& features) const {
    //     return this->operator()(this->matrix_to_vector(input), features);
    // }

    private:
    const std::function<std::vector<T>(RefMatrix)> matrix_to_vector;

    /**
     * @brief Factory function that returns a function that converts an Eigen
     * matrix into a std::vector. Used when multi-channel signals are provided
     * as a matrix, like in the case of providing a numpy array as input in the
     * Python extension.
     *
     * @param on_channel Which channel to use. If negative, all channels will
     *      be averaged.
     * @return std::function<std::vector<T>(RefMatrix)> Function that converts
     *      a matrix into a std::vector
     */
    static std::function<std::vector<T>(RefMatrix)>
    matrix_to_vector_factory(const int& on_channel) {
        // Return a channel averaging function
        if (on_channel < 0) {
            return [](RefMatrix m) {
                // ? Can we use num_samples instead of Dynamic ?
                Eigen::Vector<T, Eigen::Dynamic> temp = m.colwise().mean();
                return std::vector<T>(temp.data(), temp.data() + temp.size());
            };
        }
        // Return a row extractor
        return [on_channel](RefMatrix m) {
            Eigen::Vector<T, Eigen::Dynamic> temp = m.row(on_channel);
            return std::vector<T>(temp.data(), temp.data() + temp.size());
        };
    }
};

} // namespace extractor
} // namespace ao

// Include subclasses
// ! Don't know if it is the best way
#include "extractor/GammatoneFilterbank.hpp"