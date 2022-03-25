#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#pragma message("Constant M_PI was not defined, check your `cmath` imports")
#define M_PI 3.14159265358979323846
#endif

#include "common.hpp"

#include <fmt/core.h>

#include <algorithm>
#include <array>
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
template <Arithmetic T> class Extractor {
    public:
    const size_t num_samples;  // Number of samples in the input signal
    const size_t num_features; // Number of features to extract
    const int sample_rate;     // Samples per second of the input signal [Hz]

    /**
     * @brief Construct a new Extractor object.
     *
     * @param num_samples Number of samples needed to extract a vector of
     * features.
     * @param num_features Number of features per vector.
     * @param sample_rate Samples per second of the input signal.
     */
    Extractor(
        const size_t& num_samples  = 1024,
        const size_t& num_features = 12,
        const int& sample_rate     = 44100)
    : num_samples(num_samples), num_features(num_features),
      sample_rate(sample_rate) {}

    /**
     * @brief Destroy the Extractor object.
     *
     */
    virtual ~Extractor() {}

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
                this->num_samples, input.size()));
        }
        std::vector<T> features(this->num_features);
        this->compute(input, features);
        return features;
    }

    /**
     * @brief Wrapper of std::vector<T> compute(const std::vector<T>& input)
     *
     * @param input A vector of samples that compose a signal.
     * @throw std::invalid_argument if the input signal size is different than
     * `num_samples`.
     * @return std::vector<T> Vector of features with size `num_features`.
     */
    virtual std::vector<T> operator()(const std::vector<T>& input) const {
        return this->compute(input);
    }

    protected:
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
};

/**
 * @brief Extractor based on gammatone filters.
 * https://en.wikipedia.org/wiki/Gammatone_filter
 *
 * @tparam T Input signal type
 */
template <Arithmetic T> class GammatoneFilterbank : public Extractor<T> {
    public:
    class Filter {
        public:
        const T cf;               // Center frequency
        const std::array<T, 5> a; // Filter coefficients
        const T gain;

        private:
        const T coscf; // Cosine of the center frequency
        const T sincf; // Sine of the center frequency

        public:
        /**
         * @brief Construct a new Filter object. It is important to highlight
         * that filters do not have access to the parent filterbank. They do
         * not mind the number of samples or sample rate once they are built.
         *
         * @param cf Center frequency
         * @param gain Gain
         * @param a Array of filter coefficients
         */
        Filter(const T cf, const T gain, const std::array<T, 5> a)
        : cf(cf), coscf(std::cos(cf)), sincf(std::sin(cf)), a(a), gain(gain) {}

        /**
         * @brief Wrapper around `Filter::compute`.
         *
         * @param input Vector of samples.
         * @return T Filter response.
         */
        T operator()(const std::vector<T>& input) const {
            T feature;
            this->compute(input, feature);
            return feature;
        }

        private:
        /**
         * @brief Compute the filter response to a vector of samples.
         *
         * @param input Vector of samples.
         * @param response Filter response.
         */
        void compute(const std::vector<T>& input, T& response) const;
    };

    const std::vector<Filter> filters; // Vector of filters

    /**
     * @brief Construct a new Gammatone Filterbank object
     *
     * @param num_samples Number of samples per input signal.
     * @param num_features Number of filters to use in the filterbank.
     * @param sample_rate Samples per second of the input signal in Hz.
     * @param low_Hz Lowest filter center frequency in Hz.
     * @param high_Hz Highest filter center frequency in Hz.
     */
    GammatoneFilterbank(
        const size_t num_samples   = 1024,
        const size_t& num_features = 64,
        const int& sample_rate     = 44100,
        const T& low_Hz            = 100,
        const T& high_Hz           = 8000)
    : Extractor<T>(num_samples, num_features, sample_rate),
      filters(make_filters(low_Hz, high_Hz, num_features, sample_rate)) {}

    using Extractor<T>::compute; // Inherit `compute` from `Extractor`

    protected:
    /**
     * @brief Compute the `filters` response to an input signal.
     *
     * @param input Input signal with size `num_samples`.
     * @param features Output feature vector of size `num_features`.
     */
    void compute(
        const std::vector<T>& input, std::vector<T>& features) const override;

    /**
     * @brief Converts a frequency in Hz to its Equivalent Rectangular
     * Bandwidth.
     *
     * @param hz Frequency in Hz.
     * @return T Equivalent Rectangular Bandwidth.
     */
    static T Hz_to_ERB(const T hz);

    /**
     * @brief Converts an Equivalent Rectangular Bandwidth to its frequency in
     * Hz.
     *
     * @param erb Equivalent Rectangular Bandwidth.
     * @return T Frequency in Hz.
     */
    static T ERB_to_Hz(const T erb);

    /**
     * @brief Builds a set of ao::extractor::GammatoneFilterbank::Filter with
     * center frequencies uniformly distributed between `low_Hz` and `high_Hz`
     * accross the Equivalent Rectangular Bandwith (ERB) scale.
     *
     * @param low_Hz Lowest filter center frequency in Hz.
     * @param high_Hz Highest filter center frequency in Hz.
     * @param num_filters Size of the output set.
     * @param sample_rate Samples per second of signals to be processed by the
     * filter.
     * @param bandwith_correction ERB bandwidth correction for the 4th order
     * filter. Defaults to 1.019.
     * @return std::vector<Filter> Set of filters with center frequencies
     * uniformly distributed between `low_Hz` and `high_Hz` accross the ERB
     * scale.
     */
    static std::vector<Filter> make_filters(
        const T& low_Hz,
        const T& high_Hz,
        const size_t& num_filters,
        const int& sample_rate,
        const T bandwith_correction = 1.019);
};

} // namespace extractor
} // namespace ao

#include "extractor.tpp"