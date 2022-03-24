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

// Base class for all extractors
template <Arithmetic T> class Extractor {
    public:
    const size_t num_samples;  // Number of samples in the input signal
    const size_t num_features; // Number of features to extract
    const int sample_rate;     // Samples per second of the input signal [Hz]

    Extractor(
        const size_t& num_samples  = 1024,
        const size_t& num_features = 12,
        const int& sample_rate     = 44100)
    : num_samples(num_samples), num_features(num_features),
      sample_rate(sample_rate) {}

    virtual ~Extractor() {}

    virtual void
    compute(const std::vector<T>& input, std::vector<T>& output) = 0;

    virtual std::vector<T> compute(const std::vector<T>& input) {
        std::vector<T> output(this->num_features);
        compute(input, output);
        return output;
    }

    virtual std::vector<T> operator()(const std::vector<T>& input) {
        return compute(input);
    }
};


template <Arithmetic T> class GammatoneFilterbank : public Extractor<T> {
    public:
    struct Filter {
        // TODO add functionality here

        const T cf;               // Center frequency
        const T coscf;            // Cosine of the center frequency
        const T sincf;            // Sine of the center frequency
        const std::array<T, 5> a; // Filter coefficients
        const T gain;

        Filter(const T cf, const T gain, const std::array<T, 5> a)
        : cf(cf), a(a), gain(gain), coscf(std::cos(cf)), sincf(std::sin(cf)) {}
    };

    const std::vector<Filter> filters;

    GammatoneFilterbank(
        const size_t num_samples   = 1024,
        const size_t& num_features = 64,
        const int& sample_rate     = 44100,
        const T& low_Hz            = 100,
        const T& high_Hz           = 8000)
    : Extractor<T>(num_samples, num_features, sample_rate),
      filters(make_filters(low_Hz, high_Hz, num_features, sample_rate)) {}

    using Extractor<T>::compute;

    void compute(const std::vector<T>& input, std::vector<T>& output);

    private:
    // TODO help on this
    static T Hz_to_ERB(const T hz);

    // TODO help on this
    static T ERB_to_Hz(const T erb);

    static std::vector<Filter> make_filters(
        const T& low_Hz,
        const T& high_Hz,
        const size_t& num_filters,
        const int& sample_rate,
        const T bandwith_correction = 1.019 // ERB bandwidth correction for the
                                            // 4th order filter
    );
};

} // namespace extractor
} // namespace ao

#include "extractor.tpp"