#ifndef AO_EXTRACTOR_HPP
#define AO_EXTRACTOR_HPP

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#pragma message("Constant M_PI was not defined, check your `cmath` imports")
#define M_PI 3.14159265358979323846
#endif

#include <fmt/core.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

// Remove
#include <iostream>

namespace ao {
namespace extractor {

// Base class for all extractors
template <typename T,
    // Restrict T to numeric types (float, double, int, etc.)
    std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>
class Extractor {
    public:
    const size_t num_samples;  // Number of samples in the input signal
    const size_t num_features; // Number of features to extract
    const int sample_rate;     // Samples per second of the input signal [Hz]

    Extractor(const size_t& num_samples = 1024,
        const size_t& num_features      = 12,
        const int& sample_rate          = 44100)
    : num_samples(num_samples), num_features(num_features),
      sample_rate(sample_rate) {
    }

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

// This gammatone filter is based on the implementation by Ning Ma from
// University of Sheffield who, in turn, based his implementation on an
// original algorithm from Martin Cooke's Ph.D thesis (Cooke, 1993) using
// the base-band impulse invariant transformation. This implementation is
// highly efficient in that a mathematical rearrangement is used to
// significantly reduce the cost of computing complex exponentials. For
// more detail on this implementation see
//   http://www.dcs.shef.ac.uk/~ning/resources/gammatone/
//
// Note: Martin Cooke's PhD has been reprinted as M. Cooke (1993): "Modelling
// Auditory Processing and Organisation", Cambridge University Press, Series
// "Distinguished Dissertations in Computer Science", August.
template <typename T,
    // Restrict T to numeric types (float, double, int, etc.)
    std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>
class GammatoneFilterbank : public Extractor<T> {
    struct Filter {
        const T cf;               // Center frequency
        const T coscf;            // Cosine of the center frequency
        const T sincf;            // Sine of the center frequency
        const std::array<T, 5> a; // Filter coefficients
        const T gain;

        Filter(const T cf, const T gain, const std::array<T, 5> a)
        : cf(cf), a(a), gain(gain), coscf(std::cos(cf)), sincf(std::sin(cf)) {
        }
    };

    public:
    const std::vector<Filter> filters;

    GammatoneFilterbank(const size_t num_samples = 1024,
        const size_t& num_features               = 64,
        const int& sample_rate                   = 44100,
        const T& low_Hz                          = 100,
        const T& high_Hz                         = 8000)
    : Extractor(num_samples, num_features, sample_rate),
      filters(make_filters(low_Hz, high_Hz, num_features, sample_rate)) {
        std::cout << "Constructing GammatoneFilterbank" << std::endl;
    }

    using Extractor::compute;
    void compute(const std::vector<T>& input, std::vector<T>& output) override {
        if (input.size() != this->num_samples) {
            throw std::runtime_error(
                fmt::format("Input signal must be of length {}. Instead it is "
                            "of length {}.",
                    this->num_samples, input.size()));
        }
        for (int j = 0; j < this->num_features; j++) {
            Filter filter = filters[j];

            // Initialize filter results to zero.
            T p1r = 0.0, p2r = 0.0, p3r = 0.0, p4r = 0.0, p1i = 0.0, p2i = 0.0,
              p3i = 0.0, p4i = 0.0;
            T qcos = 1, qsin = 0; /* t=0 & q = exp(-i*tpt*t*cf)*/

            std::vector<T> env(this->num_samples, 0);
            // T senv1 = 0;
            for (int t = 0; t < this->num_samples; t++) {
                // Filter part 1: compute p0r and p0i
                T p0r = qcos * input[t] + filter.a[0] * p1r + filter.a[1] * p2r
                    + filter.a[2] * p3r + filter.a[3] * p4r;
                if (std::fabs(p0r) < std::numeric_limits<float>::min()) {
                    p0r = 0.0;
                }
                T p0i = qsin * input[t] + filter.a[0] * p1i + filter.a[1] * p2i
                    + filter.a[2] * p3i + filter.a[3] * p4i;
                if (std::fabs(p0i) < std::numeric_limits<float>::min()) {
                    p0i = 0.0;
                }

                // Filter part 2: compute u0r and u0i
                const T u0r = p0r + filter.a[0] * p1r + filter.a[4] * p2r;
                const T u0i = p0i + filter.a[0] * p1i + filter.a[4] * p2i;

                // Update filter results
                p4r = p3r;
                p3r = p2r;
                p2r = p1r;
                p1r = p0r;
                p4i = p3i;
                p3i = p2i;
                p2i = p1i;
                p1i = p0i;

                // Envelope
                env[t] = sqrt(u0r * u0r + u0i * u0i) * filter.gain;
                // Smoothed env by temporal integration
                // senv1 = senv[i] = sqrt(u0r * u0r + u0i * u0i) * filter.gain
                //     + intdecay * senv1;

                // The basic idea of saving computational load:
                //   cos(a+b) = cos(a)*cos(b) - sin(a)*sin(b)
                //   sin(a+b) = sin(a)*cos(b) + cos(a)*sin(b)
                //   qcos = cos(tpt*cf*t) = cos(tpt*cf + tpt*cf*(t-1))
                //   qsin = -sin(tpt*cf*t) = -sin(tpt*cf + tpt*cf*(t-1))
                // TODO remove temporal value
                const T old_qcos = qcos;
                qcos = filter.coscf * old_qcos + filter.sincf * qsin;
                qsin = filter.coscf * qsin - filter.sincf * old_qcos;
            }
            output[j] =
                std::accumulate(env.begin(), env.end(), 0.0) / env.size();
        }
    };

    private:
    // TODO help on this
    // http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/doc/voicebox/frq2erb.html
    static T Hz_to_ERB(const T hz) {
        return (hz > 0) ?
            11.17268 * std::log(1 + 46.06538 * hz / (hz + 14678.49)) :
            -1;
    }

    // TODO help on this
    // http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/doc/voicebox/erb2frq.html
    static T ERB_to_Hz(const T erb) {
        return (erb > 0) ?
            676170.4 / std::max<T>(47.06538 - std::exp(0.0895 * erb), 0)
                - 14678.5 :
            0;
    }


    static std::vector<Filter> make_filters(const T& low_Hz,
        const T& high_Hz,
        const size_t& num_filters,
        const int& sample_rate,
        const T bandwith_correction = 1.019 // ERB bandwidth correction for the
                                            // 4th order filter
    ) {
        std::vector<Filter> filters;
        filters.reserve(num_filters);
        // Compute characteristic frequencies equally spaced on ERB scale
        // using the canonical procedure.
        const T low_erb  = Hz_to_ERB(low_Hz);
        const T high_erb = Hz_to_ERB(high_Hz);
        const T step_erb = (high_erb - low_erb) / (num_filters - 1);
        if (step_erb <= 0) {
            // TODO elaborate error handling
            throw std::invalid_argument("Invalid frequency range");
        }
        // TODO help on this
        const T tpt = static_cast<T>((M_PI + M_PI) / sample_rate);
        for (int i = 0; i < num_filters; i++) {
            const T cf = ERB_to_Hz(low_erb + i * step_erb);
            // TODO help on this
            const T tptbw = tpt * Hz_to_ERB(cf) * bandwith_correction;
            // TODO Based on integral of impulse response.
            const T gain             = (tptbw * tptbw * tptbw * tptbw) / 3.0;
            const T _a               = std::exp(-tptbw);
            const std::array<T, 5> a = { 4 * _a, -6 * _a * _a, 4 * _a * _a * _a,
                -_a * _a * _a * _a, _a * _a };
            filters.push_back(Filter(cf, gain, a));
        }
        return filters;
    }
};

} // namespace extractor
} // namespace ao

#endif // AO_EXTRACTOR_HPP