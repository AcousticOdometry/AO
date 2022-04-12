#pragma once

#include "extractor.hpp"

#include <array>
#include <vector>

namespace ao {
namespace extractor {

template <typename T> class GammatoneFilterbank : public Filterbank<T> {
    public:
    class GammatoneFilter : public Filter {
        public:
        const std::array<T, 5> a; // Filter coefficients
        const T gain;

        private:
        const T coscf; // Cosine of the center frequency
        const T sincf; // Sine of the center frequency

        public:
        /**
         * @brief Construct a new GammatoneFilter object.
         *
         * @param cf Center frequency
         * TODO
         * @param gain Gain
         * @param a Array of filter coefficients
         */
        GammatoneFilter(
            const T cf,
            const T& coscf,
            const T& sincf,
            const T gain,
            const std::array<T, 5> a
            // TODO intdecay and intgain
            )
        : Filter(cf), gain(gain), coscf(coscf), sincf(sincf), a(a) {
            // TODO Assign compute depending on `intdecay` and `intgain`
        }

        // TODO Gammatone specific help
        /**
         * @brief Compute the filter response to a vector of samples.
         *
         * @param input Vector of samples.
         * @param response Filter response.
         */
        void compute(const std::vector<T>& input, T& response) const override;

        public:
        // TODO Difference with `compute`
        void compute_with_temporal_integration(
            const std::vector<T>& input,
            T& response,
            const T& intdecay,
            const T& intgain) const;
    };

    public:
    const T bandwidth_correction = 1.019;
    const T intdecay;
    const T intgain;

    public:
    /**
     * @brief Construct a new Gammatone Filterbank object
     *
     * @param num_samples Number of samples per input signal.
     * @param num_features Number of filters to use in the filterbank.
     * @param sample_rate Samples per second of the input signal in Hz.
     * @param low_Hz Lowest filter center frequency in Hz.
     * @param high_Hz Highest filter center frequency in Hz.
     * TODO expand
     * @param temporal_integration Temporal integration in seconds.
     */
    GammatoneFilterbank(
        const size_t num_samples      = 1024,
        const size_t& num_features    = 64,
        const int& sample_rate        = 44100,
        const T& low_Hz               = 100,
        const T& high_Hz              = 8000,
        const T& temporal_integration = 0)
    : intdecay(std::exp(-1 / (sample_rate * temporal_integration))),
      intgain(1 - intdecay),
      Filterbank<T>(num_samples, num_features, sample_rate) {}

    using Filterbank<T>::compute; // Inherit `compute` from `Filterbank`

    protected:
    std::vector<Filter&> make_filters(
        const T& low_Hz,
        const T& high_Hz,
        const size_t& num_features,
        const int& sample_rate) const override;
};

} // namespace extractor
} // namespace ao

#include "extractor/GammatoneFilterbank.tpp"