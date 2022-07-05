#pragma once

namespace ao {
namespace extractor {

/**
 * @brief Extractor based on gammatone filters.
 * https://en.wikipedia.org/wiki/Gammatone_filter
 *
 * @tparam T Input signal type
 */
template <typename T> class GammatoneFilterbank : public Extractor<T> {
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
         * @brief Construct a new Filter object. It is important to
         * highlight that filters do not have access to the parent
         * filterbank. They do not mind the number of samples or sample rate
         * once they are built.
         *
         * @param cf Center frequency
         * @param gain Gain
         * @param a Array of filter coefficients
         */
        Filter(
            const T cf,
            const T& coscf,
            const T& sincf,
            const T gain,
            const std::array<T, 5> a)
        : cf(cf), coscf(coscf), sincf(sincf), a(a), gain(gain) {}

        /**
         * @brief Wrapper around `Filter::compute`.
         *
         * @param input Vector of samples.
         * @return T Filter response.
         */
        T operator()(const std::vector<T>& input, const T& intdecay = 0) const {
            T feature;
            this->compute(input, feature, intdecay);
            return feature;
        }

        private:
        /**
         * @brief Compute the filter response to a vector of samples.
         *
         * @param input Vector of samples.
         * @param response Filter response.
         * TODO intdecay
         */
        void
        compute(const std::vector<T>& input, T& response, const T& intdecay = 0)
            const;
    };

    T intdecay;
    const std::vector<Filter> filters; // Vector of filters

    /**
     * @brief Construct a new Gammatone Filterbank object
     *
     * @param num_samples Number of samples needed to extract a vector of
     *      features.
     * @param num_features Number of filters to use in the filterbank.
     * @param sample_rate Samples per second of the input signal in Hz.
     * @param transform Function to transform the output features.
     * @param on_channel If the input samples are a 2D vector, on_channel
     *      defines which channel to use. If the value is negative, all
     *      channels will be averaged.
     * @param low_Hz Lowest filter center frequency in Hz.
     * @param high_Hz Highest filter center frequency in Hz.
     * @param temporal_integration Temporal integration in seconds.
     */
    GammatoneFilterbank(
        const size_t num_samples             = 1024,
        const size_t& num_features           = 64,
        const int& sample_rate               = 44100,
        const std::function<T(T)>& transform = [](T x) { return x; },
        const int& on_channel                = -1,
        const T& low_Hz                      = 100,
        const T& high_Hz                     = 8000,
        const T& temporal_integration        = 0)
    : Extractor<T>(
        num_samples, num_features, sample_rate, transform, on_channel),
      filters(make_filters(low_Hz, high_Hz, num_features, sample_rate)),
      intdecay(std::exp(-1 / (sample_rate * temporal_integration))) {}

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
     * @brief Builds a set of ao::extractor::GammatoneFilterbank::Filter
     * with center frequencies uniformly distributed between `low_Hz` and
     * `high_Hz` accross the Equivalent Rectangular Bandwith (ERB) scale.
     *
     * @param low_Hz Lowest filter center frequency in Hz.
     * @param high_Hz Highest filter center frequency in Hz.
     * @param num_filters Size of the output set.
     * @param sample_rate Samples per second of signals to be processed by
     * the filter.
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

#include "ao/extractor/GammatoneFilterbank.tpp"