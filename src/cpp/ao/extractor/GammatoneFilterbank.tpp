#include "ao/extractor/ERB.hpp"

#include <array>
#include <vector>

namespace ao {
namespace extractor {

// This gammatone filter is based on the implementation by Ning Ma from
// University of Sheffield who, in turn, based his implementation on an
// original algorithm from Martin Cooke's Ph.D thesis (Cooke, 1993) using
// the base-band impulse invariant transformation. This implementation is
// highly efficient in that a mathematical rearrangement is used to
// significantly reduce the cost of computing complex exponential. For
// more detail on this implementation see
//   http://www.dcs.shef.ac.uk/~ning/resources/gammatone/
//
// Note: Martin Cooke's PhD has been reprinted as M. Cooke (1993): "Modelling
// Auditory Processing and Organisation", Cambridge University Press, Series
// "Distinguished Dissertations in Computer Science", August.
template <typename T>
void GammatoneFilterbank<T>::Filter::compute(
    const std::vector<T>& input, T& response, const T& intdecay) const {
    // Envelope
    std::vector<T> env(input.size(), 0);
    T last_env = 0;
    // TODO What about temporal integration

    // Initialize filter results to zero.
    T p1r = 0.0, p2r = 0.0, p3r = 0.0, p4r = 0.0, p1i = 0.0, p2i = 0.0,
      p3i = 0.0, p4i = 0.0;
    T qcos = 1, qsin = 0; /* t=0 & q = exp(-i*tpt*t*cf)*/
    // q = e^{-i*tpt*t*cf} = cos(tpt*t*cf) + i*sin(tpt*t*cf)
    // TODO q is complex
    for (int i = 0; i < input.size(); i++) {
        // Filter part 1: compute p0r and p0i
        T p0r = qcos * input[i] + this->a[0] * p1r + this->a[1] * p2r
              + this->a[2] * p3r + this->a[3] * p4r;
        if (std::fabs(p0r) < std::numeric_limits<float>::min()) {
            p0r = 0.0;
        }
        T p0i = qsin * input[i] + this->a[0] * p1i + this->a[1] * p2i
              + this->a[2] * p3i + this->a[3] * p4i;
        if (std::fabs(p0i) < std::numeric_limits<float>::min()) {
            p0i = 0.0;
        }

        // Filter part 2: compute u0r and u0i
        const T u0r = p0r + this->a[0] * p1r + this->a[4] * p2r;
        const T u0i = p0i + this->a[0] * p1i + this->a[4] * p2i;

        // Update filter results
        // TODO in array
        p4r = p3r;
        p3r = p2r;
        p2r = p1r;
        p1r = p0r;
        p4i = p3i;
        p3i = p2i;
        p2i = p1i;
        p1i = p0i;

        // Smoothed envelope by temporal integration
        last_env = env[i] =
            sqrt(u0r * u0r + u0i * u0i) * this->gain + intdecay * last_env;

        // TODO better explanation
        // The basic idea of saving computational load:
        //   cos(a+b) = cos(a)*cos(b) - sin(a)*sin(b)
        //   sin(a+b) = sin(a)*cos(b) + cos(a)*sin(b)
        //   qcos = cos(tpt*cf*t) = cos(tpt*cf + tpt*cf*(t-1))
        //   qsin = -sin(tpt*cf*t) = -sin(tpt*cf + tpt*cf*(t-1))
        const T old_qcos = qcos;
        qcos             = this->coscf * old_qcos + this->sincf * qsin;
        qsin             = this->coscf * qsin - this->sincf * old_qcos;
    }
    // We take the mean of the smoothed envelope as the energy response rather
    // than simply sampling it.
    response = (1 - intdecay) // Temporal integration gain
             * std::accumulate(env.begin(), env.end(), 0.0) / env.size();
}

template <typename T>
void GammatoneFilterbank<T>::compute(
    const std::vector<T>& input, std::vector<T>& features) const {
    for (int j = 0; j < this->num_features; j++) {
        features[j] = this->filters[j](input, this->intdecay);
    }
}

template <typename T>
std::vector<typename GammatoneFilterbank<T>::Filter>
GammatoneFilterbank<T>::make_filters(
    const T& low_Hz,
    const T& high_Hz,
    const size_t& num_filters,
    const int& sample_rate,
    const T bandwith_correction) {
    // Initialize output
    std::vector<GammatoneFilterbank<T>::Filter> filters;
    filters.reserve(num_filters);
    // Compute characteristic frequencies equally spaced on ERB scale
    // using the canonical procedure.
    const T low_erb  = Hz_to_ERBRate(low_Hz);
    const T high_erb = Hz_to_ERBRate(high_Hz);
    const T step_erb = (high_erb - low_erb) / (num_filters - 1);
    if (step_erb <= 0) {
        // TODO elaborate error handling
        throw std::invalid_argument("Invalid frequency range");
    }
    // TODO better name or help
    /*=========================================================================
     * complex z = re + i*im
     * exp(z) = exp(re + i*im) = exp(re) * exp(i*im)
     *        = exp(re) * ( cos(im) + i*sin(im) ) // <- Euler's formula
     *        = exp(re) * cos(re) + i * exp(re) * sin(im)
     * z = -i * tpti * cf
     * exp(z) = cos(tpti*cf) - i * sin(tpti*cf)
     *=========================================================================
     */
    const T tpt = static_cast<T>((M_PI + M_PI) / sample_rate); // 2*pi/fs
    for (int i = 0; i < num_filters; i++) {
        const T cf = ERBRate_to_Hz(low_erb + i * step_erb);
        // TODO help on this
        // ? why in ERB scale ?
        const T tptbw = tpt * ERB(cf) * bandwith_correction;
        // TODO Based on integral of impulse response.
        const T gain             = (tptbw * tptbw * tptbw * tptbw) / 3.0;
        const T _a               = std::exp(-tptbw); // ? e^{-2*pi*b*t}
        const std::array<T, 5> a = {
            4 * _a,
            -6 * _a * _a,
            4 * _a * _a * _a,
            -_a * _a * _a * _a,
            // Why not std::pow(_a, 4) ? Well... it is slower
            // https://baptiste-wicht.com/posts/2017/09/cpp11-performance-tip-when-to-use-std-pow.html
            _a * _a};
        filters.push_back(
            Filter(cf, std::cos(tpt * cf), std::sin(tpt * cf), gain, a));
    }
    return filters;
}

} // namespace extractor
} // namespace ao