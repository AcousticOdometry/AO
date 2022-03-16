#ifndef AO_EXTRACTOR_HPP
#define AO_EXTRACTOR_HPP

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#pragma message("Constant M_PI was not defined, check your `cmath` imports")
#define M_PI 3.14159265358979323846
#endif

#include <vector>

namespace ao {
namespace extractor {

// Base class for all extractors
template <typename T> class Extractor {
    public:
    const size_t num_samples;
    const size_t num_features;
    const int sample_rate;

    Extractor(const size_t num_samples = 1024,
        const size_t num_features      = 12,
        const int sample_rate          = 44100)
    : num_samples(num_samples), num_features(num_features),
      sample_rate(sample_rate) {
    }

    virtual void
    compute(const std::vector<T>& input, std::vector<T>& output) = 0;
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
template <typename T> class GammatoneFilterbank : public Extractor<T> {
    const double tpt = (M_PI + M_PI) / sample_rate;
    // TODO Compute characteristic frequencies equally spaced on ERB scale
    // using the canonical procedure.
    const std::vector<double> filters;

    T Hz_to_ERB(const T hz) const {
        return (hz > 0) ? 21.4 * log10(4.37 * hz + 1) : -1000;
    }

    T ERB_to_Hz(const T erb) const {
        return (erb > 0) ? (10.0 / 21.4) * (erb - 1) / (4.37 - 1) : 0;
    }

    public:
    void compute(const std::vector<T>& input, std::vector<T>& output) {
        for (auto& cf : filters) {
            // tptbw
            // a
            // gain

            // TODO initialize filter results
            // const double coscf = std::cos(tpt * cf);
            // const double sincf = std::sin(tpt * cf );
            // double qcos = 1, qsin = 0;   /* t=0 & q = exp(-i*tpt*t*cf)*/
            for (const T& sample : input) {
                // TODO Filter part 1: compute p0r and p0i
                // TODO Filter part 2: compute u0r and u0i
                // TODO Update filter results

                // Basilar membrane response
                // 1. Shift up in frequency first:
                //   (u0r+i*u0i) * exp(i*tpt*cf*t) = (u0r+i*u0i) *
                //     (qcos + i*(-qsin))
                // 2. Take the real part only:
                //   bm = real(exp(j*wcf*kT).*u) * gain.
                // TODO (*output)(i, j) = (u0r * qcos + u0i * qsin) * gain;

                // The basic idea of saving computational load:
                //   cos(a+b) = cos(a)*cos(b) - sin(a)*sin(b)
                //   sin(a+b) = sin(a)*cos(b) + cos(a)*sin(b)
                //   qcos = cos(tpt*cf*t) = cos(tpt*cf + tpt*cf*(t-1))
                //   qsin = -sin(tpt*cf*t) = -sin(tpt*cf + tpt*cf*(t-1))
                // const double old_qcos = qcos;
                // qcos = coscf * old_qcos  + sincf * qsin;
                // qsin = coscf * qsin - sincf * old_qcos;
            }
        }
    };
};

} // namespace extractor
} // namespace ao

#endif // AO_EXTRACTOR_HPP