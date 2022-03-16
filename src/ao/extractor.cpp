#include "extractor.hpp"

// template <typename T>
// void ao::extractor::GammatoneFilterbank<T>::compute(
//     const std::array<T, N>& input,
//     std::array<T, N>& output) {
    // const int num_samples  = input.size();
    // const int num_channels = stimulus_config.num_channels();
    // GOOGLE_CHECK_EQ(num_samples, output->cols());  // Samples.
    // GOOGLE_CHECK_EQ(num_channels, output->rows()); // Channels.
    // auto& center_frequencies = channel_properties->center_frequencies;
    // center_frequencies.resize(num_channels);

    // // Compute characteristic frequencies equally spaced on ERB scale using
    // // the canonical procedure.
    // const double low_erb  = erb::HzToErbRate(stimulus_config.lowest_cf_hz());
    // const double high_erb =
    // erb::HzToErbRate(stimulus_config.highest_cf_hz()); const double step_erb
    // = (high_erb - low_erb) / (num_channels - 1); const int sample_rate =
    // stimulus_config.sample_rate(); const double tpt      = (2.0 * kPi) /
    // sample_rate; for (int i = 0; i < num_channels; ++i) {
    //     // ==================
    //     // Set-up the filter:
    //     // ==================
    //     const double cf       = erb::ErbRateToHz(low_erb + i * step_erb);
    //     center_frequencies[i] = cf;
    //     const double tptbw    = tpt * erb::HzToErb(cf) *
    //     kBandwidthCorrection; const double a        = std::exp(-tptbw);
    //     // Based on integral of impulse response.
    //     const double gain = (tptbw * tptbw * tptbw * tptbw) / 3.0;

    //     // Filter coefficients.
    //     const double a1 = 4.0 * a;
    //     const double a2 = -6.0 * a * a;
    //     const double a3 = 4.0 * a * a * a;
    //     const double a4 = -a * a * a * a;
    //     const double a5 = a * a;
    //     double p1r = 0.0, p2r = 0.0, p3r = 0.0, p4r = 0.0, p1i = 0.0, p2i =
    //     0.0,
    //            p3i = 0.0, p4i = 0.0;

    //     // ===========================================================
    //     // exp(a+i*b) = exp(a)*(cos(b)+i*sin(b))
    //     // q = exp(-i*tpt*cf*t) = cos(tpt*cf*t) + i*(-sin(tpt*cf*t))
    //     // qcos = cos(tpt*cf*t)
    //     // qsin = -sin(tpt*cf*t)
    //     // ===========================================================
    //     const double coscf = std::cos(tpt * cf);
    //     const double sincf = std::sin(tpt * cf);
    //     double qcos = 1, qsin = 0; /* t=0 & q = exp(-i*tpt*t*cf)*/
    //     for (int j = 0; j < num_samples; ++j) {
    //         // Filter part 1 & shift down to d.c.
    //         double p0r =
    //             qcos * input(j) + a1 * p1r + a2 * p2r + a3 * p3r + a4 * p4r;
    //         if (p0r < kTinyEpsilon)
    //             p0r = 0.0;
    //         double p0i =
    //             qsin * input(j) + a1 * p1i + a2 * p2i + a3 * p3i + a4 * p4i;
    //         if (p0i < kTinyEpsilon)
    //             p0i = 0.0;

    //         // Filter part 2.
    //         const double u0r = p0r + a1 * p1r + a5 * p2r;
    //         const double u0i = p0i + a1 * p1i + a5 * p2i;

    //         // Update filter results.
    //         p4r = p3r;
    //         p3r = p2r;
    //         p2r = p1r;
    //         p1r = p0r;
    //         p4i = p3i;
    //         p3i = p2i;
    //         p2i = p1i;
    //         p1i = p0i;

    //         // Basilar membrane response
    //         // 1. Shift up in frequency first:
    //         //   (u0r+i*u0i) * exp(i*tpt*cf*t) = (u0r+i*u0i) *
    //         //     (qcos + i*(-qsin))
    //         // 2. Take the real part only:
    //         //   bm = real(exp(j*wcf*kT).*u) * gain.
    //         (*output)(i, j) = (u0r * qcos + u0i * qsin) * gain;

    //         // The basic idea of saving computational load:
    //         //   cos(a+b) = cos(a)*cos(b) - sin(a)*sin(b)
    //         //   sin(a+b) = sin(a)*cos(b) + cos(a)*sin(b)
    //         //   qcos = cos(tpt*cf*t) = cos(tpt*cf + tpt*cf*(t-1))
    //         //   qsin = -sin(tpt*cf*t) = -sin(tpt*cf + tpt*cf*(t-1))
    //         const double old_qcos = qcos;
    //         qcos                  = coscf * old_qcos + sincf * qsin;
    //         qsin                  = coscf * qsin - sincf * old_qcos;
    //     }
    // }
// }
