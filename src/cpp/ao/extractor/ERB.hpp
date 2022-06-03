#pragma once

namespace ao {
namespace extractor {

/**
 * ! Review
 * @brief Converts a frequency in Hz to its Equivalent Rectangular
 * Bandwidth.
 *
 * @param hz Frequency in Hz.
 * @return T Equivalent Rectangular Bandwidth.
 */
template <typename T> inline T Hz_to_ERBRate(const T hz) {
    return 21.4 * std::log10(4.37e-3 * (hz) + 1.0);
}

/**
 * ! Review
 * @brief Converts an Equivalent Rectangular Bandwidth to its frequency
 * in Hz.
 *
 * @param erb Equivalent Rectangular Bandwidth.
 * @return T Frequency in Hz.
 */
template <typename T> inline T ERBRate_to_Hz(const T erb) {
    return (std::pow(10.0, ((erb) / 21.4)) - 1.0) / 4.37e-3;
}

// TODO
template <typename T> inline T ERB(const T f) {
    return 24.7 * (4.37e-3 * (f) + 1.0);
}

} // namespace extractor
} // namespace ao