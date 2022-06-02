#pragma once

#include "extractor.hpp"

namespace ao {

/**
 * @brief Acoustic Odometry class
 *
 * @tparam T Input signal type
 */
template <typename T> class AO {
    public:
    const extractor::Extractor<T> extract;
    // TODO model input parameters
    const size_t num_frames;
    const size_t num_channels;

    /**
     * @brief Construct a new AO object
     *
     */
    AO(const std::string& model_path,
       const std::string& device_string,
       const extractor::Extractor<T>& extract)
    : extract(extract) {
        // TODO load model
        // TODO build input tensor
    }

    protected:
    torch::jit::script::Module model;
}

} // namespace ao