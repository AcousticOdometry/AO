// #include "ao.hpp"

#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
// using namespace py::literals;

void declareExtractor(py::module& mod);

PYBIND11_MODULE(_ao, mod) {
    mod.doc() = R"pbdoc(
        Acoustic Odometry library 
    )pbdoc";

    // py::class_<ao::AO<double>>(mod, "AO")
    //     .def(
    //         py::init<
    //             std::filesystem::path,
    //             std::vector<ao::extractor::Extractor<double>*>,
    //             size_t,
    //             std::string>(),
    //         "model_path"_a,
    //         "extractors"_a,
    //         "num_frames"_a,
    //         "device_string"_a = "cpu")
    //     .def(
    //         "__call__",
    //         &ao::AO<double>::operator(),
    //         "samples"_a,
    //         py::return_value_policy::move);

    declareExtractor(mod);

#ifdef VERSION_INFO
    mod.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    mod.attr("__version__") = "dev";
#endif
}
