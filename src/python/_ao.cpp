#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

void declareExtractor(py::module& mod);

PYBIND11_MODULE(_ao, mod) {
    mod.doc() = R"pbdoc(
        Acoustic Odometry library 
    )pbdoc";

    // py::class_<ao::AO<double>>(mod, "AO")
    //     .def(
    //         py::init<size_t, size_t, int>(),
    //         "num_samples"_a  = 1024,
    //         "num_features"_a = 64,
    //         "sample_rate"_a  = 44100)
    //     .def(
    //         "__call__",
    //         &ao::AO<double>::operator(),
    //         "input"_a,
    //         py::return_value_policy::move);

    declareExtractor(mod);

#ifdef VERSION_INFO
    mod.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    mod.attr("__version__") = "dev";
#endif
}
