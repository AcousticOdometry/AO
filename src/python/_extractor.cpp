#include "ao/extractor.hpp"

#include <pybind11/pybind11.h>

#include <pybind11/functional.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;
using namespace ao::extractor;

class PyExtractor : public Extractor<double> {
    public:
    /* Inherit the constructors */
    using Extractor<double>::Extractor;

    /* Trampoline (need one for each virtual function) */
    void compute(const std::vector<double>& input, std::vector<double>& output)
        const override {
        PYBIND11_OVERRIDE_PURE(
            void,      /* Return type */
            Extractor, /* Parent class */
            compute,   /* Name of function in C++ (must match Python name) */
            /* Argument(s) */
            input,
            output);
    }
};

void declareExtractor(py::module& mod) {
    // Bind the `ao.extractor` namespace as a python submodule
    py::module modExtractor = mod.def_submodule("extractor");

    // Bind Extractor base class, python users should be able to extend this
    // class, creating pure python extractors at the cost of performance
    py::class_<Extractor<double>, PyExtractor>(modExtractor, "Extractor")
        .def(
            py::init<size_t, size_t, int, std::function<double(double)>>(),
            "num_samples"_a  = 1024,
            "num_features"_a = 64,
            "sample_rate"_a  = 44100,
            "transform"_a    = py::cpp_function([](double x) { return x; }))
        .def(
            "__call__",
            static_cast<std::vector<double> (Extractor<double>::*)(
                const std::vector<double>&) const>(
                &Extractor<double>::operator()),
            "input"_a,
            py::return_value_policy::move);
    // Bind the Gammatone Filterbank as a subclass of Extractor
    auto gammatone_filterbank =
        py::class_<GammatoneFilterbank<double>, Extractor<double>>(
            modExtractor, "GammatoneFilterbank")
            .def(
                py::init<
                    size_t,
                    size_t,
                    int,
                    std::function<double(double)>,
                    double,
                    double,
                    double>(),
                "num_samples"_a  = 1024,
                "num_features"_a = 64,
                "sample_rate"_a  = 44100,
                "transform"_a    = py::cpp_function(
                    static_cast<double (*)(double)>(std::log10)),
                "low_Hz"_a               = 50,
                "high_Hz"_a              = 8000,
                "temporal_integration"_a = 0)
            .def_readonly("filters", &GammatoneFilterbank<double>::filters);
    // Bind the Filter nested class into the GammatoneFilterbank
    py::class_<GammatoneFilterbank<double>::Filter>(
        gammatone_filterbank, "Filter")
        .def(
            py::init<double, double, double, double, std::array<double, 5>>(),
            "cf"_a,
            "coscf"_a,
            "sincf"_a,
            "gain"_a,
            "a"_a)
        .def_readonly("cf", &GammatoneFilterbank<double>::Filter::cf)
        .def_readonly("gain", &GammatoneFilterbank<double>::Filter::gain)
        .def_readonly("a", &GammatoneFilterbank<double>::Filter::a);
}
