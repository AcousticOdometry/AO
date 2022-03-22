#include "extractor.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace py::literals;
using namespace ao::extractor;

class PyExtractor : public Extractor<double> {
    public:
    /* Inherit the constructors */
    using Extractor<double>::Extractor;


    /* Trampoline (need one for each virtual function) */
    void compute(const std::vector<double>& input,
        std::vector<double>& output) override {
        PYBIND11_OVERRIDE_PURE(void, /* Return type */
            Extractor,               /* Parent class */
            compute,      /* Name of function in C++ (must match Python name) */
            input, output /* Argument(s) */
        );
    }
};

PYBIND11_MODULE(_python_api, m) {
    m.doc() = R"pbdoc(
        Python wrapper for `myproject`.

        This information will be displayed when using `help()`:
        $ python -c "import myproject; help(myproject)"
    )pbdoc";

    // Extractor
    py::class_<Extractor<double>, PyExtractor>(m, "Extractor")
        .def(py::init<size_t, size_t, int>(), "num_samples"_a = 1024,
            "num_features"_a = 64, "sample_rate"_a = 44100);

    // GammatoneFilterbank
    py::class_<GammatoneFilterbank<double>, Extractor<double>>(
        m, "GammatoneFilterbank")
        .def(py::init<size_t, size_t, int, double, double>(),
            "num_samples"_a = 1024, "num_features"_a = 64,
            "sample_rate"_a = 44100, "low_Hz"_a = 50, "high_Hz"_a = 8000)
        .def("__call__", &GammatoneFilterbank<double>::operator(), "input"_a,
            py::return_value_policy::move);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
