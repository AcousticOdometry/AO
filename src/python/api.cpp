#include "extractor.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(_python_api, m) {
    m.doc() = R"pbdoc(
        Python wrapper for `myproject`.

        This information will be displayed when using `help()`:
        $ python -c "import myproject; help(myproject)"
    )pbdoc";

    py::class_<ao::extractor::GammatoneFilterbank<double>>(
        m, "GammatoneFilterbank")
        .def(py::init<size_t, size_t, int, double, double>())
        .def("__call__",
            &ao::extractor::GammatoneFilterbank<double>::operator(),
            py::return_value_policy::move);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
