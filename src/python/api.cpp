#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

void declareExtractor(py::module& mod);

PYBIND11_MODULE(_python_api, mod) {
    mod.doc() = R"pbdoc(
        Python wrapper for `myproject`.

        This information will be displayed when using `help()`:
        $ python -c "import myproject; help(myproject)"
    )pbdoc";

    declareExtractor(mod);

#ifdef VERSION_INFO
    mod.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    mod.attr("__version__") = "dev";
#endif
}
