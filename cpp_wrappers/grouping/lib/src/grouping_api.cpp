#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "grouping_gpu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("grouping_wrapper", &grouping_wrapper, "grouping operation in gpu");

}