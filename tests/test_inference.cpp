#include <gtest/gtest.h>

#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

TEST(TestInference, LibTorch) {
    torch::jit::script::Module module;

    module = torch::jit::load(
        "/home/esdandreu/AO/models/"
        "torch-script;name_numpy-arrays;date_2022-05-23;time_13-39-14.pt",
        torch::Device("cpu"));
    module.dump(true, true, true);
    std::cout << "ok\n";
}