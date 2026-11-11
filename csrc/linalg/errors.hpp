// Errors raised by the core (docs/design/linalg.md §2.2).
#pragma once

#include <nanobind/nanobind.h>

#include <stdexcept>
#include <string>

namespace sd::linalg {

// Numerical failure: singular matrix, not positive definite, no convergence.
// Becomes stardust.linalg.LinAlgError (a ValueError) in Python.
struct LinAlgError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

void register_error_translator();

}  // namespace sd::linalg
