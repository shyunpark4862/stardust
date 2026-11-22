// sd.random's bindings: the submodule stardust._core.random (docs/design/random.md).
#pragma once

#include <nanobind/nanobind.h>

#include <atomic>
#include <cstdint>

namespace sd {

// Call after the array bindings: the kernels return the ndarray type registered there.
void bind_random(nanobind::module_& core);

namespace random {

// A Generator's key and stream position (stardust._core.random.State; the Generator keeps
// one): reserve(n) takes n elements, (key, position) read and the position moved under one
// spin lock, so a scalar draw needs no Python lock and a state assigned from another thread
// is seen whole or not at all. Native callers (the scalar samplers) reserve directly.
class State {
public:
    State(std::uint64_t k0, std::uint64_t k1, std::uint64_t pos) : k0_(k0), k1_(k1), pos_(pos) {}
    struct Taken {
        std::uint64_t k0, k1, pos;
    };
    Taken reserve(std::uint64_t n) {
        Guard g(busy_);
        const Taken t{k0_, k1_, pos_};
        pos_ += n;  // modulo 2^64
        return t;
    }
    Taken get() {
        Guard g(busy_);
        return {k0_, k1_, pos_};
    }
    void set(std::uint64_t k0, std::uint64_t k1, std::uint64_t pos) {
        Guard g(busy_);
        k0_ = k0, k1_ = k1, pos_ = pos;
    }

private:
    struct Guard {
        std::atomic_flag& f;
        explicit Guard(std::atomic_flag& flag) : f(flag) {
            while (f.test_and_set(std::memory_order_acquire)) {}
        }
        ~Guard() { f.clear(std::memory_order_release); }
    };
    std::uint64_t k0_, k1_, pos_;
    std::atomic_flag busy_ = ATOMIC_FLAG_INIT;
};

}  // namespace random

}  // namespace sd
