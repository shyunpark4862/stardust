// The Hessian of minimize's Newton methods (docs/design/optimize.md §3.5): the user's hess and
// hessp, and scipy 1.18.1's difference Hessian, as ScalarFunction and the methods give them.
//
// ScalarFunction keeps the Hessian of its last point: hess(x) is called at x0 when it is built
// (after fun and the gradient there) and again only at another point (its x moves with every
// fun, gradient or Hessian request; Objective::point counts the moves). hess='2-point' and
// '3-point' are approx_derivative's linear operator of the gradient at the point: no call when
// it is made (it takes the point's gradient, computing it if it is not known), and a product with
// p != 0 is (grad(x + dx p) - g) / dx with dx = sqrt(eps) / |p|, or (grad(x + dx/2 p) -
// grad(x - dx/2 p)) / dx with dx = 2 eps^(1/3) / |p|; the gradients there are the user's jac
// past the memo (counted in njev, as scipy's ngev counts them). A product with p = 0 is 0.
//
// hess(x) returns (n, n) and hessp(x, p) returns (n,) (O63): a value that is not a float64
// sd.ndarray of that shape goes through Python's check. The matrix is kept row-major (numpy's
// C order, as scipy's np.asarray of the user's value), a copy shared with the subproblems that
// hold it: hess called again while one is held fills another buffer (kept for reuse).
#pragma once

#include "optimize/diff.hpp"
#include "optimize/objective.hpp"

#include "linalg/types.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace sd::optimize {

// numpy's np.dot of two vectors: a[0] b[0] for one element (numpy's scalar path), else 0.0 +
// cblas_ddot (DOUBLE_dot; the 0.0 turns BLAS's -0.0 into +0.0).
inline double np_dot(const double* a, const double* b, std::size_t n) {
    if (n == 1) return a[0] * b[0];
    return 0.0 + cblas_ddot(static_cast<linalg::lint>(n), a, 1, b, 1);
}

// numpy's np.dot of a C-ordered (n, n) matrix and a vector into out: cblas_dgemv row-major, or
// the product of the two numbers when n = 1.
inline void np_matvec(const double* H, const double* p, double* out, std::size_t n) {
    if (n == 1) {
        out[0] = p[0] * H[0];
        return;
    }
    const auto ln = static_cast<linalg::lint>(n);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, ln, ln, 1.0, H, ln, p, 1, 0.0, out, 1);
}

enum HessMode : int {
    kHessNone = 0,        // no hess: hessp, or (newton-cg) the gradient's differences
    kHessCallable = 1,    // hess(x) -> (n, n)
    kHessTwoPoint = 2,    // '2-point': products by differences of the gradient
    kHessThreePoint = 3,  // '3-point'
};

struct Hessian {
    using vec = std::vector<double>;

    Objective& obj;
    nb::handle hess, hessp;  // the user's functions (None when not given)
    int mode = kHessNone;
    std::size_t n;

    std::shared_ptr<vec> H;                  // kHessCallable: the matrix at the memo's point
    std::vector<std::shared_ptr<vec>> pool;  // the matrices' buffers (one held only here is free)
    vec hx, hg;  // differences: the point and its gradient
    std::uint64_t at = 0;
    bool valid = false;
    std::int64_t calls = 0;  // the calls of hess (ScalarFunction's nhev)
    vec xt, g1, g2;

    Hessian(Objective& o, nb::handle h, nb::handle hp, int m) : obj(o), hess(h), hessp(hp), mode(m), n(o.n) {
        H = buffer();
        xt.resize(n);
        g1.resize(n);
        g2.resize(n);
    }

    std::shared_ptr<vec> buffer() {
        for (auto& b : pool)
            if (b.use_count() == 1) return b;
        pool.push_back(std::make_shared<vec>(n * n));
        return pool.back();
    }

    // A returned (n, n) matrix into out (row-major).
    void matrix(nb::handle o, double* out) const {
        nb::object keep;
        const auto ln = static_cast<std::int64_t>(n);
        if (!(is_ndarray(o) && as_ndarray(o).dtype == DType::Float64 && as_ndarray(o).ndim == 2 &&
              as_ndarray(o).shape[0] == ln && as_ndarray(o).shape[1] == ln)) {
            keep = obj.check(o, "hess");
            o = keep;
        }
        const NDArray& a = as_ndarray(o);
        const std::byte* d = a.data;
        const std::int64_t s0 = a.strides[0], s1 = a.strides[1];
        if (s1 == static_cast<std::int64_t>(sizeof(double)) && s0 == ln * s1) {  // C order
            std::memcpy(out, d, n * n * sizeof(double));
            return;
        }
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                std::memcpy(
                    &out[i * n + j], d + static_cast<std::int64_t>(i) * s0 + static_cast<std::int64_t>(j) * s1,
                    sizeof(double)
                );
    }

    // A returned (n,) product into out.
    void product_value(nb::handle o, double* out) const {
        nb::object keep;
        if (!(is_ndarray(o) && as_ndarray(o).dtype == DType::Float64 && as_ndarray(o).ndim == 1 &&
              as_ndarray(o).shape[0] == static_cast<std::int64_t>(n))) {
            keep = obj.check(o, "hessp");
            o = keep;
        }
        const NDArray& a = as_ndarray(o);
        const std::byte* d = a.data;
        const std::int64_t s = a.strides[0];
        for (std::size_t i = 0; i < n; ++i) std::memcpy(&out[i], d + static_cast<std::int64_t>(i) * s, sizeof(double));
    }

    // ScalarFunction.hess(v): the matrix (kHessCallable) or the difference operator at v.
    void update(const double* v) {
        obj.set_x(v);
        if (valid && at == obj.point) return;
        if (mode == kHessCallable) {
            H.reset();
            H = buffer();  // the last one unless a subproblem holds it
            matrix(hess(obj.array(obj.x.data())), H->data());
            ++calls;
        } else {
            obj.update_grad();
            hx = obj.x;
            hg = obj.g;
        }
        at = obj.point;
        valid = true;
    }

    bool matrix_finite() const {
        for (double t : *H)
            if (!std::isfinite(t)) return false;
        return true;
    }

    // The difference operator's product with p into out (approx_derivative's matvec).
    void difference_product(const double* p, double* out) {
        bool zero = true;
        for (std::size_t i = 0; i < n && zero; ++i) zero = p[i] == 0;
        if (zero) {
            for (std::size_t i = 0; i < n; ++i) out[i] = 0.0;
            return;
        }
        const double norm = std::sqrt(np_dot(p, p, n));  // numpy's norm: sqrt(p . p)
        if (mode == kHessTwoPoint) {
            const double dx = rel_step_2point() / norm;
            for (std::size_t i = 0; i < n; ++i) xt[i] = hx[i] + dx * p[i];
            obj.raw_gradient(xt.data(), g1.data());
            for (std::size_t i = 0; i < n; ++i) out[i] = (g1[i] - hg[i]) / dx;
            return;
        }
        const double dx = 2 * rel_step_3point() / norm;
        for (std::size_t i = 0; i < n; ++i) xt[i] = hx[i] - (dx / 2) * p[i];
        obj.raw_gradient(xt.data(), g1.data());
        for (std::size_t i = 0; i < n; ++i) xt[i] = hx[i] + (dx / 2) * p[i];
        obj.raw_gradient(xt.data(), g2.data());
        for (std::size_t i = 0; i < n; ++i) out[i] = (g2[i] - g1[i]) / dx;
    }

    // The user's hessp(x, p) into out.
    void user_product(const double* x, const double* p, double* out) const {
        product_value(hessp(obj.array(x), obj.array(p)), out);
    }
};

}  // namespace sd::optimize
