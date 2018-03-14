/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#include <maths/CSpline.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

namespace ml {
namespace maths {
namespace spline_detail {

namespace {

//! Sanity check the diagonals and the vector dimensions are
//! consistent.
bool checkTridiagonal(const TDoubleVec& a,
                      const TDoubleVec& b,
                      const TDoubleVec& c,
                      const TDoubleVec& x) {
    if (a.size() + 1 != b.size()) {
        LOG_ERROR("Lower diagonal and main diagonal inconsistent:"
                  << " a = " << core::CContainerPrinter::print(a)
                  << " b = " << core::CContainerPrinter::print(b));
        return false;
    }
    if (c.size() + 1 != b.size()) {
        LOG_ERROR("Upper diagonal and main diagonal inconsistent:"
                  << " b = " << core::CContainerPrinter::print(b)
                  << " c = " << core::CContainerPrinter::print(c));
        return false;
    }
    if (b.size() != x.size()) {
        LOG_ERROR("Dimension mismatch:"
                  << " x = " << core::CContainerPrinter::print(x)
                  << ", b = " << core::CContainerPrinter::print(b))
        return false;
    }
    return true;
}
}

bool solveTridiagonal(const TDoubleVec& a, const TDoubleVec& b, TDoubleVec& c, TDoubleVec& x) {
    if (!checkTridiagonal(a, b, c, x)) {
        return false;
    }

    LOG_TRACE("a = " << core::CContainerPrinter::print(a));
    LOG_TRACE("b = " << core::CContainerPrinter::print(b));
    LOG_TRACE("c = " << core::CContainerPrinter::print(c));
    LOG_TRACE("x = " << core::CContainerPrinter::print(x));

    // Solve using the (Llewellyn) Thomas algorithm.
    //
    // Note that we do not attempt to handle the case that
    // the matrix is rank deficient to working precision and
    // simply fail. This should only happen if the main knot
    // points in the spline get too close which is handled
    // in the calling code.

    std::size_t n = x.size();

    // Eliminate the lower diagonal.
    if (b[0] == 0.0) {
        LOG_ERROR("Badly conditioned: " << core::CContainerPrinter::print(b));
        return false;
    }
    c[0] = c[0] / b[0];
    x[0] = x[0] / b[0];
    for (std::size_t i = 1; i + 1 < n; ++i) {
        double m = (b[i] - a[i - 1] * c[i - 1]);
        if (m == 0.0) {
            LOG_ERROR("Badly conditioned: " << core::CContainerPrinter::print(b));
            return false;
        }
        c[i] = c[i] / m;
        x[i] = (x[i] - a[i - 1] * x[i - 1]) / m;
    }
    double m = (b[n - 1] - a[n - 2] * c[n - 2]);
    if (m == 0.0) {
        LOG_ERROR("Badly conditioned: " << core::CContainerPrinter::print(b));
        return false;
    }
    x[n - 1] = (x[n - 1] - a[n - 2] * x[n - 2]) / m;

    // Back substitution.
    for (std::size_t i = n - 1; i-- > 0; /**/) {
        x[i] -= c[i] * x[i + 1];
    }

    LOG_TRACE("x = " << core::CContainerPrinter::print(x));

    return true;
}

bool solvePeturbedTridiagonal(const TDoubleVec& a,
                              const TDoubleVec& b,
                              TDoubleVec& c,
                              TDoubleVec& u,
                              const TDoubleVec& v,
                              TDoubleVec& x) {
    if (!checkTridiagonal(a, b, c, x)) {
        return false;
    }

    // This uses the Sherman-Morrison formula and notes that we
    // can write the solution of (A + uv') x = y as the solution
    // to A z = y plus a correction. Specifically, the correction
    // is found by solving A w = u, and is given by:
    //   z - ((v'z) / (1 + v'w)) w
    //
    // The elimination step updates the superdiagonal in the same
    // way for both solutions so they can be done together.

    // Solve using the (Llewellyn) Thomas algorithm.
    //
    // Note that we do not attempt to handle the case that
    // the matrix is rank deficient to working precision and
    // simply fail. This should only happen if the main knot
    // points in the spline get too close which is handled
    // in the calling code.

    LOG_TRACE("a = " << core::CContainerPrinter::print(a));
    LOG_TRACE("b = " << core::CContainerPrinter::print(b));
    LOG_TRACE("c = " << core::CContainerPrinter::print(c));
    LOG_TRACE("u = " << core::CContainerPrinter::print(u));
    LOG_TRACE("v = " << core::CContainerPrinter::print(v));
    LOG_TRACE("x = " << core::CContainerPrinter::print(x));

    std::size_t n = x.size();

    // Eliminate the lower diagonal.
    if (b[0] == 0.0) {
        LOG_ERROR("Badly conditioned: " << core::CContainerPrinter::print(b));
        return false;
    }
    c[0] = c[0] / b[0];
    x[0] = x[0] / b[0];
    u[0] = u[0] / b[0];
    for (std::size_t i = 1; i + 1 < n; ++i) {
        double m = (b[i] - a[i - 1] * c[i - 1]);
        if (m == 0.0) {
            LOG_ERROR("Badly conditioned: " << core::CContainerPrinter::print(b));
            return false;
        }
        c[i] = c[i] / m;
        x[i] = (x[i] - a[i - 1] * x[i - 1]) / m;
        u[i] = (u[i] - a[i - 1] * u[i - 1]) / m;
    }
    double m = (b[n - 1] - a[n - 2] * c[n - 2]);
    if (m == 0.0) {
        LOG_ERROR("Badly conditioned: " << core::CContainerPrinter::print(b));
        return false;
    }
    x[n - 1] = (x[n - 1] - a[n - 2] * x[n - 2]) / m;
    u[n - 1] = (u[n - 1] - a[n - 2] * u[n - 2]) / m;

    // Back substitution.
    for (std::size_t i = n - 1; i-- > 0; /**/) {
        x[i] = x[i] - c[i] * x[i + 1];
        u[i] = u[i] - c[i] * u[i + 1];
    }

    // Apply the correction.
    double vx = 0.0;
    double vu = 0.0;
    for (std::size_t i = 0u; i < n; ++i) {
        vx += v[i] * x[i];
        vu += v[i] * u[i];
    }
    double delta = vx / (1.0 + vu);
    for (std::size_t i = 0u; i < n; ++i) {
        x[i] -= delta * u[i];
    }

    LOG_TRACE("x = " << core::CContainerPrinter::print(x));

    return true;
}
}
}
}
