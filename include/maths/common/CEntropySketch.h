/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#ifndef INCLUDED_ml_maths_common_CEntropySketch_h
#define INCLUDED_ml_maths_common_CEntropySketch_h

#include <maths/common/ImportExport.h>

#include <cstddef>
#include <stdint.h>
#include <vector>

namespace ml {
namespace maths {
namespace common {
//! \brief A sketch data structure for computing the Shannon entropy of a data
//! stream under the turnstile model.
//!
//! DESCRIPTION:\n
//! Computes an \f$(\epsilon, \rho)\f$ approximation of the empirical Shannon
//! entropy, i.e. \f$-p_i \log(p_i)\f$ where \f$p_i\f$ is the empirical frequency
//! of the i'th category, of a data stream of length \f$T\f$ when setting
//! \f$k = O\left(\frac{1}{\epsilon^2}\right)\f$. Note the space complexity
//! is \f$\frac{1}{\epsilon^2} \log(T) \log(\frac{T}{\epsilon}\f$.
//!
//! See http://www.jmlr.org/proceedings/papers/v31/clifford13a.pdf for details.
class MATHS_COMMON_EXPORT CEntropySketch {
public:
    CEntropySketch(std::size_t k);

    //! Add \p category with count of \p count.
    void add(std::size_t category, uint64_t count = 1);

    //! Compute the entropy based on the values added so far.
    double calculate() const;

private:
    using TDoubleVec = std::vector<double>;
    using TUInt64Vec = std::vector<uint64_t>;

private:
    //! Generate the projection of the category counts.
    void generateProjection(std::size_t category, TDoubleVec& projection);

private:
    //! The overall count.
    uint64_t m_Y;

    //! The sketch count.
    TDoubleVec m_Yi;
};
}
}
}

#endif // INCLUDED_ml_maths_common_CEntropySketch_h
