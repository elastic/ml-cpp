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

#ifndef INCLUDED_ml_maths_CEntropySketch_h
#define INCLUDED_ml_maths_CEntropySketch_h

#include <maths/ImportExport.h>

#include <cstddef>
#include <stdint.h>
#include <vector>

namespace ml {
namespace maths {

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
class MATHS_EXPORT CEntropySketch {
public:
    CEntropySketch(std::size_t k);

    //! Add \p category with count of \p count.
    void add(std::size_t category, uint64_t count = 1);

    //! Compute the entropy based on the values added so far.
    double calculate(void) const;

private:
    typedef std::vector<double> TDoubleVec;
    typedef std::vector<uint64_t> TUInt64Vec;

private:
    //! Generate the projection of the category counts.
    void generateProjection(std::size_t category, TDoubleVec &projection);

private:
    //! The overall count.
    uint64_t m_Y;

    //! The sketch count.
    TDoubleVec m_Yi;
};
}
}

#endif// INCLUDED_ml_maths_CEntropySketch_h
