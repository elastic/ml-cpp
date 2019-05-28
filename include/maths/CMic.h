/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CMic_h
#define INCLUDED_ml_maths_CMic_h

#include <maths/CLinearAlgebra.h>

#include <boost/operators.hpp>

#include <vector>

class CMicTest;

namespace ml {
namespace maths {

//! \brief A mutual information based measure of the dependence between two variables.
//!
//! DESCRIPTION:\n
//! This implements the MICe measure defined in "Measuring dependence powerfully and
//! equitably" Reshef et al. The basic idea is to compute the mutual information:
//! \f$\sum_{x,y} P(x,y) \log\left(\frac{P(x,y)}{P(x)P(y)}\right)\f$ for grids satisfying
//! \f$k l < B(n)\f$ where \f$k\f$ and \f$l\f$ are the number of x- and y-axis bins and
//! \f$B(n) = n^{\alpha}\f$ for \f$\alpha < 1\f$. The score is normalized to account
//! for different granularity of bins and MICe is the maximum normalized score for this
//! collection of grids. See http://jmlr.csail.mit.edu/papers/volume17/15-308/15-308.pdf
//! for more information.
class CMic final : private boost::addable<CMic> {
public:
    //! Reserve space for the number of samples if this known in advance.
    void reserve(std::size_t n);

    //! Merge this and \p other.
    const CMic& operator+=(const CMic& other);

    //! Add a sample of the random variable (X, Y).
    void add(double x, double y);

    //! Compute MICe for the samples added.
    double compute();

private:
    using TDoubleVec = std::vector<double>;
    using TSizeVec = std::vector<std::size_t>;
    using TVector2d = CVectorNx1<double, 2>;
    using TVector2dVec = std::vector<TVector2d>;
    using TVectorXd = CVector<double>;
    using TVectorXdVec = std::vector<TVectorXd>;

private:
    void setup();
    std::size_t b() const;
    double c() const;
    TDoubleVec equipartitionAxis(std::size_t variable, std::size_t l) const;
    TDoubleVec optimizeXAxis(const TDoubleVec& q, std::size_t l, std::size_t k) const;
    static double entropy(const TVectorXd& dist);

private:
    TVector2dVec m_Samples;
    TSizeVec m_Order[2];

    friend class ::CMicTest;
};
}
}

#endif // INCLUDED_ml_maths_CMic_h
