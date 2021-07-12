/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CLowess_h
#define INCLUDED_ml_maths_CLowess_h

#include <maths/CBasicStatistics.h>
#include <maths/CLeastSquaresOnlineRegression.h>

#include <utility>
#include <vector>

namespace ml {
namespace maths {

//! \brief LOWESS regression using order N polynomial.
//!
//! DESCRIPTION:\n
//! For more details see https://en.wikipedia.org/wiki/Local_regression.
template<std::size_t N>
class CLowess {
public:
    using TDoubleDoublePr = std::pair<double, double>;
    using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
    using TPolynomial = CLeastSquaresOnlineRegression<N>;

public:
    //! Fit a polynomial LOWESS model to \p data choosing the weight function to
    //! maximize the likelihood of \p numberFolds hold out sets.
    //!
    //! \param[in] data The training data.
    //! \param[in] numberFolds The number of folds to use in cross-validation to
    //! compute the best weight function from the family exp(-k |xi - xj|) with
    //! k a free parameter which determines the amount of smoothing to use.
    void fit(TDoubleDoublePrVec data, std::size_t numberFolds);

    //! Predict the value at \p x.
    //!
    //! \note Defined as zero if no data have been fit.
    double predict(double x) const;

    //! Compute the minimum of the function on the training data interval.
    //!
    //! \note Defined as (0,0) if no data have been fit.
    TDoubleDoublePr minimum() const;

    //! \name Test Only
    //@{
    //! Get an estimate of residual variance at the observed values.
    //!
    //! \note Defined as zero if no data have been fit.
    double residualVariance() const;

    //! Get how far we are prepared to extrapolate as the interval we will search
    //! in the minimum and sublevelSet functions.
    TDoubleDoublePr extrapolationInterval() const;
    //@}

private:
    using TDoubleVec = std::vector<double>;
    using TSizeVec = std::vector<std::size_t>;
    using TSizeVecVec = std::vector<TSizeVec>;
    using TSizeVecCItr = TSizeVec::const_iterator;
    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

private:
    void setupMasks(std::size_t numberFolds, TSizeVecVec& trainingMasks, TSizeVecVec& testingMasks) const;
    double likelihood(TSizeVecVec& trainingMasks, TSizeVecVec& testingMasks, double k) const;
    TPolynomial fit(TSizeVecCItr beginMask, TSizeVecCItr endMask, double k, double x) const;
    double weight(double k, double x1, double x2) const;

private:
    TDoubleDoublePrVec m_Data;
    TSizeVec m_Mask;
    //! The weight to assign to data points when fitting polynomial at x is given
    //! by exp(-k |xi - xj|). This can therefore be thought of as the inverse of
    //! the amount of smoothing.
    double m_K{0.0};
};
}
}

#endif // INCLUDED_ml_maths_CLowess_h
