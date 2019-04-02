/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CLinearAlgebraEigen.h>
#include <maths/CPRNG.h>
#include <maths/ImportExport.h>

#include <functional>
#include <utility>
#include <vector>

namespace ml {
namespace maths {

//! \brief Implements maximum expected increase Bayesian optimisation.
//!
//! DESCRIPTION:\n
//! This is a minimizer for a black box function which is expensive to evaluate
//! that attempts to intelligently choose evaluation points to maximise the
//! improvement per evaluation. It is intended mainly for model hyperparameter
//! optimisation where evaluating a function requires fitting a model.
//!
//! The usual caveats apply: it doesn't work well for high dimensional functions
//! (more than around 10 parameters) and there are better strategies available
//! if one can can obtain information about the function more cheaply than fully
//! evaluating it, i.e. situations where multiarm bandit strategies are likely
//! to perform better.
//!
//! This implementation fits the maximum likelihood mean zero Gaussian Process
//! to a collection of function evaluations \f$(x, f(x), \sigma^2(x))\f$ where
//! \f$\sigma^2(x)\f$ is the error associated with the measurement of \f$f(x)\f$.
//! It then computes the maximum expected improvement location defined as follows:
//! <pre class="fragment">
//!   \f$\displaystyle arg\max_x \doubleE_x[max(f^* - f, 0)]\f$
//! </pre>
//!
//! Here, the expectation is w.r.t. the distribution of the maginal of the Gaussian
//! Process value at the point x.
class MATHS_EXPORT CBayesianOptimisation {
public:
    using TDoubleDoublePr = std::pair<double, double>;
    using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
    using TVector = CDenseVector<double>;
    using TLikelihoodFunc = std::function<double(const TVector&)>;
    using TLikelihoodGradientFunc = std::function<TVector(const TVector&)>;
    using TExpectedImprovementFunc = std::function<double(const TVector&)>;

public:
    CBayesianOptimisation(TDoubleDoublePrVec parameterBounds);

    //! Add the result of evaluating the function to be \p fx at p x where the
    //! variance in the value of \p fx w.r.t. the true value is \p vx.
    void add(TVector x, double fx, double vx);

    //! Compute the location which maximizes the expected improvement given the
    //! function evaluations added so far.
    TVector maximumExpectedImprovement();

    //! Get minus the data likelihood as a function of the kernel hyperparameters.
    TLikelihoodFunc minusLikelihood() const;

    //! Get the gradient of minus the data likelihood as a function of the kernel
    //! hyperparameters.
    TLikelihoodGradientFunc minusLikelihoodGradient() const;

    //! Get minus the expected improvement in the target function as a function
    //! of the evaluation position.
    TExpectedImprovementFunc minusExpectedImprovement() const;

private:
    using TDoubleVec = std::vector<double>;
    using TVectorDoublePr = std::pair<TVector, double>;
    using TVectorDoublePrVec = std::vector<TVectorDoublePr>;
    using TMatrix = CDenseMatrix<double>;

private:
    void maximumLikelihoodKernel();
    void precondition();
    TVector function() const;
    double meanErrorVariance() const;
    TMatrix distanceMatrix(int coord) const;
    TMatrix kernel(const TVector& a, double v) const;
    TVectorDoublePr kernelCovariates(const TVector& a, const TVector& x, double vx) const;
    static double kernel(const TVector& a, const TVector& x, const TVector& y);

private:
    CPRNG::CXorOShiro128Plus m_Rng;
    std::size_t m_Restarts = 10;
    double m_RangeShift = 0.0;
    double m_RangeScale = 1.0;
    TDoubleDoublePrVec m_ParameterBounds;
    TVector m_DomainScales;
    TVectorDoublePrVec m_Function;
    TDoubleVec m_ErrorVariances;
    TVector m_KernelParameters;
};
}
}
