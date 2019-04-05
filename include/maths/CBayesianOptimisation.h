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
//! evaluating it, i.e. situations where multi-arm bandit strategies are likely
//! to perform better.
//!
//! This implementation fits the maximum likelihood mean zero Gaussian Process
//! to a collection of function evaluations \f$(x, f(x), \sigma^2(x))\f$ where
//! \f$\sigma^2(x)\f$ is the error associated with the measurement of \f$f(x)\f$.
//! It then computes the maximum expected improvement location defined (for
//! function *minimization*) as follows:
//! <pre class="fragment">
//!   \f$\displaystyle arg\max_x \doubleE_x[max(f^* - f, 0)]\f$
//! </pre>
//!
//! Here, the expectation is w.r.t. the distribution of the maginal of the Gaussian
//! Process at the point x.
class MATHS_EXPORT CBayesianOptimisation {
public:
    using TDoubleDoublePr = std::pair<double, double>;
    using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
    using TVector = CDenseVector<double>;
    using TLikelihoodFunc = std::function<double(const TVector&)>;
    using TLikelihoodGradientFunc = std::function<TVector(const TVector&)>;
    using TEIFunc = std::function<double(const TVector&)>;
    using TEIGradientFunc = std::function<TVector(const TVector&)>;

public:
    CBayesianOptimisation(TDoubleDoublePrVec parameterBounds);

    //! Add the result of evaluating the function to be \p fx at \p x where the
    //! variance in the error in \p fx w.r.t. the true value is \p vx.
    void add(TVector x, double fx, double vx);

    //! Compute the location which maximizes the expected improvement given the
    //! function evaluations added so far.
    TVector maximumExpectedImprovement();

    //! \name Test Interface
    //@{
    //! Get minus the data likelihood and its gradient as a function of the kernel
    //! hyperparameters.
    std::pair<TLikelihoodFunc, TLikelihoodGradientFunc> minusLikelihoodAndGradient() const;

    //! Get minus the expected improvement in the target function and its gradient
    //! as a function of the evaluation position.
    std::pair<TEIFunc, TEIGradientFunc> minusExpectedImprovementAndGradient() const;

    //! Compute the maximum likelihood kernel parameters.
    const TVector& maximumLikelihoodKernel();
    //@}

private:
    using TDoubleVec = std::vector<double>;
    using TVectorDoublePr = std::pair<TVector, double>;
    using TVectorDoublePrVec = std::vector<TVectorDoublePr>;
    using TMatrix = CDenseMatrix<double>;

private:
    //! This lower bounds the coefficient associated with coordinate separation
    //! in the power exponential kernel:
    //! <pre class="fragment">
    //!   \f$\displaystyle K(x,y|\theta_0, \theta_1) = \theta_0^2 exp(-(x-y)^t D(\theta_1) (x-y)))\f$
    //! </pre>
    //! where \f$[D(\theta_1)]_{ij} = (\theta_{1,i}^2 + \epsilon)\delta_{ij}\f$
    //! with \f$\epsilon\f$ being this constant. This stops the expected improvement
    //! gradient collasping to zero in any direction.
    static const double MINIMUM_KERNEL_COORDINATE_DISTANCE_SCALE;

private:
    void precondition();
    TVector function() const;
    double functionVariance() const;
    double meanErrorVariance() const;
    TMatrix dKerneld(const TVector& a, int k) const;
    TMatrix kernel(const TVector& a, double v) const;
    TVectorDoublePr kernelCovariates(const TVector& a, const TVector& x, double vx) const;
    double kernel(const TVector& a, const TVector& x, const TVector& y) const;

private:
    CPRNG::CXorOShiro128Plus m_Rng;
    std::size_t m_Restarts = 10;
    double m_RangeShift = 0.0;
    double m_RangeScale = 1.0;
    TVector m_A;
    TVector m_B;
    TVectorDoublePrVec m_Function;
    TDoubleVec m_ErrorVariances;
    TVector m_KernelParameters;
    TVector m_MinimumKernelCoordinateDistanceScale;
};
}
}
