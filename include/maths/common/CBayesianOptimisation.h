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

#ifndef INCLUDED_ml_maths_common_CBayesianOptimisation_h
#define INCLUDED_ml_maths_common_CBayesianOptimisation_h

#include <core/CDataSearcher.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CoreTypes.h>

#include <maths/common/CLinearAlgebraEigen.h>
#include <maths/common/CPRNG.h>
#include <maths/common/ImportExport.h>

#include <boost/optional.hpp>

#include <functional>
#include <utility>
#include <vector>

namespace ml {
namespace maths {
namespace common {
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
//! Here, the expectation is w.r.t. the distribution of the marginal of the Gaussian
//! Process at the point x.
class MATHS_COMMON_EXPORT CBayesianOptimisation {
public:
    using TDoubleDoublePr = std::pair<double, double>;
    using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
    using TOptionalDouble = boost::optional<double>;
    using TVector = CDenseVector<double>;
    using TLikelihoodFunc = std::function<double(const TVector&)>;
    using TLikelihoodGradientFunc = std::function<TVector(const TVector&)>;
    using TEIFunc = std::function<double(const TVector&)>;
    using TEIGradientFunc = std::function<TVector(const TVector&)>;

public:
    static const std::size_t RESTARTS;
    // For values less than this the EI is treated as zero and we fallback to using
    // dissimilarity (i.e. maximizing diversity). Note that the supplied function
    // values are scaled so their variance is one thus this is a relative constant.
    static const double NEGLIGIBLE_EXPECTED_IMPROVEMENT;

public:
    CBayesianOptimisation(TDoubleDoublePrVec parameterBounds, std::size_t restarts = RESTARTS);
    CBayesianOptimisation(core::CStateRestoreTraverser& traverser);

    //! Add the result of evaluating the function to be \p fx at \p x where the
    //! variance in the error in \p fx w.r.t. the true value is \p vx.
    void add(TVector x, double fx, double vx);

    //! Any portion of the variance of the function error which is explained and
    //! shouldn't be included in the kernel.
    void explainedErrorVariance(double vx);

    //! Get the bounding box (in the function domain) in which we're minimizing.
    std::pair<TVector, TVector> boundingBox() const;

    //! Compute the location which maximizes the expected improvement given the
    //! function evaluations added so far.
    std::pair<TVector, TOptionalDouble>
    maximumExpectedImprovement(double negligibleExpectedImprovement = NEGLIGIBLE_EXPECTED_IMPROVEMENT);

    //! Persist by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Populate the object from serialized data
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Get the memory used by this object.
    std::size_t memoryUsage() const;

    //! Estimate the maximum booking memory used by this class for optimising
    //! \p numberParameters using \p numberRounds rounds.
    static std::size_t estimateMemoryUsage(std::size_t numberParameters,
                                           std::size_t numberRounds);

    //! Evaluate the Guassian process at the point \p input.
    double evaluate(const TVector& input) const;

    //! Compute the marginalized value of the Gaussian process in the dimension
    //! \p dimension for the values \p input.
    double evaluate1D(double input, int dimension) const;

    //! Get the constant factor of the ANOVA decomposition of the Gaussian process.
    double anovaConstantFactor() const;

    //! Get the total variance of the Gaussian process in the search bounding box
    //! using ANOVA decomposition.
    double anovaTotalVariance() const;

    //! Get the coefficiet of variation of the Gaussian process in the search
    //! bounding box using ANOVA decomposition.
    double anovaTotalCoefficientOfVariation();

    //! Get the main effect of the parameter \p dimension in the Gaussian process
    //! using ANOVA decomposition.
    double anovaMainEffect(int dimension) const;

    //! Get the vector of main effects as an absolute value and as a fraction
    //! of the total variance.
    TDoubleDoublePrVec anovaMainEffects() const;

    //! Set kernel \p parameters explicitly.
    void kernelParameters(const TVector& parameters);

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
    //! gradient collapsing to zero in any direction.
    static const double MINIMUM_KERNEL_COORDINATE_DISTANCE_SCALE;

private:
    //! \name ANOVA
    //@{
    double evaluate(const TVector& Kinvf, const TVector& input) const;
    double evaluate1D(const TVector& Kinvf, double input, int dimension) const;
    double anovaConstantFactor(const TVector& Kinvf) const;
    double anovaTotalVariance(const TVector& Kinvf) const;
    double anovaMainEffect(const TVector& Kinvf, int dimension) const;
    //@}

    void precondition();
    TVector function() const;
    double meanErrorVariance() const;
    TMatrix dKerneld(const TVector& a, int k) const;
    TMatrix kernel(const TVector& a, double v) const;
    TVectorDoublePr kernelCovariates(const TVector& a, const TVector& x, double vx) const;
    double kernel(const TVector& a, const TVector& x, const TVector& y) const;
    TVector kinvf() const;
    TVector transformTo01(const TVector& x) const;
    double dissimilarity(const TVector& x) const;
    void checkRestoredInvariants() const;

private:
    CPRNG::CXorOShiro128Plus m_Rng;
    std::size_t m_Restarts;
    double m_RangeShift{0.0};
    double m_RangeScale{1.0};
    double m_ExplainedErrorVariance{0.0};
    TVector m_MinBoundary;
    TVector m_MaxBoundary;
    TVectorDoublePrVec m_FunctionMeanValues;
    TDoubleVec m_ErrorVariances;
    TVector m_KernelParameters;
    TVector m_MinimumKernelCoordinateDistanceScale;
};
}
}
}

#endif // INCLUDED_ml_maths_common_CBayesianOptimisation_h
