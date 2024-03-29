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

#ifndef INCLUDED_ml_maths_common_CNormalMeanPrecConjugate_h
#define INCLUDED_ml_maths_common_CNormalMeanPrecConjugate_h

#include <core/CMemoryUsage.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CEqualWithTolerance.h>
#include <maths/common/CPrior.h>
#include <maths/common/ImportExport.h>

#include <string>
#include <vector>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {
namespace common {
struct SDistributionRestoreParams;

//! \brief A conjugate prior distribution for a normal variable.
//!
//! DESCRIPTION:\n
//! Our normal variable Y is assumed to have unknown mean and precision.
//!
//! The parameters of Y are modeled as a joint gamma-normal distribution (which
//! is the conjugate prior for a normal with unknown mean and precision).
//!
//! All prior distributions implement a process whereby they relax back to the
//! non-informative over some period without update (see propagateForwardsByTime).
//! The rate at which they relax is controlled by the decay factor supplied to the
//! constructor.
//!
//! IMPLEMENTATION DECISIONS:\n
//! All priors are derived from CPrior which defines the contract that is used
//! by composite priors. This allows us to select the most appropriate model for
//! the data when using one-of-n composition (see COneOfNPrior) or model data with
//! multiple modes when using multi-modal composition (see CMultimodalPrior).
//! From a design point of view this is the composite pattern.
class MATHS_COMMON_EXPORT CNormalMeanPrecConjugate : public CPrior {
public:
    //! See core::CMemory.
    static constexpr bool dynamicSizeAlwaysZero() { return true; }

    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TEqualWithTolerance = CEqualWithTolerance<double>;

    //! Lift the overloads of addSamples into scope.
    using CPrior::addSamples;
    //! Lift the overloads of print into scope.
    using CPrior::print;

public:
    //! \name Life-Cycle
    //@{
    //! \param[in] dataType The type of data being modeled (see maths_t::EDataType
    //! for details).
    //! \param[in] gaussianMean The mean of the normal component of the prior.
    //! \param[in] gaussianPrecision The precision of the normal component of
    //! the prior.
    //! \param[in] gammaShape The shape parameter of the gamma component of the
    //! prior.
    //! \param[in] gammaRate The rate parameter of the gamma component of the
    //! prior.
    //! \param[in] decayRate The rate at which to revert to non-informative.
    CNormalMeanPrecConjugate(maths_t::EDataType dataType,
                             double gaussianMean,
                             double gaussianPrecision,
                             double gammaShape,
                             double gammaRate,
                             double decayRate = 0.0);

    //! Construct from sample central moments.
    CNormalMeanPrecConjugate(maths_t::EDataType dataType,
                             const TMeanVarAccumulator& moments,
                             double decayRate = 0.0);

    //! Construct from part of a state document.
    CNormalMeanPrecConjugate(const SDistributionRestoreParams& params,
                             core::CStateRestoreTraverser& traverser);

    // Default copy constructor and assignment operator work.

    //! Create an instance of a non-informative prior.
    //!
    //! \param[in] dataType The type of data being modeled (see maths_t::EDataType
    //! for details).
    //! \param[in] decayRate The rate at which to revert to the non-informative prior.
    //! \return A non-informative prior.
    static CNormalMeanPrecConjugate nonInformativePrior(maths_t::EDataType dataType,
                                                        double decayRate = 0.0);
    //@}

    //! Reset the prior based on the sample central moments.
    void reset(maths_t::EDataType dataType,
               const TMeanVarAccumulator& moments,
               double decayRate = 0.0);

    //! \name Prior Contract
    //@{
    //! Get the type of this prior.
    EPrior type() const override;

    //! Create a copy of the prior.
    //!
    //! \return A pointer to a newly allocated clone of this prior.
    //! \warning The caller owns the object returned.
    CNormalMeanPrecConjugate* clone() const override;

    //! Reset the prior to non-informative.
    void setToNonInformative(double offset = 0.0, double decayRate = 0.0) override;

    //! Returns false.
    bool needsOffset() const override;

    //! No-op.
    double adjustOffset(const TDouble1Vec& samples, const TDoubleWeightsAry1Vec& weights) override;

    //! Returns zero.
    double offset() const override;

    //! Update the prior with a collection of independent samples from
    //! the normal variable.
    //!
    //! \param[in] samples A collection of samples of the variable.
    //! \param[in] weights The weights of each sample in \p samples.
    void addSamples(const TDouble1Vec& samples, const TDoubleWeightsAry1Vec& weights) override;

    //! Propagate the prior density function forwards by \p time.
    //!
    //! The prior distribution relaxes back to non-informative at a rate
    //! controlled by the decay rate parameter (optionally supplied to the
    //! constructor).
    //!
    //! \param[in] time The time increment to apply.
    //! \note \p time must be non negative.
    void propagateForwardsByTime(double time) override;

    //! Get the support for the marginal likelihood function.
    TDoubleDoublePr marginalLikelihoodSupport() const override;

    //! Get the mean of the marginal likelihood function.
    double marginalLikelihoodMean() const override;

    //! Get the mode of the marginal likelihood function.
    double marginalLikelihoodMode(const TDoubleWeightsAry& weights = TWeights::UNIT) const override;

    //! Get the variance of the marginal likelihood.
    double marginalLikelihoodVariance(const TDoubleWeightsAry& weights = TWeights::UNIT) const override;

    //! Get the \p percentage symmetric confidence interval for the marginal
    //! likelihood function, i.e. the values \f$a\f$ and \f$b\f$ such that:
    //! <pre class="fragment">
    //!   \f$P([a,m]) = P([m,b]) = p / 100 / 2\f$
    //! </pre>
    //!
    //! where \f$m\f$ is the median of the distribution and \f$p\f$ is the
    //! the percentage of interest \p percentage.
    //!
    //! \param[in] percentage The percentage of interest.
    //! \param[in] weights Optional variance scale weights.
    //! \note \p percentage should be in the range [0.0, 100.0).
    TDoubleDoublePr marginalLikelihoodConfidenceInterval(
        double percentage,
        const TDoubleWeightsAry& weights = TWeights::UNIT) const override;

    //! Compute the log marginal likelihood function at \p samples integrating
    //! over the prior density function for the normal mean and precision.
    //!
    //! \param[in] samples A collection of samples of the variable.
    //! \param[in] weights The weights of each sample in \p samples.
    //! \param[out] result Filled in with the joint likelihood of \p samples.
    //! \note The samples are assumed to be independent and identically
    //! distributed.
    maths_t::EFloatingPointErrorStatus
    jointLogMarginalLikelihood(const TDouble1Vec& samples,
                               const TDoubleWeightsAry1Vec& weights,
                               double& result) const override;

    //! Sample the marginal likelihood function.
    //!
    //! \see CPrior::sampleMarginalLikelihood() for a detailed description.
    //!
    //! \param[in] numberSamples The number of samples required.
    //! \param[out] samples Filled in with samples from the prior.
    //! \note \p numberSamples is truncated to the number of samples received.
    void sampleMarginalLikelihood(std::size_t numberSamples, TDouble1Vec& samples) const override;

    //! Compute minus the log of the joint c.d.f. of the marginal likelihood
    //! at \p samples.
    //!
    //! \param[in] samples The samples of interest.
    //! \param[in] weights The weights of each sample in \p samples. For
    //! the count variance scale weight style the weight is interpreted as
    //! a scale of the likelihood variance. So we interpret the likelihood
    //! function as:\n
    //! <pre class="fragment">
    //!   \f$\displaystyle f(x_i) = \sqrt{\frac{p}{2\pi\gamma_i}} e^{-\frac{p}{2\gamma_i}(x_i - m)^2}\f$
    //! </pre>
    //! Here, \f$m\f$ is the mean and \f$p\f$ are the mean and precision for
    //! which this is the prior.
    //! \param[out] lowerBound Filled in with \f$-\log(\prod_i{F(x_i)})\f$
    //! where \f$F(.)\f$ is the c.d.f. and \f$\{x_i\}\f$ are the samples.
    //! \param[out] upperBound Equal to \p lowerBound.
    //! \note The samples are assumed to be independent and identically
    //! distributed.
    //! \note The samples are assumed to be independent.
    //! \warning The variance scales \f$\gamma_i\f$ must be in the range
    //! \f$(0,\infty)\f$, i.e. a value of zero is not well defined and
    //! a value of infinity is not well handled. (Very large values are
    //! handled though.)
    bool minusLogJointCdf(const TDouble1Vec& samples,
                          const TDoubleWeightsAry1Vec& weights,
                          double& lowerBound,
                          double& upperBound) const override;

    //! Compute minus the log of the one minus the joint c.d.f. of the
    //! marginal likelihood at \p samples without losing precision due to
    //! cancellation errors at one, i.e. the smallest non-zero value this
    //! can return is the minimum double rather than epsilon.
    //!
    //! \see minusLogJointCdf for more details.
    bool minusLogJointCdfComplement(const TDouble1Vec& samples,
                                    const TDoubleWeightsAry1Vec& weights,
                                    double& lowerBound,
                                    double& upperBound) const override;

    //! Compute the probability of a less likely, i.e. lower likelihood,
    //! collection of independent samples from the variable.
    //!
    //! \param[in] calculation The style of the probability calculation
    //! (see model_t::EProbabilityCalculation for details).
    //! \param[in] samples The samples of interest.
    //! \param[in] weights The weights. See minusLogJointCdf for discussion.
    //! \param[out] lowerBound Filled in with the probability of the set
    //! for which the joint marginal likelihood is less than that of
    //! \p samples (subject to the measure \p calculation).
    //! \param[out] upperBound Equal to \p lowerBound.
    //! \param[out] tail The tail that (left or right) that all the
    //! samples are in or neither.
    //! \note The samples are assumed to be independent.
    //! \warning The variance scales must be in the range \f$(0,\infty)\f$,
    //! i.e. a value of zero is not well defined and a value of infinity
    //! is not well handled. (Very large values are handled though.)
    bool probabilityOfLessLikelySamples(maths_t::EProbabilityCalculation calculation,
                                        const TDouble1Vec& samples,
                                        const TDoubleWeightsAry1Vec& weights,
                                        double& lowerBound,
                                        double& upperBound,
                                        maths_t::ETail& tail) const override;

    //! Check if this is a non-informative prior.
    bool isNonInformative() const override;

    //! Get a human readable description of the prior.
    //!
    //! \param[in] indent The indent to use at the start of new lines.
    //! \param[in,out] result Filled in with the description.
    void print(const std::string& indent, std::string& result) const override;

    //! Print the prior density function in a specified format.
    //!
    //! \see CPrior::printJointDensityFunction for details.
    std::string printJointDensityFunction() const override;

    //! Get a checksum for this object.
    std::uint64_t checksum(std::uint64_t seed = 0) const override;

    //! Get the memory used by this component
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const override;

    //! Get the memory used by this component
    std::size_t memoryUsage() const override;

    //! Get the static size of this object - used for virtual hierarchies
    std::size_t staticSize() const override;

    //! Persist state by passing information to the supplied inserter
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const override;
    //@}

    //! The current expected mean for the variable.
    double mean() const;

    //! The current expected precision for the variable.
    double precision() const;

    //! \name Test Functions
    //@{
    //! Compute the specified percentage confidence interval for the variable
    //! mean.
    TDoubleDoublePr confidenceIntervalMean(double percentage) const;

    //! Compute the specified percentage confidence interval for the variable
    //! precision.
    TDoubleDoublePr confidenceIntervalPrecision(double percentage) const;

    //! Check if two priors are equal to the specified tolerance.
    bool equalTolerance(const CNormalMeanPrecConjugate& rhs,
                        const TEqualWithTolerance& equal) const;
    //@}

private:
    //! Generate statistics - mean and standard deviation - that are useful in providing a description of this prior
    //! \return A pair of strings containing representations of the marginal likelihood mean and standard deviation
    CPrior::TStrStrPr doPrintMarginalLikelihoodStatistics() const override;

    //! Read parameters from \p traverser.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Check that the state is valid.
    bool isBad() const;

    //! Full debug dump of the state of this prior.
    std::string debug() const override;

private:
    //! The mean parameter of a non-informative prior.
    static const double NON_INFORMATIVE_MEAN;

    //! The precision parameter of a non-informative prior.
    static const double NON_INFORMATIVE_PRECISION;

    //! The shape parameter of a non-informative prior.
    static const double NON_INFORMATIVE_SHAPE;

    //! The rate parameter of a non-informative prior.
    static const double NON_INFORMATIVE_RATE;

private:
    //! The mean of the prior conditional distribution for the mean of the
    //! normal variable (conditioned on its precision).
    CFloatStorage m_GaussianMean;

    //! The precision of the prior conditional distribution for the mean
    //! of the normal variable (conditioned on its precision).
    CFloatStorage m_GaussianPrecision;

    //! The shape of the marginal gamma distribution for the precision of the
    //! normal variable.
    CFloatStorage m_GammaShape;

    //! The rate of the marginal gamma distribution for the precision of the
    //! normal variable.
    double m_GammaRate;
};
}
}
}

#endif // INCLUDED_ml_maths_common_CNormalMeanPrecConjugate_h
