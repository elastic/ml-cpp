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

#ifndef INCLUDED_ml_maths_CLogNormalMeanPrecConjugate_h
#define INCLUDED_ml_maths_CLogNormalMeanPrecConjugate_h

#include <core/CMemory.h>

#include <maths/CEqualWithTolerance.h>
#include <maths/CPrior.h>
#include <maths/ImportExport.h>

#include <string>
#include <vector>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {
struct SDistributionRestoreParams;

//! \brief A conjugate prior distribution for a log-normal variable.
//!
//! DESCRIPTION:\n
//! Our log-normal variable is \f$e^Y - u\f$, where \f$Y\f$ is a normal R.V.
//! with unknown mean and precision and \f$u\f$ is a fixed constant offset.
//!
//! If u is positive then the samples can be negative. The parameters of \f$Y\f$
//! are modeled as a joint gamma-normal distribution (which is the conjugate prior
//! for a normal with unknown mean and precision).
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
class MATHS_EXPORT CLogNormalMeanPrecConjugate : public CPrior {
public:
    //! See core::CMemory.
    static bool dynamicSizeAlwaysZero(void) { return true; }

    typedef CEqualWithTolerance<double> TEqualWithTolerance;

    //! Lift the overloads of addSamples into scope.
    using CPrior::addSamples;
    //! Lift the overloads of print into scope.
    using CPrior::print;

public:
    //! \name Life-Cycle
    //@{
    //! \param[in] dataType The type of data being modeled (see maths_t::EDataType
    //! for details).
    //! \param[in] offset The offset to apply to the data.
    //! \param[in] gaussianMean The mean of the normal component of the prior.
    //! \param[in] gaussianPrecision The precision of the normal component of
    //! the prior.
    //! \param[in] gammaShape The shape parameter of the gamma component of the
    //! prior.
    //! \param[in] gammaRate The rate parameter of the gamma component of the
    //! prior.
    //! \param[in] decayRate The rate at which to revert to non-informative.
    //! \param[in] offsetMargin The margin between the smallest value and the support
    //! left end.
    CLogNormalMeanPrecConjugate(maths_t::EDataType dataType,
                                double offset,
                                double gaussianMean,
                                double gaussianPrecision,
                                double gammaShape,
                                double gammaRate,
                                double decayRate = 0.0,
                                double offsetMargin = LOG_NORMAL_OFFSET_MARGIN);

    //! Construct from part of a state document.
    CLogNormalMeanPrecConjugate(const SDistributionRestoreParams &params,
                                core::CStateRestoreTraverser &traverser,
                                double offsetMargin = LOG_NORMAL_OFFSET_MARGIN);

    // Default copy constructor and assignment operator work.

    //! Create an instance of a non-informative prior.
    //!
    //! \param[in] dataType The type of data being modeled (see maths_t::EDataType
    //! for details).
    //! \param[in] offset The offset to apply to the data.
    //! \param[in] decayRate The rate at which to revert to the non-informative prior.
    //! \param[in] offsetMargin The margin between the smallest value and the support
    //! left end.
    //! \return A non-informative prior.
    static CLogNormalMeanPrecConjugate
    nonInformativePrior(maths_t::EDataType dataType,
                        double offset = 0.0,
                        double decayRate = 0.0,
                        double offsetMargin = LOG_NORMAL_OFFSET_MARGIN);
    //@}

    //! \name Prior Contract
    //@{
    //! Get the type of this prior.
    virtual EPrior type(void) const;

    //! Create a copy of the prior.
    //!
    //! \return A pointer to a newly allocated clone of this prior.
    //! \warning The caller owns the object returned.
    virtual CLogNormalMeanPrecConjugate *clone(void) const;

    //! Reset the prior to non-informative.
    virtual void setToNonInformative(double offset = 0.0, double decayRate = 0.0);

    //! Get the margin between the smallest value and the support left
    //! end. Priors with non-negative support, automatically adjust the
    //! offset if a value is seen which is smaller than offset + margin.
    virtual double offsetMargin(void) const;

    //! Returns true.
    virtual bool needsOffset(void) const;

    //! Reset m_Offset so the smallest sample is not within some minimum
    //! offset of the support left end. Note that translating the mean of
    //! a log-normal affects its shape, so there is no easy adjustment of
    //! the prior parameters which preserves the distribution after
    //! translation.
    //!
    //! This samples the current marginal likelihood and uses these samples
    //! to reconstruct the prior with adjusted offset.
    //!
    //! \param[in] weightStyles Controls the interpretation of the weight(s)
    //! that are associated with each sample. See maths_t::ESampleWeightStyle
    //! for more details.
    //! \param[in] samples The samples from which to determine the offset.
    //! \param[in] weights The weights of each sample in \p samples.
    //! \return The penalty to apply in model selection.
    virtual double adjustOffset(const TWeightStyleVec &weightStyles,
                                const TDouble1Vec &samples,
                                const TDouble4Vec1Vec &weights);

    //! Get the current offset.
    virtual double offset(void) const;

    //! Update the prior with a collection of independent samples from
    //! the log-normal variable.
    //!
    //! \param[in] weightStyles Controls the interpretation of the weight(s)
    //! that are associated with each sample. See maths_t::ESampleWeightStyle
    //! for more details.
    //! \param[in] samples A collection of samples of the variable.
    //! \param[in] weights The weights of each sample in \p samples.
    virtual void addSamples(const TWeightStyleVec &weightStyles,
                            const TDouble1Vec &samples,
                            const TDouble4Vec1Vec &weights);

    //! Propagate the prior density function forwards by \p time.
    //!
    //! The prior distribution relaxes back to non-informative at a rate
    //! controlled by the decay rate parameter (optionally supplied to the
    //! constructor).
    //!
    //! \param[in] time The time increment to apply.
    //! \note \p time must be non negative.
    virtual void propagateForwardsByTime(double time);

    //! Get the support for the marginal likelihood function.
    virtual TDoubleDoublePr marginalLikelihoodSupport(void) const;

    //! Get the mean of the marginal likelihood function.
    virtual double marginalLikelihoodMean(void) const;

    //! Get the mode of the marginal likelihood function.
    virtual double
    marginalLikelihoodMode(const TWeightStyleVec &weightStyles = TWeights::COUNT_VARIANCE,
                           const TDouble4Vec &weights = TWeights::UNIT) const;

    //! Get the variance of the marginal likelihood.
    virtual double
    marginalLikelihoodVariance(const TWeightStyleVec &weightStyles = TWeights::COUNT_VARIANCE,
                               const TDouble4Vec &weights = TWeights::UNIT) const;

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
    //! \param[in] weightStyles Optional variance scale weight styles.
    //! \param[in] weights Optional variance scale weights.
    //! \note \p percentage should be in the range [0.0, 100.0).
    virtual TDoubleDoublePr marginalLikelihoodConfidenceInterval(
        double percentage,
        const TWeightStyleVec &weightStyles = TWeights::COUNT_VARIANCE,
        const TDouble4Vec &weights = TWeights::UNIT) const;

    //! Compute the log marginal likelihood function at \p samples integrating
    //! over the prior density function for the exponentiated normal mean
    //! and precision.
    //!
    //! \param[in] weightStyles Controls the interpretation of the weight(s)
    //! that are associated with each sample. See maths_t::ESampleWeightStyle
    //! for more details.
    //! \param[in] samples A collection of samples of the variable.
    //! \param[in] weights The weights of each sample in \p samples.
    //! \param[out] result Filled in with the joint likelihood of \p samples.
    //! \note The samples are assumed to be independent and identically
    //! distributed.
    virtual maths_t::EFloatingPointErrorStatus
    jointLogMarginalLikelihood(const TWeightStyleVec &weightStyles,
                               const TDouble1Vec &samples,
                               const TDouble4Vec1Vec &weights,
                               double &result) const;

    //! Sample the marginal likelihood function.
    //!
    //! \see CPrior::sampleMarginalLikelihood() for a detailed description.
    //!
    //! \param[in] numberSamples The number of samples required.
    //! \param[out] samples Filled in with samples from the prior.
    //! \note \p numberSamples is truncated to the number of samples received.
    virtual void sampleMarginalLikelihood(std::size_t numberSamples, TDouble1Vec &samples) const;

    //! Compute minus the log of the joint c.d.f. of the marginal likelihood
    //! at \p samples.
    //!
    //! \param[in] weightStyles Controls the interpretation of the weight(s)
    //! that are associated with each sample. See maths_t::ESampleWeightStyle
    //! for more details.
    //! \param[in] samples The samples of interest.
    //! \param[in] weights The weights of each sample in \p samples. For the
    //! count variance scale weight style the weight is interpreted as a scale
    //! of the likelihood variance. The mean and variance of a log-normal are:
    //! <pre class="fragment">
    //!   \f$\displaystyle e^{m + 1/2p}\f$
    //!   \f$\displaystyle (e^{1/p} - 1)e^{2m + 1/p}\f$
    //! </pre>
    //! Here, \f$m\f$ is the mean and \f$p\f$ is the precision for which
    //! this is the prior. Our assumption implies:
    //! <pre class="fragment">
    //!   \f$\displaystyle e^{m_i' + 1/2p_i'} = e^{m + 1/2p}\f$
    //!   \f$\displaystyle (e^{1/p_i'} - 1)e^{2m_i' + 1/p_i'} = \gamma_i(e^{1/p} - 1)e^{2m + 1/p}\f$
    //! </pre>
    //! where, \f$m_i'\f$ is the mean and \f$p_i'\f$ are the scaled parameters
    //! of the exponentiated normal. We can solve for \f$m_i'\f$ and \f$p_i'\f$
    //! to give:
    //! <pre class="fragment">
    //!   \f$\displaystyle m_i' = m + \frac{1/p - \log(1 + \gamma_i(e^{1/p} - 1))}{2}\f$
    //!   \f$\displaystyle p_i' = \frac{1}{\log(1 + \gamma_i(e^{1/p} - 1))}\f$
    //! </pre>
    //! We then interpret the likelihood function as:\n
    //! <pre class="fragment">
    //!   \f$\displaystyle f(x_i) = \sqrt{\frac{p_i'}{2\pi}}e^{-p_i'/2(\log(x_i) - m_i')^2}\f$
    //! </pre>
    //! \param[out] lowerBound Filled in with \f$-\log(\prod_i{F(x_i)})\f$
    //! where \f$F(.)\f$ is the c.d.f. and \f$\{x_i\}\f$ are the samples.
    //! \param[out] upperBound Equal to \p lowerBound.
    //! \note The samples are assumed to be independent.
    //! \warning The variance scales \f$\gamma_i\f$ must be in the range
    //! \f$(0,\infty)\f$, i.e. a value of zero is not well defined and
    //! a value of infinity is not well handled. The approximations we
    //! make are less good for \f$\gamma_i\f$ a long way from one.
    virtual bool minusLogJointCdf(const TWeightStyleVec &weightStyles,
                                  const TDouble1Vec &samples,
                                  const TDouble4Vec1Vec &weights,
                                  double &lowerBound,
                                  double &upperBound) const;

    //! Compute minus the log of the one minus the joint c.d.f. of the
    //! marginal likelihood at \p samples without losing precision due to
    //! cancellation errors at one, i.e. the smallest non-zero value this
    //! can return is the minimum double rather than epsilon.
    //!
    //! \see minusLogJointCdf for more details.
    virtual bool minusLogJointCdfComplement(const TWeightStyleVec &weightStyles,
                                            const TDouble1Vec &samples,
                                            const TDouble4Vec1Vec &weights,
                                            double &lowerBound,
                                            double &upperBound) const;

    //! Compute the probability of a less likely, i.e. lower likelihood,
    //! collection of independent samples from the variable.
    //!
    //! \param[in] calculation The style of the probability calculation
    //! (see model_t::EProbabilityCalculation for details).
    //! \param[in] weightStyles Controls the interpretation of the weight(s)
    //! that are associated with each sample. See maths_t::ESampleWeightStyle
    //! for more details.
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
    virtual bool probabilityOfLessLikelySamples(maths_t::EProbabilityCalculation calculation,
                                                const TWeightStyleVec &weightStyles,
                                                const TDouble1Vec &samples,
                                                const TDouble4Vec1Vec &weights,
                                                double &lowerBound,
                                                double &upperBound,
                                                maths_t::ETail &tail) const;

    //! Check if this is a non-informative prior.
    virtual bool isNonInformative(void) const;

    //! Get a human readable description of the prior.
    //!
    //! \param[in] indent The indent to use at the start of new lines.
    //! \param[in,out] result Filled in with the description.
    virtual void print(const std::string &indent, std::string &result) const;

    //! Print the prior density function in a specified format.
    //!
    //! \see CPrior::printJointDensityFunction for details.
    virtual std::string printJointDensityFunction(void) const;

    //! Get a checksum for this object.
    virtual uint64_t checksum(uint64_t seed = 0) const;

    //! Get the memory used by this component
    virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

    //! Get the memory used by this component
    virtual std::size_t memoryUsage(void) const;

    //! Get the static size of this object - used for virtual hierarchies
    virtual std::size_t staticSize(void) const;

    //! Persist state by passing information to the supplied inserter
    virtual void acceptPersistInserter(core::CStatePersistInserter &inserter) const;
    //@}

    //! Get the current expected mean for the exponentiated normal.
    //!
    //! \note This is not to be confused with the mean of the variable itself
    //! which is \f$e ^{m + 1 / 2 p}\f$, where \f$m\f$ and \f$p\f$ are the
    //! mean and precision of the exponentiated Guassian, respectively.
    double normalMean(void) const;

    //! Compute the current expected precision for the exponentiated normal.
    //!
    //! \note This is not to be confused with the precision of the variable
    //! itself which is \f$\displaystyle \frac{e ^{-2m - 1 / p}}{e ^{1 / p} - 1}\f$,
    //! where \f$m\f$ and \f$p\f$ are the mean and precision of the exponentiated
    //! Guassian, respectively.
    double normalPrecision(void) const;

    //! \name Test Functions
    //@{
    //! Compute the specified percentage confidence interval for the
    //! exponentiated normal mean.
    //!
    //! \note That this is not to be confused with the mean of variable
    //! itself which is \f$e^{m + 1 / 2 p}\f$, where \f$m\f$ and \f$p\f$
    //! are the mean and precision of the exponentiated Guassian, respectively.
    TDoubleDoublePr confidenceIntervalNormalMean(double percentage) const;

    //! Compute the specified percentage confidence interval for the
    //! exponentiated normal precision.
    //!
    //! \note This is not to be confused with the precision of the variable
    //! itself which is \f$\displaystyle \frac{e ^{-2m - 1 / p}}{e ^{1 / p} - 1}\f$,
    //! where \f$m\f$ and \f$p\f$ are the mean and precision of the exponentiated
    //! Guassian, respectively.
    TDoubleDoublePr confidenceIntervalNormalPrecision(double percentage) const;

    //! Check if two priors are equal to the specified tolerance.
    bool equalTolerance(const CLogNormalMeanPrecConjugate &rhs,
                        const TEqualWithTolerance &equal) const;
    //@}

private:
    //! Read parameters from \p traverser.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

    //! Get the of the marginal likelihood.
    double mean(void) const;

    //! Check that the state is valid.
    bool isBad(void) const;

    //! Full debug dump of the state of this prior.
    virtual std::string debug(void) const;

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
    //! We assume that the data are described by \f$X = e^Y - u\f$, where
    //! \f$u\f$ is a constant and \f$Y\f$ is normally distributed. This
    //! allows us to model data with negative values greater than \f$-u\f$.
    double m_Offset;

    //! The margin between the smallest value and the support left end.
    double m_OffsetMargin;

    //! The mean of the prior conditional distribution for the mean of the
    //! exponentiated normal (conditioned on its precision).
    double m_GaussianMean;

    //! The precision of the prior conditional distribution for the mean
    //! of the exponentiated normal (conditioned on its precision).
    double m_GaussianPrecision;

    //! The shape of the marginal gamma distribution for the precision of the
    //! exponentiated normal.
    double m_GammaShape;

    //! The rate of the marginal gamma distribution for the precision of the
    //! exponentiated normal.
    double m_GammaRate;
};
}
}

#endif// INCLUDED_ml_maths_CLogNormalMeanPrecConjugate_h
