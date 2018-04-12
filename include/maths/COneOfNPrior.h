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

#ifndef INCLUDED_ml_maths_COneOfNPrior_h
#define INCLUDED_ml_maths_COneOfNPrior_h

#include <core/CMemory.h>
#include <core/CNonCopyable.h>

#include <maths/CBasicStatistics.h>
#include <maths/CModelWeight.h>
#include <maths/CPrior.h>
#include <maths/ImportExport.h>

#include <boost/shared_ptr.hpp>

#include <string>
#include <utility>
#include <vector>

namespace ml {
namespace maths {
struct SDistributionRestoreParams;

//! \brief Interface for a prior distribution which assumes data are from one
//! of N models.
//!
//! DESCRIPTION:\n
//! Implements a prior distribution which assumes all data is from one of N
//! models (subsequently referred to as component models).
//!
//! Each component model is assumed to have a prior which implements the CPrior
//! interface. An object of this class thus comprises the individual component
//! prior objects plus a collection of weights, one per component model (see
//! addSamples for details). Each weight is the probability that the sampled
//! data comes from the corresponding component models.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This is basically an implementation of the composite pattern for CPrior
//! in a meaningful sense for the hierarchy. It holds component models by
//! pointer to the base class so that any prior in the hierarchy can be mixed
//! in. All component models are owned by the object (it wouldn't make sense
//! to share them) so this also defines the necessary functions to support
//! value semantics and manage the heap.
class MATHS_EXPORT COneOfNPrior : public CPrior {
public:
    using TPriorPtr = boost::shared_ptr<CPrior>;
    using TPriorPtrVec = std::vector<TPriorPtr>;
    using TPriorCPtrVec = std::vector<const CPrior*>;
    using TDoublePriorPtrPr = std::pair<double, TPriorPtr>;
    using TDoublePriorPtrPrVec = std::vector<TDoublePriorPtrPr>;

    //! Lift all overloads of the dataType into scope.
    using CPrior::dataType;
    //! Lift all overloads of the decayRate into scope.
    using CPrior::decayRate;
    //! Lift the overloads of addSamples into scope.
    using CPrior::addSamples;
    //! Lift the overloads of print into scope.
    using CPrior::print;

public:
    //! \name Life-Cycle
    //@{
    //! Create with a collection of models.
    //!
    //! \param[in] models The simple models which comprise the mixed model.
    //! \param[in] dataType The type of data being modeled (see maths_t::EDataType
    //! for details).
    //! \param[in] decayRate The rate at which to revert to the non-informative prior.
    //! \warning This class takes ownership of \p models.
    COneOfNPrior(const TPriorPtrVec& models, maths_t::EDataType dataType, double decayRate = 0.0);

    //! Create with a weighted collection of models.
    //!
    //! \param[in] models The simple models and their weights which comprise
    //! the mixed model.
    //! \param[in] dataType The type of data being modeled (see maths_t::EDataType
    //! for details).
    //! \param[in] decayRate The rate at which we revert to the non-informative prior.
    //! \warning This class takes ownership of \p models.
    COneOfNPrior(const TDoublePriorPtrPrVec& models,
                 maths_t::EDataType dataType,
                 double decayRate = 0.0);

    //! Construct from part of a state document.
    COneOfNPrior(const SDistributionRestoreParams& params,
                 core::CStateRestoreTraverser& traverser);

    //! Implements value semantics for copy construction.
    COneOfNPrior(const COneOfNPrior& other);

    //! Implements value semantics for assignment.
    //!
    //! \param[in] rhs The mixed model to copy.
    //! \return The newly updated model.
    //! \note That this class has value semantics: this overwrites the current
    //! collection of models.
    COneOfNPrior& operator=(const COneOfNPrior& rhs);

    //! Efficient swap of the contents of this prior and \p other.
    void swap(COneOfNPrior& other);
    //@}

    //! \name Prior Contract
    //@{
    //! Get the type of this prior.
    virtual EPrior type() const;

    //! Create a copy of the prior.
    //!
    //! \return A pointer to a newly allocated clone of this model.
    //! \warning The caller owns the object returned.
    virtual COneOfNPrior* clone() const;

    //! Set the data type.
    virtual void dataType(maths_t::EDataType value);

    //! Set the rate at which the prior returns to non-informative.
    virtual void decayRate(double value);

    //! Reset the prior to non-informative.
    virtual void setToNonInformative(double offset = 0.0, double decayRate = 0.0);

    //! Remove models marked by \p filter.
    virtual void removeModels(CModelFilter& filter);

    //! Check if any of the models needs an offset to be applied.
    virtual bool needsOffset() const;

    //! Forward the offset to the model priors.
    //!
    //! \return The penalty to apply in model selection.
    virtual double adjustOffset(const TWeightStyleVec& weightStyles,
                                const TDouble1Vec& samples,
                                const TDouble4Vec1Vec& weights);

    //! Get the maximum model offset.
    virtual double offset() const;

    //! Update the model weights using the marginal likelihoods for
    //! the data. The component prior parameters are then updated.
    //!
    //! \param[in] weightStyles Controls the interpretation of the weight(s)
    //! that are associated with each sample. See maths_t::ESampleWeightStyle
    //! for more details.
    //! \param[in] samples A collection of samples of the variable.
    //! \param[in] weights The weights of each sample in \p samples.
    virtual void addSamples(const TWeightStyleVec& weightStyles,
                            const TDouble1Vec& samples,
                            const TDouble4Vec1Vec& weights);

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
    virtual TDoubleDoublePr marginalLikelihoodSupport() const;

    //! Get the mean of the marginal likelihood function.
    virtual double marginalLikelihoodMean() const;

    //! Get the weighted mean of the model nearest means.
    virtual double nearestMarginalLikelihoodMean(double value) const;

    //! Get the mode of the marginal likelihood function.
    virtual double
    marginalLikelihoodMode(const TWeightStyleVec& weightStyles = TWeights::COUNT_VARIANCE,
                           const TDouble4Vec& weights = TWeights::UNIT) const;

    //! Get the variance of the marginal likelihood.
    virtual double
    marginalLikelihoodVariance(const TWeightStyleVec& weightStyles = TWeights::COUNT_VARIANCE,
                               const TDouble4Vec& weights = TWeights::UNIT) const;

    //! Get the \p percentage symmetric confidence interval for the marginal
    //! likelihood function, i.e. the values \f$a\f$ and \f$b\f$ such that:
    //! <pre class="fragment">
    //!   \f$P([a,m]) = P([m,b]) = p / 100 / 2\f$
    //! </pre>
    //!
    //! where \f$m\f$ is the mode of the distribution and \f$p\f$ is the
    //! the percentage of interest \p percentage.
    //!
    //! \param[in] percentage The percentage of interest.
    //! \param[in] weightStyles Optional variance scale weight styles.
    //! \param[in] weights Optional variance scale weights.
    //! \note \p percentage should be in the range (0.0, 100.0].
    virtual TDoubleDoublePr marginalLikelihoodConfidenceInterval(
        double percentage,
        const TWeightStyleVec& weightStyles = TWeights::COUNT_VARIANCE,
        const TDouble4Vec& weights = TWeights::UNIT) const;

    //! Compute the log marginal likelihood function at \p samples integrating
    //! over the prior density function for the distribution parameters.
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
    jointLogMarginalLikelihood(const TWeightStyleVec& weightStyles,
                               const TDouble1Vec& samples,
                               const TDouble4Vec1Vec& weights,
                               double& result) const;

    //! Sample the marginal likelihood function.
    //!
    //! This samples each model in proportion to the probability the data
    //! come from that model. Since each model can only be sampled an integer
    //! number of times we find the sampling which minimizes the error from
    //! the ideal sampling.
    //!
    //! \param[in] numberSamples The number of samples required.
    //! \param[out] samples Filled in with samples from the prior.
    //! \note \p numberSamples is truncated to the number of samples received.
    virtual void sampleMarginalLikelihood(std::size_t numberSamples, TDouble1Vec& samples) const;

private:
    //! The common c.d.f. implementation.
    bool minusLogJointCdfImpl(bool complement,
                              const TWeightStyleVec& weightStyles,
                              const TDouble1Vec& samples,
                              const TDouble4Vec1Vec& weights,
                              double& lowerBound,
                              double& upperBound) const;

public:
    //! Compute minus the log of the joint c.d.f. of the marginal likelihood
    //! at \p samples.
    //!
    //! \param[in] weightStyles Controls the interpretation of the weight(s)
    //! that are associated with each sample. See maths_t::ESampleWeightStyle
    //! for more details.
    //! \param[in] samples A collection of samples of the variable.
    //! \param[in] weights The weights of each sample in \p samples.
    //! \param[out] lowerBound Filled in with a lower bound to acceptable
    //! accuracy of \f$-\log(\prod_i{F(x_i)})\f$, where \f$F(.)\f$ is the
    //! c.d.f. and \f$\{x_i\}\f$ are the samples.
    //! \param[out] upperBound Filled in with an upper bound to acceptable
    //! accuracy of \f$-\log(\prod_i{F(x_i)})\f$, where \f$F(.)\f$ is the
    //! c.d.f. and \f$\{x_i\}\f$ are the samples.
    //! \note The samples are assumed to be independent.
    //! \warning The variance scales must be in the range \f$(0,\infty)\f$,
    //! i.e. a value of zero is not well defined and a value of infinity is
    //! not well handled. (Very large values are handled though.)
    virtual bool minusLogJointCdf(const TWeightStyleVec& weightStyles,
                                  const TDouble1Vec& samples,
                                  const TDouble4Vec1Vec& weights,
                                  double& lowerBound,
                                  double& upperBound) const;

    //! Compute minus the log of the one minus the joint c.d.f. of the
    //! marginal likelihood at \p samples without losing precision due to
    //! cancellation errors at one, i.e. the smallest non-zero value this
    //! can return is the minimum double rather than epsilon.
    //!
    //! \see minusLogJointCdf for more details.
    virtual bool minusLogJointCdfComplement(const TWeightStyleVec& weightStyles,
                                            const TDouble1Vec& samples,
                                            const TDouble4Vec1Vec& weights,
                                            double& lowerBound,
                                            double& upperBound) const;

    //! Compute the probability of a less likely, i.e. lower likelihood,
    //! collection of independent samples from the variable.
    //!
    //! \param[in] calculation The style of the probability calculation
    //! (see model_t::EProbabilityCalculation for details).
    //! \param[in] weightStyles Controls the interpretation of the weight(s)
    //! that are associated with each sample. See maths_t::ESampleWeightStyle
    //! for more details.
    //! \param[in] samples A collection of samples of the variable.
    //! \param[in] weights The weights of each sample in \p samples.
    //! \param[out] lowerBound Filled in with the probability of the set
    //! for which the joint marginal likelihood is less than that of
    //! \p samples (subject to the measure \p calculation).
    //! \param[out] upperBound Equal to \p lowerBound.
    //! \param[out] tail The tail that (left or right) that all the samples
    //! are in or neither.
    //! \note The samples are assumed to be independent.
    //! \warning The variance scales must be in the range \f$(0,\infty)\f$,
    //! i.e. a value of zero is not well defined and a value of infinity is
    //! not well handled. (Very large values are handled though.)
    virtual bool probabilityOfLessLikelySamples(maths_t::EProbabilityCalculation calculation,
                                                const TWeightStyleVec& weightStyles,
                                                const TDouble1Vec& samples,
                                                const TDouble4Vec1Vec& weights,
                                                double& lowerBound,
                                                double& upperBound,
                                                maths_t::ETail& tail) const;

    //! Check if this is a non-informative prior.
    virtual bool isNonInformative() const;

    //! Get a human readable description of the prior.
    //!
    //! \param[in] indent The indent to use at the start of new lines.
    //! \param[in,out] result Filled in with the description.
    virtual void print(const std::string& indent, std::string& result) const;

    //! Print the prior density function in a specified format.
    //!
    //! \see CPrior::printJointDensityFunction for details.
    virtual std::string printJointDensityFunction() const;

    //! Get a checksum for this object.
    virtual uint64_t checksum(uint64_t seed = 0) const;

    //! Debug the memory used by this component.
    virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

    //! Get the memory used by this component.
    virtual std::size_t memoryUsage() const;

    //! Get the static size of this object - used for virtual hierarchies
    virtual std::size_t staticSize() const;

    //! Persist state by passing information to the supplied inserter
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const;
    //@}

    //! \name Test Functions
    //@{
    //! Get the current values for the model weights.
    TDoubleVec weights() const;

    //! Get the current values for the log model weights.
    TDoubleVec logWeights() const;

    //! Get the current constituent models.
    TPriorCPtrVec models() const;
    //@}

private:
    using TDoubleSizePr = std::pair<double, std::size_t>;
    using TDoubleSizePr5Vec = core::CSmallVector<TDoubleSizePr, 5>;
    using TWeightPriorPtrPr = std::pair<CModelWeight, TPriorPtr>;
    using TWeightPriorPtrPrVec = std::vector<TWeightPriorPtrPr>;
    using TMaxAccumulator = CBasicStatistics::SMax<double>::TAccumulator;

private:
    //! Read parameters from \p traverser.
    bool acceptRestoreTraverser(const SDistributionRestoreParams& params,
                                core::CStateRestoreTraverser& traverser);

    //! Add a model vector entry reading parameters from \p traverser.
    bool modelAcceptRestoreTraverser(const SDistributionRestoreParams& params,
                                     core::CStateRestoreTraverser& traverser);

    //! Get the normalized model weights.
    TDoubleSizePr5Vec normalizedLogWeights() const;

    //! Get the median of the model means.
    double medianModelMean() const;

    //! Check that the model weights are valid.
    bool badWeights() const;

    //! Full debug dump of the model weights.
    std::string debugWeights() const;

private:
    //! A collection of component models and their probabilities.
    TWeightPriorPtrPrVec m_Models;
};
}
}

#endif // INCLUDED_ml_maths_COneOfNPrior_h
