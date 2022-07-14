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

#ifndef INCLUDED_ml_maths_common_COneOfNPrior_h
#define INCLUDED_ml_maths_common_COneOfNPrior_h

#include <core/CMemoryFwd.h>
#include <core/CNonCopyable.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CModelWeight.h>
#include <maths/common/CPrior.h>
#include <maths/common/ImportExport.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ml {
namespace maths {
namespace common {
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
class MATHS_COMMON_EXPORT COneOfNPrior : public CPrior {
public:
    using TPriorPtr = std::unique_ptr<CPrior>;
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
    EPrior type() const override;

    //! Create a copy of the prior.
    //!
    //! \return A pointer to a newly allocated clone of this model.
    //! \warning The caller owns the object returned.
    COneOfNPrior* clone() const override;

    //! Set the data type.
    void dataType(maths_t::EDataType value) override;

    //! Set the rate at which the prior returns to non-informative.
    void decayRate(double value) override;

    //! Reset the prior to non-informative.
    void setToNonInformative(double offset = 0.0, double decayRate = 0.0) override;

    //! Remove models marked by \p filter.
    void removeModels(CModelFilter& filter) override;

    //! Check if any of the models needs an offset to be applied.
    bool needsOffset() const override;

    //! Forward the offset to the model priors.
    //!
    //! \return The penalty to apply in model selection.
    double adjustOffset(const TDouble1Vec& samples, const TDoubleWeightsAry1Vec& weights) override;

    //! Get the maximum model offset.
    double offset() const override;

    //! Update the model weights using the marginal likelihoods for
    //! the data. The component prior parameters are then updated.
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

    //! Get the weighted mean of the model nearest means.
    double nearestMarginalLikelihoodMean(double value) const override;

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
    //! where \f$m\f$ is the mode of the distribution and \f$p\f$ is the
    //! the percentage of interest \p percentage.
    //!
    //! \param[in] percentage The percentage of interest.
    //! \param[in] weights Optional variance scale weights.
    //! \note \p percentage should be in the range (0.0, 100.0].
    TDoubleDoublePr marginalLikelihoodConfidenceInterval(
        double percentage,
        const TDoubleWeightsAry& weights = TWeights::UNIT) const override;

    //! Compute the log marginal likelihood function at \p samples integrating
    //! over the prior density function for the distribution parameters.
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
    //! This samples each model in proportion to the probability the data
    //! come from that model. Since each model can only be sampled an integer
    //! number of times we find the sampling which minimizes the error from
    //! the ideal sampling.
    //!
    //! \param[in] numberSamples The number of samples required.
    //! \param[out] samples Filled in with samples from the prior.
    //! \note \p numberSamples is truncated to the number of samples received.
    void sampleMarginalLikelihood(std::size_t numberSamples, TDouble1Vec& samples) const override;

private:
    //! The common c.d.f. implementation.
    bool minusLogJointCdfImpl(bool complement,
                              const TDouble1Vec& samples,
                              const TDoubleWeightsAry1Vec& weights,
                              double& lowerBound,
                              double& upperBound) const;

public:
    //! Compute minus the log of the joint c.d.f. of the marginal likelihood
    //! at \p samples.
    //!
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

    //! Debug the memory used by this component.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const override;

    //! Get the memory used by this component.
    std::size_t memoryUsage() const override;

    //! Get the static size of this object - used for virtual hierarchies
    std::size_t staticSize() const override;

    //! Persist state by passing information to the supplied inserter
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const override;
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
    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

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

    //! The moments of the samples added.
    TMeanVarAccumulator m_SampleMoments;
};
}
}
}

#endif // INCLUDED_ml_maths_common_COneOfNPrior_h
