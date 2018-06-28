/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CMultimodalPrior_h
#define INCLUDED_ml_maths_CMultimodalPrior_h

#include <core/CMemory.h>

#include <maths/CBasicStatistics.h>
#include <maths/CClusterer.h>
#include <maths/CMultimodalPriorMode.h>
#include <maths/CPrior.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {

//! \brief Implementation for a multimodal prior distribution.
//!
//! DESCRIPTION:\n
//! This is used to model a variable for which we expect there to be distinct
//! modes, which can be modeled accurately by any of our basic single mode
//! distributions.
//!
//! A separate mechanism is provided to identify the clusters in the data
//! corresponding to distinct modes so that different methods for identifying
//! clusters can be used.
//!
//! All prior distributions implement a process whereby they relax back to the
//! non-informative over some period without update (see propagateForwardsByTime).
//! The rate at which they relax is controlled by the decay factor supplied to the
//! constructor.
//!
//! IMPLEMENTATION DECISIONS:\n
//! All priors are derived from CPrior which defines the contract that is used
//! by composite priors. This allows us to select the most appropriate model for
//! the data when using one-of-n composition (see COneOfNPrior). From a design
//! point of view this is the composite pattern.
class MATHS_EXPORT CMultimodalPrior : public CPrior {
public:
    using TClustererPtr = std::unique_ptr<CClusterer1d>;
    using TPriorPtr = std::unique_ptr<CPrior>;
    using TPriorPtrVec = std::vector<TPriorPtr>;
    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TMeanVarAccumulatorVec = std::vector<TMeanVarAccumulator>;

    // Lift all overloads into scope.
    //{
    using CPrior::addSamples;
    using CPrior::dataType;
    using CPrior::decayRate;
    using CPrior::print;
    //}

public:
    //! \name Life-Cycle
    //@{
    //! Create a new (empty) multimodal prior.
    CMultimodalPrior(maths_t::EDataType dataType,
                     const CClusterer1d& clusterer,
                     const CPrior& seedPrior,
                     double decayRate = 0.0);

    //! Create a mixture of normals.
    CMultimodalPrior(maths_t::EDataType dataType,
                     const TMeanVarAccumulatorVec& moments,
                     double decayRate = 0.0);

    //! Create from a collection of weights and priors.
    //!
    //! \note The priors are moved into place clearing the values in \p priors.
    //! \note This constructor doesn't support subsequent update of the prior.
    CMultimodalPrior(maths_t::EDataType dataType, double decayRate, TPriorPtrVec& priors);

    //! Construct from part of a state document.
    CMultimodalPrior(const SDistributionRestoreParams& params,
                     core::CStateRestoreTraverser& traverser);

    //! Implements value semantics for copy construction.
    CMultimodalPrior(const CMultimodalPrior& other);

    //! Implements value semantics for assignment.
    //!
    //! \param[in] rhs The mixed model to copy.
    //! \return The newly copied model.
    CMultimodalPrior& operator=(const CMultimodalPrior& rhs);

    //! An efficient swap of the contents of this and \p other.
    void swap(CMultimodalPrior& other);
    //@}

    //! \name Prior Contract.
    //@{
    //! Get the type of this prior.
    virtual EPrior type() const;

    //! Create a copy of the prior.
    //!
    //! \return A pointer to a newly allocated clone of this prior.
    //! \warning The caller owns the object returned.
    virtual CMultimodalPrior* clone() const;

    //! Set the data type.
    virtual void dataType(maths_t::EDataType value);

    //! Set the rate at which the prior returns to non-informative.
    virtual void decayRate(double value);

    //! Reset the prior to non-informative.
    virtual void setToNonInformative(double offset = 0.0, double decayRate = 0.0);

    //! Check if any of the modes needs an offset to be applied.
    virtual bool needsOffset() const;

    //! Forward the offset to the mode priors.
    //!
    //! \return The penalty to apply in model selection.
    virtual double adjustOffset(const TDouble1Vec& samples,
                                const TDoubleWeightsAry1Vec& weights);

    //! Get the current offset.
    virtual double offset() const;

    //! Update the prior with a collection of independent samples from
    //! the variable.
    //!
    //! \param[in] samples A collection of samples of the variable.
    //! \param[in] weights The weights of each sample in \p samples.
    virtual void addSamples(const TDouble1Vec& samples, const TDoubleWeightsAry1Vec& weights);

    //! Propagate the prior density function forwards by \p time.
    //!
    //! The prior distribution relaxes back to non-informative at a rate
    //! controlled by the decay rate parameter (optionally supplied to the
    //! constructor).
    //!
    //! \param[in] time The time increment to apply.
    virtual void propagateForwardsByTime(double time);

    //! Get the support for the marginal likelihood function.
    virtual TDoubleDoublePr marginalLikelihoodSupport() const;

    //! Get the mean of the marginal likelihood function.
    virtual double marginalLikelihoodMean() const;

    //! Get the nearest mean of the multimodal prior marginal likelihood.
    virtual double nearestMarginalLikelihoodMean(double value) const;

    //! Get the mode of the marginal likelihood function.
    virtual double marginalLikelihoodMode(const TDoubleWeightsAry& weights = TWeights::UNIT) const;

    //! Get the local maxima of the marginal likelihood function.
    virtual TDouble1Vec
    marginalLikelihoodModes(const TDoubleWeightsAry& weights = TWeights::UNIT) const;

    //! Get the variance of the marginal likelihood.
    virtual double
    marginalLikelihoodVariance(const TDoubleWeightsAry& weights = TWeights::UNIT) const;

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
    virtual TDoubleDoublePr
    marginalLikelihoodConfidenceInterval(double percentage,
                                         const TDoubleWeightsAry& weights = TWeights::UNIT) const;

    //! Compute the log marginal likelihood function at \p samples integrating
    //! over the prior density function for the mode parameters and summing
    //! over modes.
    //!
    //! \param[in] samples A collection of samples of the variable.
    //! \param[in] weights The weights of each sample in \p samples.
    //! \param[out] result Filled in with the joint likelihood of \p samples.
    //! \note The samples are assumed to be independent and identically
    //! distributed.
    virtual maths_t::EFloatingPointErrorStatus
    jointLogMarginalLikelihood(const TDouble1Vec& samples,
                               const TDoubleWeightsAry1Vec& weights,
                               double& result) const;

    //! Sample the marginal likelihood function.
    //!
    //! \see CPrior::sampleMarginalLikelihood() for a detailed description.
    //!
    //! \param[in] numberSamples The number of samples required.
    //! \param[out] samples Filled in with samples from the prior.
    //! \note \p numberSamples is truncated to the number of samples received.
    virtual void sampleMarginalLikelihood(std::size_t numberSamples, TDouble1Vec& samples) const;

    //! Compute minus the log of the joint c.d.f. of the marginal likelihood
    //! at \p samples.
    //!
    //! \param[in] samples A collection of samples of the variable.
    //! \param[in] weights The weights of each sample in \p samples.
    //! \param[out] lowerBound Filled in with \f$-\log(\prod_i{F(x_i)})\f$
    //! where \f$F(.)\f$ is the c.d.f. and \f$\{x_i\}\f$ are the samples.
    //! \param[out] upperBound Equal to \p lowerBound.
    //! \note The samples are assumed to be independent.
    //! \warning The variance scales \f$\gamma_i\f$ must be in the range
    //! \f$(0,\infty)\f$, i.e. a value of zero is not well defined and
    //! a value of infinity is not well handled. (Very large values are
    //! handled though.)
    virtual bool minusLogJointCdf(const TDouble1Vec& samples,
                                  const TDoubleWeightsAry1Vec& weights,
                                  double& lowerBound,
                                  double& upperBound) const;

    //! Compute minus the log of the one minus the joint c.d.f. of the
    //! marginal likelihood at \p samples without losing precision due to
    //! cancellation errors at one, i.e. the smallest non-zero value this
    //! can return is the minimum double rather than epsilon.
    //!
    //! \see minusLogJointCdf for more details.
    virtual bool minusLogJointCdfComplement(const TDouble1Vec& samples,
                                            const TDoubleWeightsAry1Vec& weights,
                                            double& lowerBound,
                                            double& upperBound) const;

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
    //! \param[out] tail The tail that (left or right) that all the
    //! samples are in or neither.
    //! \note The samples are assumed to be independent.
    //! \warning The variance scales must be in the range \f$(0,\infty)\f$,
    //! i.e. a value of zero is not well defined and a value of infinity is
    //! not well handled. (Very large values are handled though.)
    virtual bool probabilityOfLessLikelySamples(maths_t::EProbabilityCalculation calculation,
                                                const TDouble1Vec& samples,
                                                const TDoubleWeightsAry1Vec& weights,
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

    //! Get the current number of modes.
    std::size_t numberModes() const;

    //! Check if the class invariants hold.
    bool checkInvariants(const std::string& tag = std::string()) const;

private:
    //! The callback invoked when a mode is split.
    class MATHS_EXPORT CModeSplitCallback {
    public:
        CModeSplitCallback(CMultimodalPrior& prior);
        void operator()(std::size_t sourceIndex,
                        std::size_t leftSplitIndex,
                        std::size_t rightSplitIndex) const;

    private:
        CMultimodalPrior* m_Prior;
    };

    //! The callback invoked when two modes are merged.
    class MATHS_EXPORT CModeMergeCallback {
    public:
        CModeMergeCallback(CMultimodalPrior& prior);
        void operator()(std::size_t leftMergeIndex,
                        std::size_t rightMergeIndex,
                        std::size_t targetIndex) const;

    private:
        CMultimodalPrior* m_Prior;
    };

    using TMode = SMultimodalPriorMode<TPriorPtr>;
    using TModeVec = std::vector<TMode>;

private:
    //! Read parameters from \p traverser.
    bool acceptRestoreTraverser(const SDistributionRestoreParams& params,
                                core::CStateRestoreTraverser& traverser);

    //! We should only use this prior when it has multiple modes.
    virtual bool participatesInModelSelection() const;

    //! Get the number of nuisance parameters in the marginal likelihood.
    //!
    //! This is just number modes - 1 due to the normalization constraint.
    virtual double unmarginalizedParameters() const;

    //! Full debug dump of the mode weights.
    std::string debugWeights() const;

private:
    //! The object which partitions the data into clusters.
    TClustererPtr m_Clusterer;

    //! The object used to initialize new cluster priors.
    TPriorPtr m_SeedPrior;

    //! The modes of the distribution.
    TModeVec m_Modes;
};
}
}

#endif // INCLUDED_ml_maths_CMultimodalPrior_h
