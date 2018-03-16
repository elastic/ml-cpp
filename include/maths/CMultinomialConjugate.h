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

#ifndef INCLUDED_ml_maths_CMultinomialConjugate_h
#define INCLUDED_ml_maths_CMultinomialConjugate_h

#include <core/CMemory.h>

#include <maths/CEqualWithTolerance.h>
#include <maths/CPrior.h>
#include <maths/ImportExport.h>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {
struct SDistributionRestoreParams;

//! \brief A conjugate prior distribution for a multinomial variable.
//!
//! DESCRIPTION:\n
//! The probabilities are modeled by a Dirichlet prior (which is the conjugate
//! prior for a multinomial distribution). This prior has a fixed maximum number
//! of categories, supplied to the constructor, and monitors the number of values
//! in a hold all category once it has "overflowed". In this case, the upper and
//! lower bounds for the probability of less likely samples will be different.
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
class MATHS_EXPORT CMultinomialConjugate : public CPrior {
public:
    typedef CEqualWithTolerance<double> TEqualWithTolerance;

    //! Lift the overloads of addSamples into scope.
    using CPrior::addSamples;
    //! Lift the overloads of print into scope.
    using CPrior::print;

public:
    //! \name Life-Cycle
    //@{
    //! Construct an arbitrarily initialised object, suitable only for
    //! assigning to or swapping with a valid one.
    CMultinomialConjugate(void);

    CMultinomialConjugate(std::size_t maximumNumberOfCategories,
                          const TDoubleVec& categories,
                          const TDoubleVec& concentrationParameters,
                          double decayRate = 0.0);

    //! Construct from part of an state document.
    CMultinomialConjugate(const SDistributionRestoreParams& params, core::CStateRestoreTraverser& traverser);

    // Default copy constructor and assignment operator work.

    //! Efficient swap of the contents of this prior and \p other.
    void swap(CMultinomialConjugate& other);

    //! Create an instance of a non-informative prior.
    //!
    //! \param[in] maximumNumberOfCategories The number of categories in the likelihood function.
    //! \param[in] decayRate The rate at which to revert to the non-informative prior.
    //! \return A non-informative prior.
    static CMultinomialConjugate nonInformativePrior(std::size_t maximumNumberOfCategories, double decayRate = 0.0);
    //@}

    //! \name Prior Contract
    //@{
    //! Get the type of this prior.
    virtual EPrior type(void) const;

    //! Create a copy of the prior.
    //!
    //! \return A pointer to a newly allocated clone of this prior.
    //! \warning The caller owns the object returned.
    virtual CMultinomialConjugate* clone(void) const;

    //! Reset the prior to non-informative.
    virtual void setToNonInformative(double offset = 0.0, double decayRate = 0.0);

    //! Returns false.
    virtual bool needsOffset(void) const;

    //! No-op.
    virtual double
    adjustOffset(const TWeightStyleVec& weightStyles, const TDouble1Vec& samples, const TDouble4Vec1Vec& weights);

    //! Returns zero.
    virtual double offset(void) const;

    //! Update the prior with a collection of independent samples from the
    //! multinomial variable.
    //!
    //! \param[in] weightStyles Controls the interpretation of the weight(s)
    //! that are associated with each sample. See maths_t::ESampleWeightStyle
    //! for more details.
    //! \param[in] samples A collection of samples of the variable.
    //! \param[in] weights The weights of each sample in \p samples.
    virtual void
    addSamples(const TWeightStyleVec& weightStyles, const TDouble1Vec& samples, const TDouble4Vec1Vec& weights);

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
    virtual double marginalLikelihoodMode(const TWeightStyleVec& weightStyles = TWeights::COUNT_VARIANCE,
                                          const TDouble4Vec& weights = TWeights::UNIT) const;

    //! Get the variance of the marginal likelihood.
    virtual double marginalLikelihoodVariance(const TWeightStyleVec& weightStyles = TWeights::COUNT_VARIANCE,
                                              const TDouble4Vec& weights = TWeights::UNIT) const;

    //! Get the \p percentage symmetric confidence interval for the marginal
    //! likelihood function, i.e. the values \f$a\f$ and \f$b\f$ such that:
    //! <pre class="fragment">
    //!   \f$min_{a,b}\{P([a,m]) + P([m,b])\} = p / 100\f$
    //! </pre>
    //!
    //! where \f$m\f$ is the median of the distribution and \f$p\f$ is the
    //! the percentage of interest \p percentage. Note that since the
    //! distribution is discrete we can only approximate the probability.
    //!
    //! \param[in] percentage The percentage of interest.
    //! \param[in] weightStyles Ignored.
    //! \param[in] weights Ignored.
    //! \note \p percentage should be in the range [0.0, 100.0).
    virtual TDoubleDoublePr
    marginalLikelihoodConfidenceInterval(double percentage,
                                         const TWeightStyleVec& weightStyles = TWeights::COUNT_VARIANCE,
                                         const TDouble4Vec& weights = TWeights::UNIT) const;

    //! Compute the log marginal likelihood function at \p samples integrating
    //! over the prior density function for the category probability parameters.
    //!
    //! \param[in] weightStyles Controls the interpretation of the weight(s)
    //! that are associated with each sample. See maths_t::ESampleWeightStyle
    //! for more details.
    //! \param[in] samples A collection of samples of the variable.
    //! \param[in] weights The weights of each sample in \p samples.
    //! \param[out] result Filled in with the log likelihood of \p samples.
    //! Note that if the model has overflowed then this is really a lower
    //! bound, but in this case we want the model to die off gracefully from
    //! the model collection, so this is appropriate.
    //! \note The samples are assumed to be independent and identically
    //! distributed.
    virtual maths_t::EFloatingPointErrorStatus jointLogMarginalLikelihood(const TWeightStyleVec& weightStyles,
                                                                          const TDouble1Vec& samples,
                                                                          const TDouble4Vec1Vec& weights,
                                                                          double& result) const;

    //! Sample the marginal likelihood function.
    //!
    //! This samples each category in proportion to its probability. Since
    //! each category can only be sampled an integer number of times we
    //! find the sampling which minimizes the error from the ideal sampling.
    //!
    //! \param[in] numberSamples The number of samples required.
    //! \param[out] samples Filled in with samples from the prior.
    //! \note \p numberSamples is truncated to the number of samples received.
    virtual void sampleMarginalLikelihood(std::size_t numberSamples, TDouble1Vec& samples) const;

    //! Compute minus the log of the joint cumulative density function
    //! of the marginal likelihood at \p samples.
    //!
    //! \param[in] weightStyles Controls the interpretation of the weights
    //! that are associated with each sample. See maths_t::ESampleWeightStyle
    //! for more details. Note variance scales are ignored.
    //! \param[in] samples The samples of interest.
    //! \param[in] weights The weights of each sample in \p samples.
    //! \param[out] lowerBound If the model has not overflowed this is
    //! filled in with \f$-\log(\prod_i{F(x_i)})\f$ where \f$F(.)\f$ is
    //! the c.d.f. and \f$\{x_i\}\f$ are the samples. Otherwise, it is
    //! filled in with a sharp lower bound.
    //! \param[out] upperBound If the model has not overflowed this is
    //! filled in with \f$-\log(\prod_i{F(x_i)})\f$ where \f$F(.)\f$ is
    //! the c.d.f. and \f$\{x_i\}\f$ are the samples. Otherwise, it is
    //! filled in with a sharp upper bound.
    //! \note The samples are assumed to be independent.
    virtual bool minusLogJointCdf(const TWeightStyleVec& weightStyles,
                                  const TDouble1Vec& samples,
                                  const TDouble4Vec1Vec& weights,
                                  double& lowerBound,
                                  double& upperBound) const;

    //! Compute minus the log of the one minus the joint cumulative density
    //! function of the marginal likelihood at \p samples without losing
    //! precision due to cancellation errors at one, i.e. the smallest
    //! non-zero value this can return is the minimum double rather than
    //! epsilon.
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
    //! \param[in] weightStyles Controls the interpretation of the weights
    //! that are associated with each sample. See maths_t::ESampleWeightStyle
    //! for more details. Note variance scales are ignored.
    //! \param[in] samples The samples of interest.
    //! \param[in] weights The weights. See minusLogJointCdf for discussion.
    //! \param[out] lowerBound If the model has not overflowed this is filled
    //! in with the probability of the set for which the joint marginal
    //! likelihood is less than that of \p samples (subject to the measure
    //! \p calculation). Otherwise, it is filled in with a sharp lower bound.
    //! \param[out] upperBound If the model has not overflowed this is filled
    //! in with the probability of the set for which the joint marginal
    //! likelihood is less than that of \p samples (subject to the measure
    //! \p calculation). Otherwise, it is filled in with an upper bound.
    //! \param[out] tail The tail that (left or right) that all the samples
    //! are in or neither.
    //! \note The samples are assumed to be independent.
    virtual bool probabilityOfLessLikelySamples(maths_t::EProbabilityCalculation calculation,
                                                const TWeightStyleVec& weightStyles,
                                                const TDouble1Vec& samples,
                                                const TDouble4Vec1Vec& weights,
                                                double& lowerBound,
                                                double& upperBound,
                                                maths_t::ETail& tail) const;

    //! Check if this is a non-informative prior.
    virtual bool isNonInformative(void) const;

    //! Get a human readable description of the prior.
    //!
    //! \param[in] indent The indent to use at the start of new lines.
    //! \param[in,out] result Filled in with the description.
    virtual void print(const std::string& indent, std::string& result) const;

    //! Print the marginal likelihood function in a specified format.
    //!
    //! \see CPrior::printMarginalLikelihoodFunction for details.
    virtual std::string printMarginalLikelihoodFunction(double weight = 1.0) const;

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
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const;
    //@}

    //! Remove the categories in \p categoriesToRemove.
    void removeCategories(TDoubleVec categoriesToRemove);

    //! Get the index of \p category in the categories vector if it is a
    //! valid category for this prior.
    //!
    //! \param[in] category The category label.
    //! \param[out] result Set to the index of \p category in categories
    //! if they exist and maximum size_t otherwise.
    bool index(double category, std::size_t& result) const;

    //! Get the categories.
    const TDoubleVec& categories(void) const;

    //! Get the concentrations.
    const TDoubleVec& concentrations(void) const;

    //! Get the concentration for a specified category
    bool concentration(double category, double& result) const;

    //! Get the total concetration for a specified category
    double totalConcentration(void) const;

    //! Get the expected probability of \p category if it exists.
    //!
    //! \note The marginal likelihood function of a single sample is
    //! multinomial with probabilities equal to the expected values of
    //! each probability parameter in the Dirichlet prior.
    bool probability(double category, double& result) const;

    //! Get the expected probabilities for each category.
    //!
    //! \note The marginal likelihood function of a single sample is
    //! multinomial with probabilities equal to the expected values of
    //! each probability parameter in the Dirichlet prior.
    TDoubleVec probabilities(void) const;

    //! Compute upper and lower bounds for the collection of probabilities:
    //! <pre class="fragment">
    //!   \f$P_i = P(\{c : L(c) <= L(c_i)\})\f$
    //! </pre>
    //!
    //! for all categories \f$c_i\f$.
    //! \param[in] calculation The style of the probability calculation (see
    //! CTools::EProbabilityCalculation for details).
    //! \param[out] lowerBounds If the model has not overflowed this is filled
    //! in with the probabilities (subject to the measure \p calculation).
    //! Otherwise, it is filled in with a sharp lower bound.
    //! \param[out] upperBounds If the model has not overflowed this is filled
    //! in with the probability of the set (subject to the measure \p calculation).
    //! Otherwise, it is filled in an upper bound.
    void probabilitiesOfLessLikelyCategories(maths_t::EProbabilityCalculation calculation,
                                             TDoubleVec& lowerBounds,
                                             TDoubleVec& upperBounds) const;

    //! \name Test Functions
    //@{
    //! Compute the specified percentage confidence intervals for the
    //! category probabilities.
    //!
    //! The marginal distribution of the i'th probability is beta distributed.
    //! In particular, the i'th probability marginal density function is:\n
    //! <pre class="fragment">
    //!   \f$\displaystyle f(p_i) = \frac{\Gamma(a_0)}{\Gamma(a_0 - a_i)\Gamma(a_i)}(1 - p_i)^{a_0-a_i-1}p_i^{a_i-1}\f$
    //! </pre>
    //!
    //! where,\n
    //!   \f$\displaystyle a_0 = \sum_i{a_i}\f$,\n
    //!   \f$\{a_i\}\f$ are the Dirichlet prior concentrations.
    TDoubleDoublePrVec confidenceIntervalProbabilities(double percentage) const;

    //! Check if two priors are equal to the specified tolerance.
    bool equalTolerance(const CMultinomialConjugate& rhs, const TEqualWithTolerance& equal) const;
    //@}

private:
    //! Read parameters from \p traverser.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Shrinks vectors so that we don't use more memory than we need.
    //! Typically vector implements a doubling policy when growing the
    //! buffer, which means that the buffers can end up twice as large
    //! as we need. This shrinks the capacity based on the number of
    //! available categories remaining.
    void shrink(void);

private:
    //! The sum of the concentration parameters of a non-informative prior.
    static const double NON_INFORMATIVE_CONCENTRATION;

    //! Set to true if we overflow the permitted number of categories.
    int m_NumberAvailableCategories;

    //! The category values.
    TDoubleVec m_Categories;

    //! The concentration parameters of the Dirichlet prior.
    TDoubleVec m_Concentrations;

    //! The total concentration. Note that if we have observed more
    //! categories than we were permitted this is not equal to the
    //! sum of the concentration parameters.
    double m_TotalConcentration;
};
}
}

#endif // INCLUDED_ml_maths_CMultinomialConjugate_h
