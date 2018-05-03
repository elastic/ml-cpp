/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CMultivariateConstantPrior_h
#define INCLUDED_ml_maths_CMultivariateConstantPrior_h

#include <core/CMemory.h>

#include <maths/CMultivariatePrior.h>

#include <maths/ImportExport.h>

#include <boost/optional.hpp>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {

//! \brief A very lightweight prior for representing data for which
//! expect a single value.
//!
//! DESCRIPTION:\n
//! This implements the CPrior interface for a "random" process which
//! only ever takes a single value. This is useful for modeling data
//! features such as the value of an indicator function in a consistent
//! manner to all other types of data.
class MATHS_EXPORT CMultivariateConstantPrior : public CMultivariatePrior {
public:
    using TOptionalDouble10Vec = boost::optional<TDouble10Vec>;

    // Lift all overloads of into scope.
    //{
    using CMultivariatePrior::addSamples;
    using CMultivariatePrior::print;
    //}

public:
    //! \name Life-Cycle
    //@{
    CMultivariateConstantPrior(std::size_t dimension,
                               const TOptionalDouble10Vec& constant = TOptionalDouble10Vec());

    //! Construct by traversing a state document.
    CMultivariateConstantPrior(std::size_t dimension, core::CStateRestoreTraverser& traverser);
    //@}

    //! \name Prior Contract
    //@{
    //! Create a copy of the prior.
    //!
    //! \warning Caller owns returned object.
    virtual CMultivariateConstantPrior* clone() const;

    //! Get the dimension of the prior.
    virtual std::size_t dimension() const;

    //! Reset the prior to non-informative.
    virtual void setToNonInformative(double offset = 0.0, double decayRate = 0.0);

    //! No-op.
    virtual void adjustOffset(const TDouble10Vec1Vec& samples,
                              const TDouble10VecWeightsAry1Vec& weights);

    //! Set the constant if it hasn't been set.
    virtual void addSamples(const TDouble10Vec1Vec& samples,
                            const TDouble10VecWeightsAry1Vec& weights);

    //! No-op.
    virtual void propagateForwardsByTime(double time);

    //! Get the corresponding constant univariate prior.
    virtual TUnivariatePriorPtrDoublePr
    univariate(const TSize10Vec& marginalize, const TSizeDoublePr10Vec& condition) const;

    //! Compute the bivariate const bivariate prior.
    virtual TPriorPtrDoublePr bivariate(const TSize10Vec& marginalize,
                                        const TSizeDoublePr10Vec& condition) const;

    //! Get the support for the marginal likelihood function.
    virtual TDouble10VecDouble10VecPr marginalLikelihoodSupport() const;

    //! Returns constant or zero if unset (by equidistribution).
    virtual TDouble10Vec marginalLikelihoodMean() const;

    //! Returns constant or zero if unset (by equidistribution).
    virtual TDouble10Vec marginalLikelihoodMode(const TDouble10VecWeightsAry& weights) const;

    //! Get the covariance matrix of the marginal likelihood.
    virtual TDouble10Vec10Vec marginalLikelihoodCovariance() const;

    //! Get the diagonal of the covariance matrix of the marginal likelihood.
    virtual TDouble10Vec marginalLikelihoodVariances() const;

    //! Returns a large value if all samples are equal to the constant
    //! and zero otherwise.
    virtual maths_t::EFloatingPointErrorStatus
    jointLogMarginalLikelihood(const TDouble10Vec1Vec& samples,
                               const TDouble10VecWeightsAry1Vec& weights,
                               double& result) const;

    //! Get \p numberSamples times the constant.
    virtual void sampleMarginalLikelihood(std::size_t numberSamples,
                                          TDouble10Vec1Vec& samples) const;

    //! Check if this is a non-informative prior.
    bool isNonInformative() const;

    //! Get a human readable description of the prior.
    //!
    //! \param[in] separator String used to separate priors.
    //! \param[in,out] result Filled in with the description.
    virtual void print(const std::string& separator, std::string& result) const;

    //! Get a checksum for this object.
    virtual uint64_t checksum(uint64_t seed = 0) const;

    //! Get the memory used by this component
    virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

    //! Get the memory used by this component
    virtual std::size_t memoryUsage() const;

    //! Get the static size of this object - used for virtual hierarchies
    virtual std::size_t staticSize() const;

    //! Persist state by passing information to the supplied inserter
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Get the tag name for this prior.
    virtual std::string persistenceTag() const;
    //@}

    //! Get the constant value.
    const TOptionalDouble10Vec& constant() const;

private:
    //! Create by traversing a state document.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

private:
    //! The data dimension.
    std::size_t m_Dimension;

    //! The constant value.
    TOptionalDouble10Vec m_Constant;
};
}
}

#endif // INCLUDED_ml_maths_CMultivariateConstantPrior_h
