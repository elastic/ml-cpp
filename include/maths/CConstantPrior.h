/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CConstantPrior_h
#define INCLUDED_ml_maths_CConstantPrior_h

#include <core/CMemory.h>

#include <maths/CPrior.h>

#include <maths/ImportExport.h>

#include <boost/optional.hpp>

namespace ml
{
namespace core
{
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths
{

//! \brief A very lightweight prior for representing data for which
//! expect a single value.
//!
//! DESCRIPTION:\n
//! This implements the CPrior interface for a "random" process which
//! only ever takes a single value. This is useful for modeling data
//! features such as the value of an indicator function in a consistent
//! manner to all other types of data.
class MATHS_EXPORT CConstantPrior : public CPrior
{
    public:
        using TOptionalDouble = boost::optional<double>;

        //! Lift the overloads of addSamples into scope.
        using CPrior::addSamples;
        //! Lift the overloads of print into scope.
        using CPrior::print;

    public:
        //! \name Life-Cycle
        //@{
        explicit CConstantPrior(const TOptionalDouble &constant = TOptionalDouble());

        //! Construct by traversing a state document.
        CConstantPrior(core::CStateRestoreTraverser &traverser);
        //@}

        //! \name Prior Contract
        //@{
        //! Get the type of this prior.
        virtual EPrior type() const;

        //! Create a copy of the prior.
        //!
        //! \warning Caller owns returned object.
        virtual CConstantPrior *clone() const;

        //! Reset the prior to non-informative.
        virtual void setToNonInformative(double offset = 0.0, double decayRate = 0.0);

        //! Returns false.
        virtual bool needsOffset() const;

        //! No-op.
        virtual double adjustOffset(const TWeightStyleVec &weightStyle,
                                    const TDouble1Vec &samples,
                                    const TDouble4Vec1Vec &weights);

        //! Returns zero.
        virtual double offset() const;

        //! Set the constant if it hasn't been set.
        virtual void addSamples(const TWeightStyleVec &weightStyle,
                                const TDouble1Vec &samples,
                                const TDouble4Vec1Vec &weights);

        //! No-op.
        virtual void propagateForwardsByTime(double time);

        //! Get the support for the marginal likelihood function.
        virtual TDoubleDoublePr marginalLikelihoodSupport() const;

        //! Returns constant or zero if unset (by equidistribution).
        virtual double marginalLikelihoodMean() const;

        //! Returns constant or zero if unset (by equidistribution).
        virtual double marginalLikelihoodMode(const TWeightStyleVec &weightStyles = TWeights::COUNT_VARIANCE,
                                              const TDouble4Vec &weights = TWeights::UNIT) const;

        //! All confidence intervals are the point [constant, constant].
        virtual TDoubleDoublePr
            marginalLikelihoodConfidenceInterval(double percentage,
                                                 const TWeightStyleVec &weightStyles = TWeights::COUNT_VARIANCE,
                                                 const TDouble4Vec &weights = TWeights::UNIT) const;

        //! Get the variance of the marginal likelihood.
        virtual double marginalLikelihoodVariance(const TWeightStyleVec &weightStyles = TWeights::COUNT_VARIANCE,
                                                  const TDouble4Vec &weights = TWeights::UNIT) const;

        //! Returns a large value if all samples are equal to the constant
        //! and zero otherwise.
        virtual maths_t::EFloatingPointErrorStatus
            jointLogMarginalLikelihood(const TWeightStyleVec &weightStyles,
                                       const TDouble1Vec &samples,
                                       const TDouble4Vec1Vec &weights,
                                       double &result) const;

        //! Get \p numberSamples times the constant.
        virtual void sampleMarginalLikelihood(std::size_t numberSamples,
                                              TDouble1Vec &samples) const;

        //! A large number if any sample is less than the constant and
        //! zero otherwise.
        virtual bool minusLogJointCdf(const TWeightStyleVec &weightStyles,
                                      const TDouble1Vec &samples,
                                      const TDouble4Vec1Vec &weights,
                                      double &lowerBound,
                                      double &upperBound) const;

        //! A large number if any sample is larger than the constant and
        //! zero otherwise.
        virtual bool minusLogJointCdfComplement(const TWeightStyleVec &weightStyles,
                                                const TDouble1Vec &samples,
                                                const TDouble4Vec1Vec &weights,
                                                double &lowerBound,
                                                double &upperBound) const;

        //! Returns one if all samples equal the constant and one otherwise.
        virtual bool probabilityOfLessLikelySamples(maths_t::EProbabilityCalculation calculation,
                                                    const TWeightStyleVec &weightStyles,
                                                    const TDouble1Vec &samples,
                                                    const TDouble4Vec1Vec &weights,
                                                    double &lowerBound,
                                                    double &upperBound,
                                                    maths_t::ETail &tail) const;

        //! Check if this is a non-informative prior.
        bool isNonInformative() const;

        //! Get a human readable description of the prior.
        //!
        //! \param[in] indent The indent to use at the start of new lines.
        //! \param[in,out] result Filled in with the description.
        virtual void print(const std::string &indent, std::string &result) const;

        //! Print the marginal likelihood function.
        virtual std::string printMarginalLikelihoodFunction(double weight = 1.0) const;

        //! Print the prior density function of the parameters.
        virtual std::string printJointDensityFunction() const;

        //! Get a checksum for this object.
        virtual uint64_t checksum(uint64_t seed = 0) const;

        //! Get the memory used by this component
        virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this component
        virtual std::size_t memoryUsage() const;

        //! Get the static size of this object - used for virtual hierarchies
        virtual std::size_t staticSize() const;

        //! Persist state by passing information to the supplied inserter
        virtual void acceptPersistInserter(core::CStatePersistInserter &inserter) const;
        //@}

        //! Get the constant value.
        TOptionalDouble constant() const;

    private:
        //! Create by traversing a state document.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

    private:
        TOptionalDouble m_Constant;
};

}
}

#endif // INCLUDED_ml_maths_CConstantPrior_h
