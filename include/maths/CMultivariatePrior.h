/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CMultivariatePrior_h
#define INCLUDED_ml_maths_CMultivariatePrior_h

#include <core/CMemory.h>
#include <core/CSmallVector.h>

#include <maths/Constants.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <boost/shared_ptr.hpp>

#include <cstddef>

namespace ml
{
namespace core
{
class CStatePersistInserter;
}
namespace maths
{
class CPrior;

//! \brief Interface for a multivariate prior distribution function.
//!
//! DESCRIPTION:\n
//! Abstract interface for implementing multivariate prior distribution
//! functions for various classes of likelihood functions.
//!
//! This exists to support a one-of-n prior distribution which comprises
//! a weighted selection of basic likelihood functions and is implemented
//! using the composite pattern.
class MATHS_EXPORT CMultivariatePrior
{
    public:
        using TDouble10Vec = core::CSmallVector<double, 10>;
        using TDouble10Vec1Vec = core::CSmallVector<TDouble10Vec, 1>;
        using TDouble10Vec2Vec = core::CSmallVector<TDouble10Vec, 2>;
        using TDouble10Vec4Vec = core::CSmallVector<TDouble10Vec, 4>;
        using TDouble10Vec10Vec = core::CSmallVector<TDouble10Vec, 10>;
        using TDouble10Vec4Vec1Vec = core::CSmallVector<TDouble10Vec4Vec, 1>;
        using TDouble10VecDouble10VecPr = std::pair<TDouble10Vec, TDouble10Vec>;
        using TSize10Vec = core::CSmallVector<std::size_t, 10>;
        using TSizeDoublePr = std::pair<std::size_t, double>;
        using TSizeDoublePr10Vec = core::CSmallVector<TSizeDoublePr, 10>;
        using TWeightStyleVec = maths_t::TWeightStyleVec;
        using TTail10Vec = core::CSmallVector<maths_t::ETail, 10>;
        using TUnivariatePriorPtr = boost::shared_ptr<CPrior>;
        using TUnivariatePriorPtrDoublePr = std::pair<TUnivariatePriorPtr, double>;
        using TPriorPtr = boost::shared_ptr<CMultivariatePrior>;
        using TPriorPtrDoublePr = std::pair<TPriorPtr, double>;

    public:
        //! The value of the decay rate to fall back to using if the input
        //! value is inappropriate.
        static const double FALLBACK_DECAY_RATE;

        //! \name Persistence Tags
        //!
        //! Tags for the persisting objects in this hierarchy.
        //@{
        static const std::string MULTIMODAL_TAG;
        static const std::string NORMAL_TAG;
        static const std::string ONE_OF_N_TAG;
        static const std::string CONSTANT_TAG;
        //@}

    public:
        //! \name Life-cycle
        //@{
        //! Construct an arbitrarily initialised object, suitable only for
        //! assigning to or swapping with a valid one.
        CMultivariatePrior();

        //! \param[in] dataType The type of data being modeled.
        //! \param[in] decayRate The rate at which the prior returns to non-
        //! informative.
        CMultivariatePrior(maths_t::EDataType dataType, double decayRate);

        virtual ~CMultivariatePrior() = default;

        //! Swap the contents of this prior and \p other.
        void swap(CMultivariatePrior &other);
        //@}

        //! Mark the prior as being used for forecasting.
        void forForecasting();

        //! Check if this prior is being used for forecasting.
        //!
        //! \warning This is an irreversible action so if the prior
        //! is still need it should be copied first.
        bool isForForecasting() const;

        //! Check if the prior is being used to model discrete data.
        bool isDiscrete() const;

        //! Check if the prior is being used to model integer data.
        bool isInteger() const;

        //! Get the data type.
        maths_t::EDataType dataType() const;

        //! Get the rate at which the prior returns to non-informative.
        double decayRate() const;

        //! \name Prior Contract
        //@{
        //! Create a copy of the prior.
        //!
        //! \warning Caller owns returned object.
        virtual CMultivariatePrior *clone() const = 0;

        //! Get the dimension of the prior.
        virtual std::size_t dimension() const = 0;

        //! Set the data type.
        virtual void dataType(maths_t::EDataType value);

        //! Set the rate at which the prior returns to non-informative.
        virtual void decayRate(double value);

        //! Reset the prior to non-informative.
        virtual void setToNonInformative(double offset = 0.0, double decayRate = 0.0) = 0;

        //! For priors with non-negative support this adjusts the offset used
        //! to extend the support to handle negative samples.
        //!
        //! \param[in] weightStyles Controls the interpretation of the weight(s)
        //! that are associated with each sample. See maths_t::ESampleWeightStyle
        //! for more details.
        //! \param[in] samples The samples from which to determine the offset.
        //! \param[in] weights The weights of each sample in \p samples.
        virtual void adjustOffset(const TWeightStyleVec &weightStyles,
                                  const TDouble10Vec1Vec &samples,
                                  const TDouble10Vec4Vec1Vec &weights) = 0;

        //! Update the prior with a collection of independent samples from the
        //! process.
        //!
        //! \param[in] weightStyles Controls the interpretation of the weight(s)
        //! that are associated with each sample. See maths_t::ESampleWeightStyle
        //! for more details.
        //! \param[in] samples A collection of samples of the process.
        //! \param[in] weights The weights of each sample in \p samples.
        virtual void addSamples(const TWeightStyleVec &weightStyles,
                                const TDouble10Vec1Vec &samples,
                                const TDouble10Vec4Vec1Vec &weights) = 0;

        //! Update the prior for the specified elapsed time.
        virtual void propagateForwardsByTime(double time) = 0;

        //! Compute the univariate prior marginalizing over the variables
        //! \p marginalize and conditioning on the variables \p condition.
        //!
        //! \param[in] marginalize The variables to marginalize out.
        //! \param[in] condition The variables to condition on.
        //! \warning The caller owns the result.
        //! \note The variables are passed by the index of their dimension
        //! which must therefore be in range.
        //! \note It is assumed that the variables are in sorted order.
        //! \note The caller must specify dimension - 1 variables between
        //! \p marginalize and \p condition so the resulting distribution
        //! is univariate.
        virtual TUnivariatePriorPtrDoublePr univariate(const TSize10Vec &marginalize,
                                                       const TSizeDoublePr10Vec &condition) const = 0;

        //! Compute the bivariate prior marginalizing over the variables
        //! \p marginalize and conditioning on the variables \p condition.
        //!
        //! \param[in] marginalize The variables to marginalize out.
        //! \param[in] condition The variables to condition on.
        //! \warning The caller owns the result.
        //! \note The variables are passed by the index of their dimension
        //! which must therefore be in range.
        //! \note It is assumed that the variables are in sorted order.
        //! \note The caller must specify dimension - 2 variables between
        //! \p marginalize and \p condition so the resulting distribution
        //! is univariate.
        virtual TPriorPtrDoublePr bivariate(const TSize10Vec &marginalize,
                                            const TSizeDoublePr10Vec &condition) const = 0;

        //! Get the support for the marginal likelihood function.
        virtual TDouble10VecDouble10VecPr marginalLikelihoodSupport() const = 0;

        //! Get the mean of the marginal likelihood function.
        virtual TDouble10Vec marginalLikelihoodMean() const = 0;

        //! Get the nearest mean of the multimodal prior marginal likelihood,
        //! otherwise the marginal likelihood mean.
        virtual TDouble10Vec nearestMarginalLikelihoodMean(const TDouble10Vec &value) const;

        //! Get the mode of the marginal likelihood function.
        virtual TDouble10Vec marginalLikelihoodMode(const TWeightStyleVec &weightStyles,
                                                    const TDouble10Vec4Vec &weights) const = 0;

        //! Get the local maxima of the marginal likelihood function.
        virtual TDouble10Vec1Vec marginalLikelihoodModes(const TWeightStyleVec &weightStyles,
                                                         const TDouble10Vec4Vec &weights) const;

        //! Get the covariance matrix for the marginal likelihood.
        virtual TDouble10Vec10Vec marginalLikelihoodCovariance() const = 0;

        //! Get the diagonal of the covariance matrix for the marginal likelihood.
        virtual TDouble10Vec marginalLikelihoodVariances() const = 0;

        //! Calculate the log marginal likelihood function, integrating over the
        //! prior density function.
        //!
        //! \param[in] weightStyles Controls the interpretation of the weight(s)
        //! that are associated with each sample. See maths_t::ESampleWeightStyle
        //! for more details.
        //! \param[in] samples A collection of samples of the process.
        //! \param[in] weights The weights of each sample in \p samples.
        //! \param[out] result Filled in with the joint likelihood of \p samples.
        virtual maths_t::EFloatingPointErrorStatus
            jointLogMarginalLikelihood(const TWeightStyleVec &weightStyles,
                                       const TDouble10Vec1Vec &samples,
                                       const TDouble10Vec4Vec1Vec &weights,
                                       double &result) const = 0;

        //! Sample the marginal likelihood function.
        //!
        //! The marginal likelihood functions are sampled in quantile intervals
        //! of the generalized cumulative density function, specifically intervals
        //! between contours of constant probability density.
        //!
        //! The idea is to capture a set of samples that accurately and efficiently
        //! represent the information in the prior. Random sampling (although it
        //! has nice asymptotic properties) doesn't fulfill the second requirement:
        //! typically requiring many more samples than sampling in quantile intervals
        //! to capture the same amount of information.
        //!
        //! This is to allow us to transform one prior distribution into another
        //! completely generically and relatively efficiently, by updating the target
        //! prior with these samples. As such the prior needs to maintain a count of
        //! the number of samples to date so that it isn't over sampled.
        //!
        //! \param[in] numberSamples The number of samples required.
        //! \param[out] samples Filled in with samples from the prior.
        //! \note \p numberSamples is truncated to the number of samples received.
        virtual void sampleMarginalLikelihood(std::size_t numberSamples,
                                              TDouble10Vec1Vec &samples) const = 0;

        //! Calculate the joint probability of seeing a lower marginal likelihood
        //! collection of independent samples for each coordinate.
        //!
        //! \param[in] calculation The style of the probability calculation
        //! (see model_t::EProbabilityCalculation for details).
        //! \param[in] weightStyles Controls the interpretation of the weights
        //! that are associated with each sample. See maths_t::ESampleWeightStyle
        //! for more details.
        //! \param[in] samples A collection of samples of the process.
        //! \param[in] weights The weights of each sample in \p samples.
        //! \param[in] coordinates The coordinates for which to compute probabilities.
        //! \param[out] lowerBounds Filled in with lower bounds for the probability
        //! of each coordinate.
        //! \param[out] upperBounds Filled in with upper bounds for the probability
        //! of each coordinate.
        //! \param[out] tail The tail (left or right), of each coordinate, that all
        //! the samples are in or neither.
        //! \note The samples are assumed to be independent.
        //! \warning The variance scales must be in the range \f$(0,\infty)\f$, i.e.
        //! a value of zero is not well defined and a value of infinity is not well
        //! handled. (Very large values are handled though.)
        bool probabilityOfLessLikelySamples(maths_t::EProbabilityCalculation calculation,
                                            const TWeightStyleVec &weightStyles,
                                            const TDouble10Vec1Vec &samples,
                                            const TDouble10Vec4Vec1Vec &weights,
                                            const TSize10Vec &coordinates,
                                            TDouble10Vec2Vec &lowerBounds,
                                            TDouble10Vec2Vec &upperBounds,
                                            TTail10Vec &tail) const;

        //! Calculate the joint probability of seeing a lower likelihood collection
        //! of independent samples from the distribution integrating over the prior
        //! density function.
        //!
        //! \param[in] calculation The style of the probability calculation
        //! (see model_t::EProbabilityCalculation for details).
        //! \param[in] weightStyles Controls the interpretation of the weights
        //! that are associated with each sample. See maths_t::ESampleWeightStyle
        //! for more details.
        //! \param[in] samples A collection of samples of the process.
        //! \param[in] weights The weights of each sample in \p samples.
        //! \param[out] lowerBound Filled in with a lower bound for the probability
        //! of the set for which the joint marginal likelihood is less than
        //! that of \p samples (subject to the measure \p calculation).
        //! \param[out] upperBound Filled in with an upper bound for the
        //! probability of the set for which the joint marginal likelihood is
        //! less than that of \p samples (subject to the measure \p calculation).
        //! \param[out] tail The tail (left or right), of each coordinate, that all
        //! the samples are in or neither.
        //! \note The samples are assumed to be independent.
        //! \warning The variance scales must be in the range \f$(0,\infty)\f$, i.e.
        //! a value of zero is not well defined and a value of infinity is not well
        //! handled. (Very large values are handled though.)
        bool probabilityOfLessLikelySamples(maths_t::EProbabilityCalculation calculation,
                                            const TWeightStyleVec &weightStyles,
                                            const TDouble10Vec1Vec &samples,
                                            const TDouble10Vec4Vec1Vec &weights,
                                            double &lowerBound,
                                            double &upperBound,
                                            TTail10Vec &tail) const;

        //! Check if this is a non-informative prior.
        virtual bool isNonInformative() const = 0;

        //! Get a human readable description of the prior.
        std::string print() const;

        //! Get a human readable description of the prior.
        //!
        //! \param[in] separator String used to separate priors.
        //! \param[in,out] result Filled in with the description.
        virtual void print(const std::string &separator, std::string &result) const = 0;

        //! Print a projection of the marginal likelihood function onto the specified
        //! coordinates.
        //!
        //! The format is as follows:\n
        //! \code{cpp}
        //!   x = [x1 x2 .... xn ];
        //!   y = [y1 y2 .... yn ];
        //!   likelihood = [L(x1, y1) L(x1, y2) ... L(x1, yn)
        //!                 L(x2, y1) L(x2, y2) ... L(x2, yn)
        //!                      .         .            .
        //!                      .         .            .
        //!                      .         .            .
        //!                 L(xn, y1) L(xn, y2) ... L(xn, yn) ];
        //! \endcode
        //!
        //! i.e. domain values are space separated on the first and subsequent line(s)
        //! as appropriate and the density function evaluated at those values are space
        //! separated on the next line and subsequent lines as appropriate.
        std::string printMarginalLikelihoodFunction(std::size_t x, std::size_t y) const;

        //! Get a checksum for this object.
        virtual uint64_t checksum(uint64_t seed = 0) const = 0;

        //! Get the memory used by this component
        virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const = 0;

        //! Get the memory used by this component
        virtual std::size_t memoryUsage() const = 0;

        //! Get the static size of this object - used for virtual hierarchies
        virtual std::size_t staticSize() const = 0;

        //! Get the tag name for this prior.
        virtual std::string persistenceTag() const = 0;

        //! Persist state by passing information to the supplied inserter
        virtual void acceptPersistInserter(core::CStatePersistInserter &inserter) const = 0;
        //@}

        //! Get the margin between the smallest value and the support left
        //! end. Priors with non-negative support, automatically adjust the
        //! offset if a value is seen which is smaller than offset + margin.
        //! This is to avoid the numerical instability caused by adding
        //! values close to zero.
        //!
        //! \note This is overridden by CPriorTestInterface so don't replace
        //! it by a static constant in the calling functions.
        virtual double offsetMargin() const;

        //! Get the number of samples received.
        double numberSamples() const;

        //! Set the number of samples received to \p numberSamples.
        void numberSamples(double numberSamples);

        //! Check if we should use this prior at present.
        virtual bool participatesInModelSelection() const;

        //! Get the number of unmarginalized parameters in the marginal likelihood.
        //!
        //! \note That any parameters over which we explicitly integrate to
        //! compute a marginal likelihood don't need to be counted since we
        //! are interested in the estimating the usual BIC approximation for
        //! \f$int_{\theta}f(x|\theta, M)f(\theta|M)\d\theta\f$
        virtual double unmarginalizedParameters() const;

    protected:
        //! Get the scaled decay rate for use by propagateForwardsByTime.
        double scaledDecayRate() const;

        //! Update the number of samples received to date by adding \p n.
        void addSamples(double n);

        //! Check that the samples and weights are consistent.
        bool check(const TDouble10Vec1Vec &samples,
                   const TDouble10Vec4Vec1Vec &weights) const;

        //! Check that the variables to marginalize out and condition on
        //! are consistent.
        bool check(const TSize10Vec &marginalize,
                   const TSizeDoublePr10Vec &condition) const;

        //! Get the remaining variables.
        void remainingVariables(const TSize10Vec &marginalize,
                                const TSizeDoublePr10Vec &condition,
                                TSize10Vec &results) const;

        //! Get the smallest component of \p x.
        double smallest(const TDouble10Vec &x) const;

    private:
        //! Set to true if this model is being used for forecasting. Note
        //! we don't have any need to persist forecast models so this is
        //! is not persisted.
        bool m_Forecasting;

        //! If this is true then the prior is being used to model discrete
        //! data. Note that this is not persisted and deduced from context.
        maths_t::EDataType m_DataType;

        //! The rate at which the prior returns to non-informative. Note that
        //! this is not persisted.
        double m_DecayRate;

        //! The number of samples with which the prior has been updated.
        double m_NumberSamples;
};

}
}

#endif // INCLUDED_ml_maths_CMultivariatePrior_h
