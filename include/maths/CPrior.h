/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CPrior_h
#define INCLUDED_ml_maths_CPrior_h

#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CNonCopyable.h>
#include <core/CSmallVector.h>

#include <maths/Constants.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace ml
{
namespace core
{
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths
{

//! \brief Interface for a prior distribution function.
//!
//! DESCRIPTION:\n
//! Abstract interface for implementing prior distribution functions for
//! various classes of likelihood functions.
//!
//! This exists to support a one-of-n prior distribution which comprises
//! a weighted selection of basic likelihood functions and is implemented
//! using the composite pattern.
class MATHS_EXPORT CPrior
{
    public:
        using TDoubleVec = std::vector<double>;
        using TDoubleVecVec = std::vector<TDoubleVec>;
        using TDoubleDoublePr = std::pair<double, double>;
        using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
        using TWeightStyleVec = maths_t::TWeightStyleVec;
        using TDouble1Vec = core::CSmallVector<double, 1>;
        using TDouble4Vec = core::CSmallVector<double, 4>;
        using TDouble4Vec1Vec = core::CSmallVector<TDouble4Vec, 1>;
        using TWeights = CConstantWeights;

        //! \brief Data for plotting a series
        struct MATHS_EXPORT SPlot
        {
            TDouble1Vec s_Abscissa;
            TDouble1Vec s_Ordinates;
        };

        //! Enumeration of prior types.
        enum EPrior
        {
            E_Constant    = 0x1,
            E_Gamma       = 0x2,
            E_LogNormal   = 0x4,
            E_Multimodal  = 0x8,
            E_Multinomial = 0x10,
            E_Normal      = 0x20,
            E_OneOfN      = 0x40,
            E_Poisson     = 0x80
        };

        //! \brief Defines a filter for removing models from selection.
        class MATHS_EXPORT CModelFilter
        {
            public:
                CModelFilter();

                //! Mark a model to be removed.
                CModelFilter &remove(EPrior model);

                //! Check of \p model should be removed.
                bool operator()(EPrior model) const;

            private:
                //! A binary representation of the filter.
                int m_Filter;
        };

        //! \brief Wrapper around the jointLogMarginalLikelihood function.
        //!
        //! DESCRIPTION:\n
        //! This adapts the jointLogMarginalLikelihood function for use with
        //! CIntegration.
        class MATHS_EXPORT CLogMarginalLikelihood
        {
            public:
                using result_type = double;

            public:
                CLogMarginalLikelihood(const CPrior &prior,
                                       const TWeightStyleVec &weightStyles = CConstantWeights::COUNT,
                                       const TDouble4Vec1Vec &weights = CConstantWeights::SINGLE_UNIT);

                double operator()(double x) const;
                bool operator()(double x, double &result) const;

            private:
                const CPrior *m_Prior;
                const TWeightStyleVec *m_WeightStyles;
                const TDouble4Vec1Vec *m_Weights;
                //! Avoids creating the vector argument to jointLogMarginalLikelihood
                //! more than once.
                mutable TDouble1Vec m_X;
        };

    public:
        //! The value of the decay rate to fall back to using if the input
        //! value is inappropriate.
        static const double FALLBACK_DECAY_RATE;

    public:
        //! \name Life-Cycle
        //@{
        //! Construct an arbitrarily initialised object, suitable only for
        //! assigning to or swapping with a valid one.
        CPrior();

        //! \param[in] dataType The type of data being modeled.
        //! \param[in] decayRate The rate at which the prior returns to non-informative.
        CPrior(maths_t::EDataType dataType, double decayRate);

        // Default copy constructor and assignment operator work.

        //! Virtual destructor for deletion by base pointer.
        virtual ~CPrior() = default;

        //! Swap the contents of this prior and \p other.
        void swap(CPrior &other);
        //@}

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
        //! Get the type of this prior.
        virtual EPrior type() const = 0;

        //! Create a copy of the prior.
        //!
        //! \warning Caller owns returned object.
        virtual CPrior *clone() const = 0;

        //! Set the data type.
        virtual void dataType(maths_t::EDataType value);

        //! Set the rate at which the prior returns to non-informative.
        virtual void decayRate(double value);

        //! Reset the prior to non-informative.
        virtual void setToNonInformative(double offset = 0.0,
                                         double decayRate = 0.0) = 0;

        //! Remove models marked by \p filter.
        virtual void removeModels(CModelFilter &filter);

        //! Get the margin between the smallest value and the support left
        //! end. Priors with non-negative support, automatically adjust the
        //! offset if a value is seen which is smaller than offset + margin.
        virtual double offsetMargin() const;

        //! Check if the prior needs an offset to be applied.
        virtual bool needsOffset() const = 0;

        //! For priors with non-negative support this adjusts the offset used
        //! to extend the support to handle negative samples.
        //!
        //! \param[in] weightStyles Controls the interpretation of the weight(s)
        //! that are associated with each sample. See maths_t::ESampleWeightStyle
        //! for more details.
        //! \param[in] samples The samples from which to determine the offset.
        //! \param[in] weights The weights of each sample in \p samples.
        //! \return The penalty to apply in model selection.
        virtual double adjustOffset(const TWeightStyleVec &weightStyles,
                                    const TDouble1Vec &samples,
                                    const TDouble4Vec1Vec &weights) = 0;

        //! Get the current sample offset.
        virtual double offset() const = 0;

        //! Update the prior with a collection of independent samples from the
        //! variable.
        //!
        //! \param[in] weightStyles Controls the interpretation of the weight(s)
        //! that are associated with each sample. See maths_t::ESampleWeightStyle
        //! for more details.
        //! \param[in] samples A collection of samples of the variable.
        //! \param[in] weights The weights of each sample in \p samples.
        virtual void addSamples(const TWeightStyleVec &weightStyles,
                                const TDouble1Vec &samples,
                                const TDouble4Vec1Vec &weights) = 0;

        //! Update the prior for the specified elapsed time.
        virtual void propagateForwardsByTime(double time) = 0;

        //! Get the support for the marginal likelihood function.
        virtual TDoubleDoublePr marginalLikelihoodSupport() const = 0;

        //! Get the mean of the marginal likelihood function.
        virtual double marginalLikelihoodMean() const = 0;

        //! Get the nearest mean of the multimodal prior marginal likelihood,
        //! otherwise the marginal likelihood mean.
        virtual double nearestMarginalLikelihoodMean(double value) const;

        //! Get the mode of the marginal likelihood function.
        virtual double marginalLikelihoodMode(const TWeightStyleVec &weightStyles = TWeights::COUNT_VARIANCE,
                                              const TDouble4Vec &weights = TWeights::UNIT) const = 0;

        //! Get the local maxima of the marginal likelihood function.
        virtual TDouble1Vec marginalLikelihoodModes(const TWeightStyleVec &weightStyles = TWeights::COUNT_VARIANCE,
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
        virtual TDoubleDoublePr
            marginalLikelihoodConfidenceInterval(double percentage,
                                                 const TWeightStyleVec &weightStyles = TWeights::COUNT_VARIANCE,
                                                 const TDouble4Vec &weights = TWeights::UNIT) const = 0;

        //! Get the variance of the marginal likelihood.
        virtual double marginalLikelihoodVariance(const TWeightStyleVec &weightStyles = TWeights::COUNT_VARIANCE,
                                                  const TDouble4Vec &weights = TWeights::UNIT) const = 0;

        //! Calculate the log marginal likelihood function integrating over the
        //! prior density function.
        //!
        //! \param[in] weightStyles Controls the interpretation of the weight(s)
        //! that are associated with each sample. See maths_t::ESampleWeightStyle
        //! for more details.
        //! \param[in] samples A collection of samples of the variable.
        //! \param[in] weights The weights of each sample in \p samples.
        //! \param[out] result Filled in with the joint likelihood of \p samples.
        virtual maths_t::EFloatingPointErrorStatus
            jointLogMarginalLikelihood(const TWeightStyleVec &weightStyles,
                                       const TDouble1Vec &samples,
                                       const TDouble4Vec1Vec &weights,
                                       double &result) const = 0;

        //! Sample the marginal likelihood function.
        //!
        //! The marginal likelihood functions are sampled in quantile intervals.
        //! The idea is to capture a set of samples that accurately and efficiently
        //! represent the information in the prior. Random sampling (although it has
        //! nice asymptotic properties) doesn't fulfill the second requirement:
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
                                              TDouble1Vec &samples) const = 0;

        //! Calculate minus the log of the joint c.d.f. of the marginal likelihood
        //! for a collection of independent samples from the variable.
        //!
        //! \param[in] weightStyles Controls the interpretation of the weights
        //! that are associated with each sample. See maths_t::ESampleWeightStyle
        //! for more details.
        //! \param[in] samples A collection of samples of the variable.
        //! \param[in] weights The weights of each sample in \p samples.
        //! \param[out] lowerBound Filled in with a lower bound for
        //! \f$-\log(\prod_i{F(x_i)})\f$ where \f$F(.)\f$ is the c.d.f. and
        //! \f$\{x_i\}\f$ are the samples.
        //! \param[out] upperBound Filled in with a upper bound for
        //! \f$-\log(\prod_i{F(x_i)})\f$ where \f$F(.)\f$ is the c.d.f. and
        //! \f$\{x_i\}\f$ are the samples.
        //! \note The samples are assumed to be independent.
        //! \note In general, \p lowerBound equals \p upperBound. However,
        //! in some cases insufficient data is held to exactly compute the
        //! c.d.f. in which case the we use sharp upper and lower bounds.
        //! \warning The variance scales must be in the range \f$(0,\infty)\f$,
        //! i.e. a value of zero is not well defined and a value of infinity
        //! is not well handled. (Very large values are handled though.)
        virtual bool minusLogJointCdf(const TWeightStyleVec &weightStyles,
                                      const TDouble1Vec &samples,
                                      const TDouble4Vec1Vec &weights,
                                      double &lowerBound,
                                      double &upperBound) const = 0;

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
                                                double &upperBound) const = 0;

        //! Calculate the joint probability of seeing a lower likelihood collection
        //! of independent samples from the variable integrating over the prior
        //! density function.
        //!
        //! \param[in] calculation The style of the probability calculation
        //! (see model_t::EProbabilityCalculation for details).
        //! \param[in] weightStyles Controls the interpretation of the weights
        //! that are associated with each sample. See maths_t::ESampleWeightStyle
        //! for more details.
        //! \param[in] samples A collection of samples of the variable.
        //! \param[in] weights The weights of each sample in \p samples.
        //! \param[out] lowerBound Filled in with a lower bound for the probability
        //! of the set for which the joint marginal likelihood is less than
        //! that of \p samples (subject to the measure \p calculation).
        //! \param[out] upperBound Filled in with an upper bound for the
        //! probability of the set for which the joint marginal likelihood is
        //! less than that of \p samples (subject to the measure \p calculation).
        //! \param[out] tail The tail that (left or right) that all the
        //! samples are in or neither.
        //! \note The samples are assumed to be independent.
        //! \note In general, \p lowerBound equals \p upperBound. However,
        //! in some cases insufficient data is held to exactly compute the
        //! c.d.f. in which case the we use sharp upper and lower bounds.
        //! \warning The variance scales must be in the range \f$(0,\infty)\f$,
        //! i.e. a value of zero is not well defined and a value of infinity
        //! is not well handled. (Very large values are handled though.)
        virtual bool probabilityOfLessLikelySamples(maths_t::EProbabilityCalculation calculation,
                                                    const TWeightStyleVec &weightStyles,
                                                    const TDouble1Vec &samples,
                                                    const TDouble4Vec1Vec &weights,
                                                    double &lowerBound,
                                                    double &upperBound,
                                                    maths_t::ETail &tail) const = 0;

        //! Check if this is a non-informative prior.
        virtual bool isNonInformative() const = 0;

        //! Get a human readable description of the prior.
        std::string print() const;

        //! Get a human readable description of the prior.
        //!
        //! \param[in] indent The indent to use at the start of new lines.
        //! \param[in,out] result Filled in with the description.
        virtual void print(const std::string &indent, std::string &result) const = 0;

        //! Print the marginal likelihood function.
        //!
        //! The format is as follows:\n
        //! \code{cpp}
        //!   x = [x1 x2 .... xn ];
        //!   likelihood = [L(x1) L(x2) ... L(xn) ];
        //! \endcode
        //!
        //! i.e. domain values are space separated on the first line and the likelihood
        //! evaluated at those values are space separated on the next line.
        virtual std::string printMarginalLikelihoodFunction(double weight = 1.0) const;

        //! Return the plot data for the marginal likelihood function.
        //!
        //! \param[in] numberPoints Number of points to use in the returned plot.
        //! \param[in] weight A scale which is applied to all likelihoods.
        SPlot marginalLikelihoodPlot(unsigned int numberPoints, double weight = 1.0) const;

        //! Print the prior density function of the parameters.
        //!
        //! The format is as follows:\n
        //! \code{cpp}
        //!    x = [x1 x2 ... xn ];
        //!    y = [y1 y2 ... yn ];
        //!    pdf = [f(x1, y1) f(x1, y2) ... f(x1, yn)
        //!           f(x2, y1) f(x2, y2) ... f(x2, yn)
        //!           ...
        //!           f(xn, y1) f(xn, y2) ... f(xn, yn)
        //!          ];
        //! \endcode
        //!
        //! i.e. domain values are space separated on the first and subsequent line(s)
        //! as appropriate and the density function evaluated at those values are space
        //! separated on the next line and subsequent lines as appropriate.
        virtual std::string printJointDensityFunction() const = 0;

        //! Get a checksum for this object.
        virtual uint64_t checksum(uint64_t seed = 0) const = 0;

        //! Get the memory used by this component
        virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const = 0;

        //! Get the memory used by this component
        virtual std::size_t memoryUsage() const = 0;

        //! Get the static size of this object - used for virtual hierarchies
        virtual std::size_t staticSize() const = 0;

        //! Persist state by passing information to the supplied inserter
        virtual void acceptPersistInserter(core::CStatePersistInserter &inserter) const = 0;
        //@}

        //! Compute the expectation of the specified function w.r.t. to the marginal
        //! likelihood.
        //!
        //! This computes the expectation using order three Gauss-Legendre quadrature
        //! in \p numberIntervals subdivisions of a high confidence interval for the
        //! marginal likelihood.
        //!
        //! \param f The function to integrate.
        //! \param numberIntervals The number intervals to use for integration.
        //! \param result Filled in with the result if the expectation could be calculated.
        //!
        //! \tparam F This must conform to the function type expected by
        //! CIntegration::gaussLegendre.
        //! \tparam T The return type of the function F which must conform to the type
        //! expected by CIntegration::gaussLegendre.
        template<typename F, typename T>
        bool expectation(const F &f,
                         const std::size_t numberIntervals,
                         T &result,
                         const TWeightStyleVec &weightStyles = TWeights::COUNT_VARIANCE,
                         const TDouble4Vec &weights = TWeights::UNIT) const;

        //! Get the number of samples received to date.
        double numberSamples() const;

        //! Set the number of samples received to \p numberSamples.
        //!
        //! This is managed internally and generally should not be called by users.
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

        //! Get a set of sample for the prior to use in adjust offset.
        void adjustOffsetResamples(double minimumSample,
                                   TDouble1Vec &resamples,
                                   TDouble4Vec1Vec &resamplesWeights) const;

    protected:
        //! \brief Defines a set of operations to adjust the offset parameter
        //! of those priors with non-negative support.
        class MATHS_EXPORT COffsetParameters
        {
            public:
                COffsetParameters(CPrior &prior);
                virtual ~COffsetParameters() = default;

                //! Add a collection of samples.
                void samples(const maths_t::TWeightStyleVec &weightStyles,
                             const TDouble1Vec &samples,
                             const TDouble4Vec1Vec &weights);

                //! Capture a collection of re-samples from the prior.
                virtual void resample(double minimumSample);

            protected:
                CPrior &prior() const;
                const maths_t::TWeightStyleVec &weightStyles() const;
                const TDouble1Vec &samples() const;
                const TDouble4Vec1Vec &weights() const;
                const TDouble1Vec &resamples() const;
                const TDouble4Vec1Vec &resamplesWeights() const;

            private:
                CPrior *m_Prior;
                const maths_t::TWeightStyleVec *m_WeightStyles;
                const TDouble1Vec *m_Samples;
                const TDouble4Vec1Vec *m_Weights;
                TDouble1Vec m_Resamples;
                TDouble4Vec1Vec m_ResamplesWeights;
        };

        //! \brief Computes the likelihood of a collection of samples and
        //! resamples at different offsets.
        //!
        //! This is used to maximize the data likelihood w.r.t. the choice
        //! of offset.
        class MATHS_EXPORT COffsetCost : public COffsetParameters
        {
            public:
                using result_type = double;

            public:
                COffsetCost(CPrior &prior);

                //! Compute the cost.
                double operator()(double offset) const;

            protected:
                virtual void resetPriors(double offset) const;
                virtual double computeCost(double offset) const;
        };

        //! \brief Apply a specified offset to a prior.
        class MATHS_EXPORT CApplyOffset : public COffsetParameters
        {
            public:
                CApplyOffset(CPrior &prior);

                //! Apply the offset.
                virtual void operator()(double offset) const;
        };

    protected:
        //! The number of times we sample the prior when adjusting the offset.
        static const std::size_t ADJUST_OFFSET_SAMPLE_SIZE;

    protected:
        //! For priors with non-negative support this adjusts the offset used
        //! to extend the support to handle negative samples by maximizing a
        //! specified reward.
        //!
        //! \return The penalty to apply to the model in selection.
        double adjustOffsetWithCost(const TWeightStyleVec &weightStyles,
                                    const TDouble1Vec &samples,
                                    const TDouble4Vec1Vec &weights,
                                    COffsetCost &cost,
                                    CApplyOffset &apply);

        //! Update the number of samples received to date by adding \p n.
        void addSamples(double n);

        //! Get a debug description of the prior parameters.
        virtual std::string debug() const;

    private:
        //! If this is true then the prior is being used to model discrete
        //! data. Note that this is not persisted and deduced from context.
        maths_t::EDataType m_DataType;

        //! The rate at which the prior returns to non-informative. Note that
        //! this is not persisted.
        CFloatStorage m_DecayRate;

        //! The number of samples with which the prior has been updated.
        CFloatStorage m_NumberSamples;
};

}
}

#endif // INCLUDED_ml_maths_CPrior_h
