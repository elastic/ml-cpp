/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_ProbabilityAggregators_h
#define INCLUDED_ml_maths_ProbabilityAggregators_h

#include <core/CoreTypes.h>

#include <maths/CBasicStatistics.h>
#include <maths/ImportExport.h>

#include <boost/optional.hpp>

#include <iosfwd>

namespace ml
{
namespace core
{
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths
{

//! \brief Computes the joint probability of seeing a more extreme
//! collection of samples.
//!
//! DESCRIPTION:\n
//! This assumes that the samples are jointly normal and independent
//! and computes the probability of seeing a more extreme event:
//! <pre class="fragment">
//!   \f$P(\{(s_1, ..., s_n)\ |\ L(s_1, ..., s_n) \leq \prod_{i}\frac{P_i}{2}\})\f$
//! </pre>
//!
//! where \f$P(i)\f$ are the probabilities supplied to the add function.
//! Note that because it receives probabilities, which it interprets as
//! from a standard Gaussian, these can in fact be computed from any
//! distribution.
//!
//! It is possible to supply a weight with each probability. This can
//! be used to approximately compute the expectation of the joint
//! probability of a collection of samples which are sampled where
//! each sample only appears with some specified frequency. The weights
//! must be non-negative.
class MATHS_EXPORT CJointProbabilityOfLessLikelySamples : private boost::addable<CJointProbabilityOfLessLikelySamples>
{
    public:
        typedef boost::optional<double> TOptionalDouble;

        //! Functor wrapper of CJointProbabilityOfLessLikelySamples::add.
        struct SAddProbability
        {
            CJointProbabilityOfLessLikelySamples &
                operator()(CJointProbabilityOfLessLikelySamples &jointProbability,
                           double probability,
                           double weight = 1.0) const;
        };

    public:
        CJointProbabilityOfLessLikelySamples(void);

        //! Initialize from \p value if possible.
        bool fromDelimited(const std::string &value);

        //! Convert to a delimited string.
        std::string toDelimited(void) const;

        //! Combine two joint probability calculators.
        const CJointProbabilityOfLessLikelySamples &
            operator+=(const CJointProbabilityOfLessLikelySamples &other);

        //! Add \p probability.
        void add(double probability, double weight = 1.0);

        //! Calculate the joint probability of less likely samples
        //! than those added so far.
        bool calculate(double &result) const;

        //! Compute the average probability of less likely samples
        //! added so far.
        bool averageProbability(double &result) const;

        //! Get the first probability.
        TOptionalDouble onlyProbability(void) const;

        //! Get the total deviation of all samples added.
        double distance(void) const;

        //! Get the count of all samples added.
        double numberSamples(void) const;

        //! Get a checksum for an object of this class.
        uint64_t checksum(uint64_t seed) const;

        //! Print the joint probability for debugging.
        std::ostream &print(std::ostream &o) const;

    private:
        TOptionalDouble m_OnlyProbability;
        double m_Distance;
        double m_NumberSamples;
};

MATHS_EXPORT
std::ostream &operator<<(std::ostream &o,
                         const CJointProbabilityOfLessLikelySamples &probability);

//! \brief Computes log of the joint probability of seeing a more
//! extreme collection of samples.
//!
//! DESCRIPTION:\n
//! This assumes that the samples are jointly normal and independent
//! and computes upper and lower bound on log of the probability of
//! seeing a more extreme event:
//! <pre class="fragment">
//!   \f$P(\{(s_1, ..., s_n)\ |\ L(s_1, ..., s_n) \leq \prod_{i}\frac{P_i}{2}\})\f$
//! </pre>
//!
//! where \f$P(i)\f$ are the probabilities supplied to the add function.
//! It is intended for use when the joint probability is likely to be
//! very small and, in particular, will underflow double. Otherwise, the
//! upper and lower bound functions just return the log of the result of
//! CJointProbabilityOfLessLikelySamples::calculate. Note that because it
//! receives probabilities, which it interprets as from a standard Gaussian,
//! these can in fact be computed from any distribution.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This implements an upper and lower bound because because exact
//! calculation for very small probabilities is expensive and it is intended
//! that the function be used to generate an ordering of a collection of
//! joint probabilities, which should respect the error in the bounds.
//! For example, two probabilities should be treated as equal if the
//! intervals defined by their upper and lower bounds intersect.
class MATHS_EXPORT CLogJointProbabilityOfLessLikelySamples : protected CJointProbabilityOfLessLikelySamples,
                                                             private boost::addable<CLogJointProbabilityOfLessLikelySamples>
{
    public:
        CLogJointProbabilityOfLessLikelySamples(void);

        //! Combine two log joint probability calculators.
        const CLogJointProbabilityOfLessLikelySamples &
            operator+=(const CLogJointProbabilityOfLessLikelySamples &other);

        //! Add \p probability.
        void add(double probability, double weight = 1.0);

        //! Calculate a lower bound for the log of the joint probability
        //! of less likely samples than those added so far.
        bool calculateLowerBound(double &result) const;

        //! Calculate an upper bound for the log of the joint probability
        //! of less likely samples than those added so far.
        bool calculateUpperBound(double &result) const;
};

//! \brief Computes probability of seeing the most extreme sample
//! in a collection of N samples.
//!
//! DESCRIPTION:\n
//! By most extreme sample we mean the sample \f$x\f$ for which:
//! <pre class="fragment">
//!   \f$ p = P(\{s\ |\ L(s) \leq L(x)\}) \f$
//! </pre>
//!
//! is the smallest. For simplicity assuming that the p.d.f. \f$f(.)\f$
//! is symmetric, it can be shown that marginal likelihood of the
//! most extreme sample as a function of \f$s\f$ is:
//! <pre class="fragment">
//!   \f$ L'(s) = \frac{N!}{1!(N - 1)!} (F(s) - F(-s)) ^{(N-1)} f(s) \f$
//! </pre>
//!
//! where \f$F(.)\f$ is the c.d.f. At this point we could evaluate
//! the probability of the event:
//! <pre class="fragment">
//!   \f$ R' = \{s\ |\ L'(s) \leq L'(x)\} \f$
//! </pre>
//!
//! However, we are not interested in the case that p is too large
//! since we believe that the signals we model are not truly random
//! and may often exhibit lower deviation than our models predict.
//! Therefore, we instead seek the probability of the event:
//! <pre class="fragment">
//!   \f$ R = \{s\ |\ s > x\ or\ s < x\} \f$
//! </pre>
//!
//! Integrating over \f$(-\infty, x] \bigcup [x, \infty)\f$ and again
//! by symmetry we have that:
//! <pre class="fragment">
//!   \f$ P(R) = \frac{2N}{2N} \left[(2t - 1) ^{N}\right]_{F(x)}^1 = 1 - (1 - p) ^{N} \f$
//! </pre>
//!
//! where we have used the fact that \f$(1 - F(x)) = p / 2\f$.
class MATHS_EXPORT CProbabilityOfExtremeSample : private boost::addable<CProbabilityOfExtremeSample>
{
    public:
        CProbabilityOfExtremeSample(void);

        //! Initialize from \p value if possible.
        bool fromDelimited(const std::string &value);

        //! Convert to a delimited string.
        std::string toDelimited(void) const;

        //! Combine two extreme probability calculators.
        const CProbabilityOfExtremeSample &
            operator+=(const CProbabilityOfExtremeSample &other);

        //! Add \p probability.
        bool add(double probability, double weight = 1.0);

        //! Calculate the probability of seeing the most extreme
        //! sample added so far.
        bool calculate(double &result) const;

        //! Get a checksum for an object of this class.
        uint64_t checksum(uint64_t seed) const;

        //! Print the extreme probability for debugging.
        std::ostream &print(std::ostream &o) const;

    private:
        typedef CBasicStatistics::COrderStatisticsStack<double, 1u> TMinValueAccumulator;

    private:
        TMinValueAccumulator m_MinValue;
        double m_NumberSamples;
};

MATHS_EXPORT
std::ostream &operator<<(std::ostream &o,
                         const CProbabilityOfExtremeSample &probability);


//! \brief Computes the probability of seeing the M most extreme
//! samples in a collection of N samples.
//!
//! DESCRIPTION:\n
//! A sample \f$x_2\f$ is more extreme than a sample \f$x_1\f$ if:
//! <pre class="fragment">
//!   \f$ p_2 = P(\{s\ |\ L(s) \leq L(x_2)\}) < p_1 = P(\{s\ |\ L(s) \leq L(x_1)\}) \f$
//! </pre>
//!
//! For simplicity assuming that the p.d.f. \f$f(.)\f$ is symmetric,
//! it can be shown that marginal likelihood of the M most extreme
//! samples as a function of \f$(s_1, s_2, ..., s_M)\f$ is:
//! <pre class="fragment">
//!   \f$ L'(s_1, ..., s_M) = \frac{N!}{(N-M)!} (F(s_1) - F(-s_1)) ^{N-M} \prod_{i=1}^M f(s_i) \f$
//! </pre>
//!
//! where,\n
//!   \f$(s_1, s_2, ..., s_M)\f$ are ordered s.t. \f$p_1 > p_2 > ...> p_M\f$\n
//!   \f$F(.)\f$ is the is the c.d.f.
//!
//! At this point we could evaluate the probability of the event:
//! <pre class="fragment">
//!   \f$ R'' = \{s\ |\ L'(s) \leq L'(x)\} \f$
//! </pre>
//!
//! where \f$s\f$ and \f$x\f$ are understood to be vector quantities.
//! However, we are not interested in the case that probabilities are
//! too large since we believe that the signals we model are not truly
//! random and may often exhibit lower deviation than our models
//! predict. Therefore, we will instead seek the probability of the
//! of the event:
//! <pre class="fragment">
//!   \f$ R' = \{s\ |\ s_i > x_i\ or\ s_i < x_i\} \f$
//! </pre>
//!
//! By symmetry:
//! <pre class="fragment">
//!   \f$ P(R') = 2 ^{M} P(\{s\ |\ s_i > x_i\}) \f$
//! </pre>
//!
//! Note that since \f$s_i\f$ are ordered, \f$s_i < s_{i+1}\f$ for
//! \f$i < M\f$ and \f$s_M < \infty\f$.
//!
//! The integral representing \f$P(R)\f$ can be evaluated in order \f$M^2\f$
//! as a polynomial in the individual probabilities \f$\{p_1, ..., p_M\}\f$
//! with recurrence relations used to compute the coefficients.
class MATHS_EXPORT CLogProbabilityOfMFromNExtremeSamples : private boost::addable<CLogProbabilityOfMFromNExtremeSamples>
{
    public:
        CLogProbabilityOfMFromNExtremeSamples(std::size_t m);

        //! Initialize from \p value if possible.
        bool fromDelimited(const std::string &value);

        //! Convert to a delimited string.
        std::string toDelimited(void) const;

        //! Combine two extreme probability calculators.
        const CLogProbabilityOfMFromNExtremeSamples &
            operator+=(const CLogProbabilityOfMFromNExtremeSamples &other);

        //! Add \p probability.
        void add(double probability);

        //! Calculate the probability of seeing the "M" most extreme
        //! samples added so far.
        bool calculate(double &result);

        //! Calculate the calibrated probability of seeing the "M" most
        //! extreme samples added so far.
        bool calibrated(double &result);

        //! Get a checksum for an object of this class.
        uint64_t checksum(uint64_t seed) const;

    private:
        typedef CBasicStatistics::COrderStatisticsHeap<double> TMinValueAccumulator;

    private:
        TMinValueAccumulator m_MinValues;
        std::size_t m_NumberSamples;
};

}
}

#endif // INCLUDED_ml_maths_ProbabilityAggregators_h
