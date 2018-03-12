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

#ifndef INCLUDED_ml_model_CAnomalyScore_h
#define INCLUDED_ml_model_CAnomalyScore_h

#include <core/CoreTypes.h>

#include <maths/CBasicStatistics.h>
#include <maths/CQDigest.h>

#include <model/ImportExport.h>

#include <boost/optional.hpp>
#include <boost/shared_ptr.hpp>

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <stdint.h>


namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {
class CPrior;
}
namespace model {
class CAnomalyDetectorModelConfig;
class CLimits;

//! \brief An anomaly score calculator.
//!
//! DESCRIPTION:\n
//! A collection of utility functions for computing and normalizing
//! anomaly scores.
class MODEL_EXPORT CAnomalyScore {
    public:
        typedef std::vector<double>          TDoubleVec;
        typedef TDoubleVec::iterator         TDoubleVecItr;
        typedef TDoubleVec::const_iterator   TDoubleVecCItr;
        typedef boost::optional<double>      TOptionalDouble;
        typedef std::vector<TOptionalDouble> TOptionalDoubleVec;
        typedef std::vector<std::string>     TStrVec;

        //! Attributes for a persisted normalizer
        static const std::string MLCUE_ATTRIBUTE;
        static const std::string MLKEY_ATTRIBUTE;
        static const std::string MLQUANTILESDESCRIPTION_ATTRIBUTE;
        static const std::string MLVERSION_ATTRIBUTE;
        static const std::string TIME_ATTRIBUTE;

        static const std::string CURRENT_FORMAT_VERSION;

        //! Severities
        static const std::string WARNING_SEVERITY;
        static const std::string MINOR_SEVERITY;
        static const std::string MAJOR_SEVERITY;
        static const std::string CRITICAL_SEVERITY;

    public:
        //! \brief Wrapper around CAnomalyScore::compute.
        class MODEL_EXPORT CComputer {
            public:
                CComputer(double jointProbabilityWeight,
                          double extremeProbabilityWeight,
                          std::size_t minExtremeSamples,
                          std::size_t maxExtremeSamples,
                          double maximumAnomalousProbability);

                //! Compute the overall anomaly score and aggregate probability.
                bool operator()(const TDoubleVec &probabilities,
                                double &overallAnomalyScore,
                                double &overallProbability) const;

            private:
                //! The weight to assign the joint probability.
                double m_JointProbabilityWeight;
                //! The weight to assign the extreme probability.
                double m_ExtremeProbabilityWeight;
                //! The minimum number of samples to include in the extreme
                //! probability calculation.
                std::size_t m_MinExtremeSamples;
                //! The maximum number of samples to include in the extreme
                //! probability calculation.
                std::size_t m_MaxExtremeSamples;
                //! The maximum probability which is deemed to be anomalous.
                double m_MaximumAnomalousProbability;
        };

        //! \brief Manages the normalization of aggregate anomaly scores
        //! based on historic values percentiles.
        class MODEL_EXPORT CNormalizer : private core::CNonCopyable {
            public:
                explicit CNormalizer(const CAnomalyDetectorModelConfig &config);

                //! Does this normalizer have enough information to normalize
                //! anomaly scores?
                bool canNormalize(void) const;

                //! This normalizes the aggregate scores, i.e. the sum
                //! of \p scores, and scales all the scores by a constant
                //! s.t. they sum to the normalized score. The normalized
                //! score is in the range [0, 100].
                //!
                //! \param[in,out] scores The raw scores to normalize.
                //! Filled in with the normalized scores.
                bool normalize(TDoubleVec &scores) const;

                //! As above but taking a single pre-aggregated \p score instead
                //! of a vector of scores to be aggregated.
                bool normalize(double &score) const;

                //! Estimate the quantile range including the \p score.
                //!
                //! \param[in] score The score to estimate.
                //! \param[in] confidence The quantile central confidence interval.
                //! \param[out] lowerBound The quantile lower bound of \p score.
                //! \param[out] upperBound The quantile upper bound of \p score.
                void quantile(double score,
                              double confidence,
                              double &lowerBound,
                              double &upperBound) const;

                //! Updates the quantile summaries with the total of
                //! \p scores.
                //! \return true if a big change occurred, otherwise false
                bool updateQuantiles(const TDoubleVec &scores);

                //! Updates the quantile summaries with \p score.
                //! \return true if a big change occurred, otherwise false
                bool updateQuantiles(double score);

                //! Age the maximum score and quantile summary.
                void propagateForwardByTime(double time);

                //! Report whether it would be possible to upgrade one version
                //! of the quantiles to another.
                static bool isUpgradable(const std::string &fromVersion,
                                         const std::string &toVersion);

                //! Scale the maximum score and quantile summary.  To be used
                //! after upgrades if different versions of the product produce
                //! different raw anomaly scores.
                bool upgrade(const std::string &loadedVersion,
                             const std::string &currentVersion);

                //! Set the normalizer back to how it was immediately after
                //! construction
                void clear(void);

                //! \name Serialization
                //@{
                //! Persist state by passing information to the supplied inserter
                void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

                //! Create from a state document.
                bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);
                //@}

            public:
                //! Get a checksum of the object.
                uint64_t checksum(void) const;

            private:
                typedef std::pair<double, double> TDoubleDoublePr;
                typedef std::vector<TDoubleDoublePr> TDoubleDoublePrVec;
                typedef TDoubleDoublePrVec::const_iterator TDoubleDoublePrVecCItr;
                typedef std::greater<double> TGreaterDouble;
                typedef maths::CBasicStatistics::COrderStatisticsStack<double, 1u, TGreaterDouble> TMaxValueAccumulator;

            private:
                //! Used to convert raw scores in to integers so that we
                //! can use the q-digest.
                static const double DISCRETIZATION_FACTOR;

                //! We maintain a separate digest for the scores greater
                //! than some high percentile (specified by this constant).
                //! This is because we want the highest resolution in the
                //! scores for the extreme (high quantile) raw scores.
                static const double HIGH_PERCENTILE;

                //! The time between aging quantiles. These age at a slower
                //! rate which we achieve by only aging them after a certain
                //! period has elapsed.
                static const double QUANTILE_DECAY_TIME;

                //! The increase in maximum score that will be considered a
                //! big change when updating the quantiles.
                static const double BIG_CHANGE_FACTOR;

            private:
                //! Compute the discrete score from a raw score.
                uint32_t discreteScore(double rawScore) const;

                //! Extract the raw score from a discrete score.
                double rawScore(uint32_t discreteScore) const;

            private:
                //! The percentile defining the largest noise score.
                double m_NoisePercentile;
                //! The multiplier used to estimate the anomaly threshold.
                double m_NoiseMultiplier;
                //! The normalized anomaly score knot points.
                TDoubleDoublePrVec m_NormalizedScoreKnotPoints;
                //! The maximum possible normalized score.
                double m_MaximumNormalizedScore;

                //! The approximate HIGH_PERCENTILE percentile raw score.
                uint32_t m_HighPercentileScore;
                //! The number of scores less than the approximate
                //! HIGH_PERCENTILE percentile raw score.
                uint64_t m_HighPercentileCount;

                //! The maximum score ever received.
                TMaxValueAccumulator m_MaxScore;

                //! The factor used to scale the quantile scores to convert
                //! values per bucket length to values in absolute time. We
                //! scale all values to an effective bucket length 30 mins.
                //! So, a percentile of 99% would correspond to a 1 in 50
                //! hours event.
                double m_BucketNormalizationFactor;

                //! A quantile summary of the raw scores.
                maths::CQDigest m_RawScoreQuantileSummary;
                //! A quantile summary of the raw score greater than the
                //! approximate HIGH_PERCENTILE percentile raw score.
                maths::CQDigest m_RawScoreHighQuantileSummary;

                //! The rate at which information is lost.
                double m_DecayRate;
                //! The time to when we next age the quantiles.
                double m_TimeToQuantileDecay;
        };

        typedef boost::shared_ptr<CNormalizer> TNormalizerP;

    public:
        //! Compute a joint anomaly score for a collection of probabilities.
        //!
        //! The joint anomaly score is assigned pro-rata to each member
        //! of the collection based on its anomalousness.
        //!
        //! \param[in] jointProbabilityWeight The weight to assign the
        //! joint probability.
        //! \param[in] extremeProbabilityWeight The weight to assign the
        //! extreme probability.
        //! \param[in] minExtremeSamples The minimum number of samples to
        //! include in the extreme probability calculation.
        //! \param[in] maxExtremeSamples The maximum number of samples to
        //! include in the extreme probability calculation.
        //! \param[in] maximumAnomalousProbability The largest probability
        //! with non-zero anomaly score.
        //! \param[in] probabilities A collection of probabilities for
        //! which to compute an aggregate probability and total score.
        //! \param[out] overallAnomalyScore Filled in with the overall
        //! anomaly score.
        //! \param[out] overallProbability Filled in with the overall
        //! probability.
        static bool compute(double jointProbabilityWeight,
                            double extremeProbabilityWeight,
                            std::size_t minExtremeSamples,
                            std::size_t maxExtremeSamples,
                            double maximumAnomalousProbability,
                            const TDoubleVec &probabilities,
                            double &overallAnomalyScore,
                            double &overallProbability);

        //! Given a normalized score, find the most appropriate severity string
        static const std::string &normalizedScoreToSeverity(double normalizedScore);

        //! Populate \p normalizer from its JSON representation
        static bool normalizerFromJson(const std::string &json,
                                       CNormalizer &normalizer);

        //! Populate \p normalizer from the restore traverser
        static bool normalizerFromJson(core::CStateRestoreTraverser &traverser,
                                       CNormalizer &normalizer);

        //! Convert \p normalizer to its JSON representation with a restoration
        //! cue and description specified by the caller
        static void normalizerToJson(const CNormalizer &normalizer,
                                     const std::string &searchKey,
                                     const std::string &cue,
                                     const std::string &description,
                                     core_t::TTime time,
                                     std::string &json);
};


}
}

#endif // INCLUDED_ml_model_CAnomalyScore_h

