/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CAnomalyScore.h>

#include <core/CContainerPrinter.h>
#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>
#include <core/CPersistUtils.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/CMathsFuncs.h>
#include <maths/CTools.h>
#include <maths/Constants.h>
#include <maths/ProbabilityAggregators.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CLimits.h>

#include <boost/optional.hpp>
#include <boost/range.hpp>
#include <boost/ref.hpp>

#include <numeric>
#include <vector>

namespace ml {
namespace model {

namespace {

using TDoubleVec = std::vector<double>;

//! Add valid \p probabilities to \p aggregator and return the
//! number of valid probabilities.
template<typename AGGREGATOR>
std::size_t addProbabilities(const TDoubleVec& probabilities, AGGREGATOR& aggregator) {
    std::size_t n = 0u;
    for (std::size_t i = 0u; i < probabilities.size(); ++i) {
        double p = probabilities[i];
        if (!(p >= 0.0 && p <= 1.0)) {
            LOG_ERROR(<< "Invalid probability " << p);
        } else {
            ++n;
            aggregator.add(p);
        }
    }
    return n;
}

//! The function to convert probabilities to *raw* scores.
double probabilityToScore(double probability) {
    return maths::CTools::anomalyScore(probability);
}

//! The function to convert *raw* scores to probabilities.
double scoreToProbability(double score) {
    return maths::CTools::inverseAnomalyScore(score);
}

// We use short field names to reduce the state size
const std::string HIGH_PERCENTILE_SCORE_TAG("a");
const std::string HIGH_PERCENTILE_COUNT_TAG("b");
const std::string MAX_SCORE_TAG("c");
const std::string RAW_SCORE_QUANTILE_SUMMARY("d");
const std::string RAW_SCORE_HIGH_QUANTILE_SUMMARY("e");
const std::string TIME_TO_QUANTILE_DECAY_TAG("f");

// Since version >= 6.5
const std::string MAX_SCORES_PER_PARTITION_TAG("g");

const std::string EMPTY_STRING;

// This is the version to assume for the format that doesn't contain a version
// attribute - NEVER CHANGE THIS
const std::string MISSING_VERSION_FORMAT_VERSION("1");

const std::string TIME_INFLUENCER("bucket_time");
}

const std::string CAnomalyScore::MLCUE_ATTRIBUTE("mlcue");
const std::string CAnomalyScore::MLKEY_ATTRIBUTE("mlkey");
const std::string CAnomalyScore::MLQUANTILESDESCRIPTION_ATTRIBUTE("mlquantilesdescription");
const std::string CAnomalyScore::MLVERSION_ATTRIBUTE("mlversion");
const std::string CAnomalyScore::TIME_ATTRIBUTE("time");

// Increment this every time a change to the format is made that requires
// existing state to be discarded
const std::string CAnomalyScore::CURRENT_FORMAT_VERSION("3");

const std::string CAnomalyScore::WARNING_SEVERITY("warning");
const std::string CAnomalyScore::MINOR_SEVERITY("minor");
const std::string CAnomalyScore::MAJOR_SEVERITY("major");
const std::string CAnomalyScore::CRITICAL_SEVERITY("critical");

bool CAnomalyScore::compute(double jointProbabilityWeight,
                            double extremeProbabilityWeight,
                            std::size_t minExtremeSamples,
                            std::size_t maxExtremeSamples,
                            double maximumAnomalousProbability,
                            const TDoubleVec& probabilities,
                            double& overallAnomalyScore,
                            double& overallProbability) {
    overallAnomalyScore = 0.0;
    overallProbability = 1.0;

    if (probabilities.empty()) {
        // Nothing to do.
        return true;
    }

    maths::CLogJointProbabilityOfLessLikelySamples logPJointCalculator;
    std::size_t n = std::min(addProbabilities(probabilities, logPJointCalculator),
                             maxExtremeSamples);

    // Note the upper bound is significantly tighter, so we just
    // use that in the following calculation.
    double logPJoint;
    if (!logPJointCalculator.calculateUpperBound(logPJoint)) {
        LOG_ERROR(<< "Unable to calculate anomaly score"
                  << ", probabilities = " << core::CContainerPrinter::print(probabilities));
        return false;
    }

    // Sanity check the probability not greater than 1.0.
    if (logPJoint > 0.0) {
        LOG_ERROR(<< "Invalid log joint probability " << logPJoint << ", probabilities = "
                  << core::CContainerPrinter::print(probabilities));
        return false;
    }

    double logPExtreme = 0.0;
    for (std::size_t m = 1u, i = maths::CTools::truncate(minExtremeSamples, m, n);
         i <= n; ++i) {
        maths::CLogProbabilityOfMFromNExtremeSamples logPExtremeCalculator(i);
        addProbabilities(probabilities, logPExtremeCalculator);
        double logPi;
        if (!logPExtremeCalculator.calibrated(logPi)) {
            LOG_ERROR(<< "Unable to calculate anomaly score"
                      << ", probabilities = " << core::CContainerPrinter::print(probabilities));
            return false;
        }
        if (logPi < logPExtreme) {
            logPExtreme = logPi;
        }
    }

    // Sanity check the probability in the range [0, 1].
    if (logPExtreme > 0.0) {
        LOG_ERROR(<< "Invalid log extreme probability " << logPExtreme << ", probabilities = "
                  << core::CContainerPrinter::print(probabilities));
        return false;
    }

    double logMaximumAnomalousProbability = std::log(maximumAnomalousProbability);
    if (logPJoint > logMaximumAnomalousProbability && logPExtreme > logMaximumAnomalousProbability) {
        overallProbability = std::exp(jointProbabilityWeight * logPJoint) *
                             std::exp(extremeProbabilityWeight * logPExtreme);
        return true;
    }

    // The following extends the range of the joint probability to
    // [e^-100000, 1].

    static const double NORMAL_RANGE_SCORE_FRACTION = 0.8;
    static const double LOG_SMALLEST_PROBABILITY =
        std::log(maths::CTools::smallestProbability());
    static const double SMALLEST_PROBABILITY_DEVIATION =
        probabilityToScore(maths::CTools::smallestProbability());
    static const double SMALLEST_LOG_JOINT_PROBABILTY = -100000.0;
    static const double SMALLEST_LOG_EXTREME_PROBABILTY = -1500.0;

    if (logPJoint < LOG_SMALLEST_PROBABILITY) {
        double interpolate = std::min((logPJoint - LOG_SMALLEST_PROBABILITY) /
                                          (SMALLEST_LOG_JOINT_PROBABILTY - LOG_SMALLEST_PROBABILITY),
                                      1.0);
        overallAnomalyScore = (NORMAL_RANGE_SCORE_FRACTION +
                               (1.0 - NORMAL_RANGE_SCORE_FRACTION) * interpolate) *
                              jointProbabilityWeight * SMALLEST_PROBABILITY_DEVIATION;
    } else {
        overallAnomalyScore = NORMAL_RANGE_SCORE_FRACTION * jointProbabilityWeight *
                              probabilityToScore(std::exp(logPJoint));
    }

    if (logPExtreme < LOG_SMALLEST_PROBABILITY) {
        double interpolate = std::min((logPExtreme - LOG_SMALLEST_PROBABILITY) /
                                          (SMALLEST_LOG_EXTREME_PROBABILTY - LOG_SMALLEST_PROBABILITY),
                                      1.0);
        overallAnomalyScore += (NORMAL_RANGE_SCORE_FRACTION +
                                (1.0 - NORMAL_RANGE_SCORE_FRACTION) * interpolate) *
                               extremeProbabilityWeight * SMALLEST_PROBABILITY_DEVIATION;
    } else {
        overallAnomalyScore += NORMAL_RANGE_SCORE_FRACTION * extremeProbabilityWeight *
                               probabilityToScore(std::exp(logPExtreme));
    }

    // Invert the deviation in the region it is 1-to-1 otherwise
    // use the weighted harmonic mean.
    overallProbability =
        overallAnomalyScore > 0.0
            ? scoreToProbability(std::min(overallAnomalyScore / NORMAL_RANGE_SCORE_FRACTION,
                                          SMALLEST_PROBABILITY_DEVIATION))
            : std::exp(jointProbabilityWeight * logPJoint) *
                  std::exp(extremeProbabilityWeight * logPExtreme);

    LOG_TRACE(<< "logJointProbability = " << logPJoint << ", jointProbabilityWeight = "
              << jointProbabilityWeight << ", logExtremeProbability = " << logPExtreme
              << ", extremeProbabilityWeight = " << extremeProbabilityWeight
              << ", overallProbability = " << overallProbability
              << ", overallAnomalyScore = " << overallAnomalyScore
              << ", # probabilities = " << probabilities.size()
              << ", probabilities = " << core::CContainerPrinter::print(probabilities));

    return true;
}

CAnomalyScore::CComputer::CComputer(double jointProbabilityWeight,
                                    double extremeProbabilityWeight,
                                    std::size_t minExtremeSamples,
                                    std::size_t maxExtremeSamples,
                                    double maximumAnomalousProbability)
    : m_JointProbabilityWeight(jointProbabilityWeight),
      m_ExtremeProbabilityWeight(extremeProbabilityWeight),
      m_MinExtremeSamples(std::min(minExtremeSamples, maxExtremeSamples)),
      m_MaxExtremeSamples(maxExtremeSamples),
      m_MaximumAnomalousProbability(maximumAnomalousProbability) {
}

bool CAnomalyScore::CComputer::operator()(const TDoubleVec& probabilities,
                                          double& overallAnomalyScore,
                                          double& overallProbability) const {
    return CAnomalyScore::compute(m_JointProbabilityWeight, m_ExtremeProbabilityWeight,
                                  m_MinExtremeSamples, m_MaxExtremeSamples,
                                  m_MaximumAnomalousProbability, probabilities,
                                  overallAnomalyScore, overallProbability);
}

CAnomalyScore::CNormalizer::CNormalizer(const CAnomalyDetectorModelConfig& config)
    : m_NoisePercentile(config.noisePercentile()),
      m_NoiseMultiplier(config.noiseMultiplier()),
      m_NormalizedScoreKnotPoints(config.normalizedScoreKnotPoints()),
      m_MaximumNormalizedScore(100.0),
      m_HighPercentileScore(std::numeric_limits<uint32_t>::max()),
      m_HighPercentileCount(0ull),
      m_BucketNormalizationFactor(config.bucketNormalizationFactor()),
      m_RawScoreQuantileSummary(201, config.decayRate()),
      m_RawScoreHighQuantileSummary(201, config.decayRate()),
      m_DecayRate(config.decayRate() *
                  std::max(static_cast<double>(config.bucketLength()) /
                               static_cast<double>(CAnomalyDetectorModelConfig::STANDARD_BUCKET_LENGTH),
                           1.0)),
      m_TimeToQuantileDecay(QUANTILE_DECAY_TIME) {
}

bool CAnomalyScore::CNormalizer::canNormalize() const {
    return m_RawScoreQuantileSummary.n() > 0;
}

bool CAnomalyScore::CNormalizer::normalize(TDoubleVec& scores,
                                           const std::string& partitionName,
                                           const std::string& partitionValue,
                                           const std::string& personName,
                                           const std::string& personValue) const {

    double origScore(std::accumulate(scores.begin(), scores.end(), 0.0));
    double normalizedScore(origScore);

    if (this->normalize(normalizedScore, partitionName, partitionValue,
                        personName, personValue) == false) {
        // Error will have been logged by the called method
        return false;
    }

    if (normalizedScore == origScore) {
        // Nothing to do.
        return true;
    }

    // Normalize the individual scores.
    for (TDoubleVecItr scoreItr = scores.begin(); scoreItr != scores.end(); ++scoreItr) {
        *scoreItr *= normalizedScore / origScore;
    }

    return true;
}

bool CAnomalyScore::CNormalizer::normalize(double& score,
                                           const std::string& partitionName,
                                           const std::string& partitionValue,
                                           const std::string& personName,
                                           const std::string& personValue) const {

    if (score == 0.0) {
        // Nothing to do.
        return true;
    }

    if (m_RawScoreQuantileSummary.n() == 0) {
        LOG_ERROR(<< "No scores have been added to the quantile summary");
        return false;
    }

    LOG_TRACE(<< "Normalising " << score);

    static const double CONFIDENCE_INTERVAL = 70.0;

    double normalizedScores[] = {m_MaximumNormalizedScore, m_MaximumNormalizedScore,
                                 m_MaximumNormalizedScore, m_MaximumNormalizedScore};

    uint32_t discreteScore = this->discreteScore(score);

    // Our normalized score is the minimum of a set of different
    // score ceilings. The idea is that we have a number of factors
    // which can reduce the score based on the other score values
    // observed so far and the absolute score for some of our
    // functions. We want the following properties:
    //   1) The noise like scores, i.e. the smallest scores which
    //      relatively frequently, to have low normalized score.
    //   2) Higher resolution of the highest scores we've seen.
    //   3) The highest raw score ever observed has the highest
    //      normalized score.
    //   4) We don't want to generate significant anomalies for
    //      moderately unusual events before we have enough history
    //      to assess their probability accurately.
    //
    // To ensure 1) we use a ceiling for the score that is a
    // multiple of a low(ish) percentile score. To ensure 2) we
    // use non-linear (actually piecewise linear) mapping from
    // percentiles to scores, i.e. low percentile map to a small
    // normalized score range and high percentiles map to a large
    // normalized score range. Finally to ensure 3) we
    // use a ceiling which is linear interpolation between a raw
    // score of zero and the maximum ever raw score for the partition.
    // To achieve 4), we cap the maximum score such that probabilities
    // near the  cutoff don't generate large normalized scores.

    // Compute the noise ceiling. Note that since the scores are
    // logarithms (base e) the difference corresponds to the scaled,
    // 10 / log(10), signal strength in dB. By default a signal of
    // 75dB corresponds to a normalized score of 75. Note that if
    // the noise percentile is zero then really it is unknown,
    // since scores are truncated to zero. In this case we don't
    // want this term to constrain the normalized score. However,
    // we also want this term to be smooth, i.e. the score should
    // be nearly continuous when F(0) = pn. Here, F(.) denotes the
    // c.d.f. of the score and pn the noise percentile. We achieve
    // this by adding "max score" * min(F(0) / "noise percentile",
    // to the score.
    uint32_t noiseScore;
    m_RawScoreQuantileSummary.quantile(m_NoisePercentile / 100.0, noiseScore);
    TDoubleDoublePrVecCItr knotPoint = std::lower_bound(
        m_NormalizedScoreKnotPoints.begin(), m_NormalizedScoreKnotPoints.end(),
        TDoubleDoublePr(m_NoisePercentile, 0.0));
    double signalStrength =
        m_NoiseMultiplier * 10.0 / DISCRETIZATION_FACTOR *
        (static_cast<double>(discreteScore) - static_cast<double>(noiseScore));
    double l0;
    double u0;
    m_RawScoreQuantileSummary.cdf(0, 0.0, l0, u0);
    normalizedScores[0] =
        knotPoint->second * std::max(1.0 + signalStrength, 0.0) +
        m_MaximumNormalizedScore *
            std::max(2.0 * std::min(50.0 * (l0 + u0) / m_NoisePercentile, 1.0) - 1.0, 0.0);
    LOG_TRACE(<< "normalizedScores[0] = " << normalizedScores[0]
              << ", knotPoint = " << knotPoint->second << ", discreteScore = " << discreteScore
              << ", noiseScore = " << noiseScore << ", l(0) = " << l0
              << ", u(0) = " << u0 << ", signalStrength = " << signalStrength);

    // Compute the raw normalized score. Note we compute the probability
    // of seeing a lower score on the normal bucket length and convert
    // this to an equivalent percentile. This is just the probability
    // that all n buckets in a normal bucket length have a lower quantile
    // which is P^n where P is the quantile expressed as a probability.
    double lowerBound;
    double upperBound;
    this->quantile(score, CONFIDENCE_INTERVAL, lowerBound, upperBound);
    double lowerPercentile = 100.0 * std::pow(lowerBound, 1.0 / m_BucketNormalizationFactor);
    double upperPercentile = 100.0 * std::pow(upperBound, 1.0 / m_BucketNormalizationFactor);
    if (lowerPercentile > upperPercentile) {
        std::swap(lowerPercentile, upperPercentile);
    }
    lowerPercentile = maths::CTools::truncate(lowerPercentile, 0.0, 100.0);
    upperPercentile = maths::CTools::truncate(upperPercentile, 0.0, 100.0);

    std::size_t lowerKnotPoint =
        std::max(std::lower_bound(m_NormalizedScoreKnotPoints.begin(),
                                  m_NormalizedScoreKnotPoints.end(), lowerPercentile,
                                  maths::COrderings::SFirstLess()) -
                     m_NormalizedScoreKnotPoints.begin(),
                 ptrdiff_t(1));
    std::size_t upperKnotPoint =
        std::max(std::lower_bound(m_NormalizedScoreKnotPoints.begin(),
                                  m_NormalizedScoreKnotPoints.end(), upperPercentile,
                                  maths::COrderings::SFirstLess()) -
                     m_NormalizedScoreKnotPoints.begin(),
                 ptrdiff_t(1));
    if (lowerKnotPoint < m_NormalizedScoreKnotPoints.size()) {
        const TDoubleDoublePr& left = m_NormalizedScoreKnotPoints[lowerKnotPoint - 1];
        const TDoubleDoublePr& right = m_NormalizedScoreKnotPoints[lowerKnotPoint];
        // Linearly interpolate between the two knot points.
        normalizedScores[1] = left.second + (right.second - left.second) *
                                                (lowerPercentile - left.first) /
                                                (right.first - left.first);
    } else {
        normalizedScores[1] = m_MaximumNormalizedScore;
    }
    if (upperKnotPoint < m_NormalizedScoreKnotPoints.size()) {
        const TDoubleDoublePr& left = m_NormalizedScoreKnotPoints[upperKnotPoint - 1];
        const TDoubleDoublePr& right = m_NormalizedScoreKnotPoints[upperKnotPoint];
        // Linearly interpolate between the two knot points.
        normalizedScores[1] = (normalizedScores[1] + left.second +
                               (right.second - left.second) * (upperPercentile - left.first) /
                                   (right.first - left.first)) /
                              2.0;
    } else {
        normalizedScores[1] = (normalizedScores[1] + m_MaximumNormalizedScore) / 2.0;
    }
    LOG_TRACE(<< "normalizedScores[1] = " << normalizedScores[1] << ", lowerBound = " << lowerBound
              << ", upperBound = " << upperBound << ", lowerPercentile = " << lowerPercentile
              << ", upperPercentile = " << upperPercentile);

    double maxScore{0.0};
    bool hasValidMaxScore = this->maxScore(partitionName, partitionValue,
                                           personName, personValue, maxScore);
    if (hasValidMaxScore == false) {
        LOG_TRACE(<< "normalize: Maximum score NOT found for partitionId = "
                  << (partitionName + "_" + partitionValue + "_" + personName + "_" + personValue));
        normalizedScores[2] = m_MaximumNormalizedScore;
    } else {
        LOG_TRACE(<< "normalize: Maximum score FOUND for partitionId = "
                  << (partitionName + "_" + partitionValue + "_" + personName + "_" + personValue)
                  << ". " << maxScore);

        // Compute the maximum score ceiling
        double ratio = score / maxScore;
        double curves[] = {0.0 + 1.5 * ratio, 0.5 + 0.5 * ratio};
        normalizedScores[2] = m_MaximumNormalizedScore *
                              (*std::min_element(curves, curves + 2));
        LOG_TRACE(<< "normalizedScores[2] = " << normalizedScores[2] << ", score = " << score
                  << ", maxScore = " << maxScore << ", ratio = \"" << ratio << "\""
                  << ", partitionId = \""
                  << (partitionName + "_" + partitionValue + "_" + personName + "_" + personValue)
                  << "\"");
    }

    // Logarithmically interpolate the maximum score between the
    // largest significant and small probability.
    static const double M = (probabilityToScore(maths::SMALL_PROBABILITY) -
                             probabilityToScore(maths::LARGEST_SIGNIFICANT_PROBABILITY)) /
                            (std::log(maths::SMALL_PROBABILITY) -
                             std::log(maths::LARGEST_SIGNIFICANT_PROBABILITY));
    static const double C = std::log(maths::LARGEST_SIGNIFICANT_PROBABILITY);
    normalizedScores[3] = m_MaximumNormalizedScore *
                          (0.95 * M * (std::log(scoreToProbability(score)) - C) + 0.05);
    LOG_TRACE(<< "normalizedScores[3] = " << normalizedScores[3] << ", score = " << score
              << ", probability = " << scoreToProbability(score));

    score = std::min(*std::min_element(boost::begin(normalizedScores),
                                       boost::end(normalizedScores)),
                     m_MaximumNormalizedScore);
    LOG_TRACE(<< "normalizedScore = " << score << ", partitionId = \""
              << (partitionName + "_" + partitionValue + "_" + personName + "_" + personValue)
              << "\"");

    return true;
}

bool CAnomalyScore::CNormalizer::maxScore(const std::string& partitionName,
                                          const std::string& partitionValue,
                                          const std::string& personName,
                                          const std::string& personValue,
                                          double& maxScore) const {
    // The influencer named 'bucket_time' corresponds to the root level normalizer for which we have keyed the maximum
    // score on a blank personName
    const std::string person = (personName == TIME_INFLUENCER) ? "" : personName;

    auto itr = m_MaxScores.find(
        m_Dictionary.word(partitionName, partitionValue, person, personValue));
    if (itr == m_MaxScores.end()) {
        return false;
    }
    maxScore = itr->second[0];
    if (maxScore == 0.0) {
        return false;
    }
    return true;
}

void CAnomalyScore::CNormalizer::quantile(double score,
                                          double confidence,
                                          double& lowerBound,
                                          double& upperBound) const {
    uint32_t discreteScore = this->discreteScore(score);
    double n = static_cast<double>(m_RawScoreQuantileSummary.n());
    double lowerQuantile = (100.0 - confidence) / 200.0;
    double upperQuantile = (100.0 + confidence) / 200.0;

    double h = static_cast<double>(m_HighPercentileCount);
    double f = h / n;
    if (!(f >= 0.0 && f <= 1.0)) {
        LOG_ERROR(<< "h = " << h << ", n = " << n);
    }
    double fl = maths::CQDigest::cdfQuantile(n, f, lowerQuantile);
    double fu = maths::CQDigest::cdfQuantile(n, f, upperQuantile);

    if (discreteScore <= m_HighPercentileScore || m_RawScoreHighQuantileSummary.n() == 0) {
        m_RawScoreQuantileSummary.cdf(discreteScore, 0.0, lowerBound, upperBound);

        double pdfLowerBound;
        double pdfUpperBound;
        m_RawScoreQuantileSummary.pdf(discreteScore, 0.0, pdfLowerBound, pdfUpperBound);
        lowerBound = maths::CTools::truncate(lowerBound - pdfUpperBound, 0.0, fl);
        upperBound = maths::CTools::truncate(upperBound - pdfLowerBound, 0.0, fu);
        if (!(lowerBound >= 0.0 && lowerBound <= 1.0) ||
            !(upperBound >= 0.0 && upperBound <= 1.0)) {
            LOG_ERROR(<< "score = " << score << ", cdf = [" << lowerBound << ","
                      << upperBound << "]"
                      << ", pdf = [" << pdfLowerBound << "," << pdfUpperBound << "]");
        }
        lowerBound = maths::CQDigest::cdfQuantile(n, lowerBound, lowerQuantile);
        upperBound = maths::CQDigest::cdfQuantile(n, upperBound, upperQuantile);

        LOG_TRACE(<< "score = " << score << ", cdf = [" << lowerBound << "," << upperBound << "]"
                  << ", pdf = [" << pdfLowerBound << "," << pdfUpperBound << "]");

        return;
    }

    // Note if cutoffUpperBound were ever equal to one to working
    // precision then lowerBound will be zero. The same is true
    // for the cutoffLowerBound and upperBound. In practice both
    // these situations should never happen but we trap them
    // to avoid NaNs in the following calculation.

    m_RawScoreHighQuantileSummary.cdf(discreteScore, 0.0, lowerBound, upperBound);

    double cutoffCdfLowerBound;
    double cutoffCdfUpperBound;
    m_RawScoreHighQuantileSummary.cdf(m_HighPercentileScore, 0.0,
                                      cutoffCdfLowerBound, cutoffCdfUpperBound);

    double pdfLowerBound;
    double pdfUpperBound;
    m_RawScoreHighQuantileSummary.pdf(discreteScore, 0.0, pdfLowerBound, pdfUpperBound);
    lowerBound = fl + (1.0 - fl) *
                          std::max(lowerBound - cutoffCdfUpperBound - pdfUpperBound, 0.0) /
                          std::max(1.0 - cutoffCdfUpperBound,
                                   std::numeric_limits<double>::epsilon());
    upperBound = fu + (1.0 - fu) *
                          std::max(upperBound - cutoffCdfLowerBound - pdfLowerBound, 0.0) /
                          std::max(1.0 - cutoffCdfLowerBound,
                                   std::numeric_limits<double>::epsilon());
    if (!(lowerBound >= 0.0 && lowerBound <= 1.0) ||
        !(upperBound >= 0.0 && upperBound <= 1.0)) {
        LOG_ERROR(<< "score = " << score << ", cdf = [" << lowerBound << "," << upperBound << "]"
                  << ", cutoff = [" << cutoffCdfLowerBound << "," << cutoffCdfUpperBound << "]"
                  << ", pdf = [" << pdfLowerBound << "," << pdfUpperBound << "]"
                  << ", f = " << f);
    }
    lowerBound = maths::CQDigest::cdfQuantile(n, lowerBound, lowerQuantile);
    upperBound = maths::CQDigest::cdfQuantile(n, upperBound, upperQuantile);

    LOG_TRACE(<< "score = " << score << ", cdf = [" << lowerBound << "," << upperBound << "]"
              << ", cutoff = [" << cutoffCdfLowerBound << "," << cutoffCdfUpperBound << "]"
              << ", pdf = [" << pdfLowerBound << "," << pdfUpperBound << "]"
              << ", f = " << f);
}

bool CAnomalyScore::CNormalizer::updateQuantiles(const TDoubleVec& scores,
                                                 const std::string& partitionName,
                                                 const std::string& partitionValue,
                                                 const std::string& personName,
                                                 const std::string& personValue) {
    return this->updateQuantiles(std::accumulate(scores.begin(), scores.end(), 0.0),
                                 partitionName, partitionValue, personName, personValue);
}

bool CAnomalyScore::CNormalizer::updateQuantiles(double score,
                                                 const std::string& partitionName,
                                                 const std::string& partitionValue,
                                                 const std::string& personName,
                                                 const std::string& personValue) {
    using TUInt32UInt64Pr = std::pair<uint32_t, uint64_t>;
    using TUInt32UInt64PrVec = std::vector<TUInt32UInt64Pr>;

    bool bigChange(false);

    TMaxValueAccumulator& maxScore =
        m_MaxScores[m_Dictionary.word(partitionName, partitionValue, personName, personValue)];

    double oldMaxScore(maxScore.count() == 0 ? 0.0 : maxScore[0]);

    maxScore.add(score);

    // BWC with versions < 6.5
    m_MaxScore.add(score);

    LOG_TRACE(<< "updateQuantiles: partitionId = \""
              << (partitionName + "_" + partitionValue + "_" + personName + "_" + personValue)
              << "\", oldMaxScore = " << oldMaxScore << ", score = " << score
              << ", maxScore[0] = " << maxScore[0]);
    if (maxScore[0] > BIG_CHANGE_FACTOR * oldMaxScore) {
        bigChange = true;
        LOG_TRACE(<< "Big change in normalizer - max score updated from "
                  << oldMaxScore << " to " << maxScore[0]);
    }
    uint32_t discreteScore = this->discreteScore(score);
    LOG_TRACE(<< "score = " << score << ", discreteScore = " << discreteScore
              << ", maxScore = " << maxScore[0] << ", partitionId = \""
              << (partitionName + "_" + partitionValue + "_" + personName + "_" + personValue)
              << "\"");

    uint64_t n = m_RawScoreQuantileSummary.n();
    uint64_t k = m_RawScoreQuantileSummary.k();
    LOG_TRACE(<< "n = " << n << ", k = " << k);

    // We are about to compress the q-digest, at the moment it comprises
    // the unique values we have seen so far. So we extract the values
    // greater than the HIGH_PERCENTILE percentile at this point to
    // initialize the fine grain high quantile summary.
    if ((n + 1) == k) {
        LOG_TRACE(<< "Initializing H");

        TUInt32UInt64PrVec L;
        m_RawScoreQuantileSummary.summary(L);
        if (L.empty()) {
            LOG_ERROR(<< "High quantile summary is empty: "
                      << m_RawScoreQuantileSummary.print());
        } else {
            uint64_t highPercentileCount = static_cast<uint64_t>(
                (HIGH_PERCENTILE / 100.0) * static_cast<double>(n) + 0.5);

            // Estimate the high percentile score and update the count.
            std::size_t i = 1u;
            m_HighPercentileScore = L[0].first;
            m_HighPercentileCount = L[0].second;
            for (/**/; i < L.size(); ++i) {
                if (L[i].second > highPercentileCount) {
                    m_HighPercentileScore = L[i - 1].first;
                    m_HighPercentileCount = L[i - 1].second;
                    break;
                }
            }
            if (m_HighPercentileCount > n) {
                LOG_ERROR(<< "Invalid c(H) " << m_HighPercentileCount);
                LOG_ERROR(<< "target " << highPercentileCount);
                LOG_ERROR(<< "L " << core::CContainerPrinter::print(L));
                m_HighPercentileCount = n;
            }
            LOG_TRACE(<< "s(H) = " << m_HighPercentileScore
                      << ", c(H) = " << m_HighPercentileCount << ", percentile = "
                      << 100.0 * static_cast<double>(m_HighPercentileCount) /
                             static_cast<double>(n)
                      << "%"
                      << ", desired c(H) = " << highPercentileCount);

            // Populate the high quantile summary.
            for (/**/; i < L.size(); ++i) {
                uint32_t x = L[i].first;
                uint64_t m = L[i].second - L[i - 1].second;

                LOG_TRACE(<< "Adding (" << x << ", " << m << ") to H");
                m_RawScoreHighQuantileSummary.add(x, m);
            }
        }
    }

    m_RawScoreQuantileSummary.add(discreteScore);

    if (discreteScore <= m_HighPercentileScore) {
        ++m_HighPercentileCount;
    } else {
        m_RawScoreHighQuantileSummary.add(discreteScore);
    }
    LOG_TRACE(<< "percentile = "
              << static_cast<double>(m_HighPercentileCount) / static_cast<double>(n + 1));

    // Periodically refresh the high percentile score.
    if ((n + 1) > k && (n + 1) % k == 0) {
        LOG_TRACE(<< "Refreshing high quantile summary");

        uint64_t highPercentileCount = static_cast<uint64_t>(
            (HIGH_PERCENTILE / 100.0) * static_cast<double>(n + 1) + 0.5);

        LOG_TRACE(<< "s(H) = " << m_HighPercentileScore << ", c(H) = " << m_HighPercentileCount
                  << ", desired c(H) = " << highPercentileCount);

        if (m_HighPercentileCount > highPercentileCount) {
            TUInt32UInt64PrVec L;
            m_RawScoreQuantileSummary.summary(L);
            TUInt32UInt64PrVec H;
            m_RawScoreHighQuantileSummary.summary(H);

            std::size_t i0 = std::min(
                static_cast<std::size_t>(std::lower_bound(L.begin(), L.end(), highPercentileCount,
                                                          maths::COrderings::SSecondLess()) -
                                         L.begin()),
                L.size() - 1);
            std::size_t j = std::min(
                static_cast<std::size_t>(std::upper_bound(H.begin(), H.end(), L[i0],
                                                          maths::COrderings::SFirstLess()) -
                                         H.begin()),
                H.size() - 1);

            uint64_t r = L[i0].second;
            for (std::size_t i = i0 + 1;
                 i < L.size() && L[i0].second + m_RawScoreHighQuantileSummary.n() < n + 1;
                 ++i) {
                for (/**/; j < H.size() && H[j].first <= L[i].first; ++j) {
                    r += (H[j].second -
                          (j == 0 ? static_cast<uint64_t>(0) : H[j - 1].second));
                }

                uint32_t x = L[i].first;
                uint64_t m = r < L[i].second ? L[i].second - r
                                             : static_cast<uint64_t>(0);
                r += m;
                if (m > 0) {
                    LOG_TRACE(<< "Adding (" << x << ',' << m << ") to H");
                    m_RawScoreHighQuantileSummary.add(x, m);
                }
            }

            m_HighPercentileScore = L[i0].first;
            m_HighPercentileCount = L[i0].second;
            if (m_HighPercentileCount > n + 1) {
                LOG_ERROR(<< "Invalid c(H) " << m_HighPercentileCount);
                LOG_ERROR(<< "target " << highPercentileCount);
                LOG_ERROR(<< "L " << core::CContainerPrinter::print(L));
                m_HighPercentileCount = n;
            }

            LOG_TRACE(<< "s(H) = " << m_HighPercentileScore
                      << ", c(H) = " << m_HighPercentileCount << ", percentile = "
                      << 100.0 * static_cast<double>(m_HighPercentileCount) /
                             static_cast<double>(n + 1)
                      << "%");
        } else {
            m_RawScoreQuantileSummary.quantile(HIGH_PERCENTILE / 100.0, m_HighPercentileScore);
            double lowerBound, upperBound;
            m_RawScoreQuantileSummary.cdf(m_HighPercentileScore, 0.0, lowerBound, upperBound);
            m_HighPercentileCount =
                static_cast<uint64_t>(static_cast<double>(n + 1) * lowerBound + 0.5);

            LOG_TRACE(<< "s(H) = " << m_HighPercentileScore << ", c(H) = " << m_HighPercentileCount
                      << ", percentile = " << 100.0 * lowerBound << "%");
        }
    }

    return bigChange;
}

void CAnomalyScore::CNormalizer::propagateForwardByTime(double time) {
    if (time < 0.0) {
        LOG_ERROR(<< "Can't propagate normalizer backwards in time");
        return;
    }

    double alpha = std::exp(-2.0 * m_DecayRate * time);
    for (auto& element : m_MaxScores) {
        element.second.age(alpha);
    }

    // Quantiles age at a much slower rate than everything else so that
    // we can accurately estimate high quantiles. We achieve this by only
    // aging them a certain fraction of the time.
    m_TimeToQuantileDecay -= time;
    if (m_TimeToQuantileDecay <= 0.0) {
        time = std::floor((QUANTILE_DECAY_TIME - m_TimeToQuantileDecay) / QUANTILE_DECAY_TIME);

        uint64_t n = m_RawScoreQuantileSummary.n();
        m_RawScoreQuantileSummary.propagateForwardsByTime(time);
        m_RawScoreHighQuantileSummary.propagateForwardsByTime(time);
        if (n > 0) {
            m_HighPercentileCount = static_cast<uint64_t>(
                static_cast<double>(m_RawScoreQuantileSummary.n()) /
                    static_cast<double>(n) * static_cast<double>(m_HighPercentileCount) +
                0.5);
        }
        m_TimeToQuantileDecay += QUANTILE_DECAY_TIME +
                                 std::floor(-m_TimeToQuantileDecay / QUANTILE_DECAY_TIME);
    }
}

bool CAnomalyScore::CNormalizer::isUpgradable(const std::string& fromVersion,
                                              const std::string& toVersion) {
    // Any changes to this method need to be reflected in the upgrade() method
    // below to prevent an inconsistency where this method says an upgrade is
    // possible but the upgrade() method can't do it.
    return (fromVersion == "1" && toVersion == "2") ||
           (fromVersion == "1" && toVersion == "3") ||
           (fromVersion == "2" && toVersion == "3");
}

bool CAnomalyScore::CNormalizer::upgrade(const std::string& loadedVersion,
                                         const std::string& currentVersion) {
    if (loadedVersion == currentVersion) {
        // No upgrade required.
        return true;
    }

    // We know how to upgrade between versions 1, 2 and 3.
    static const double HIGH_SCORE_UPGRADE_FACTOR[][3] = {
        {1.0, 0.3, 0.3}, {1.0 / 0.3, 1.0, 1.0}, {1.0 / 0.3, 1.0, 1.0}};
    static const double Q_DIGEST_UPGRADE_FACTOR[][3] = {
        {1.0, 3.0, 30.0}, {1.0 / 3.0, 1.0, 10.0}, {1.0 / 30.0, 1.0 / 10.0, 1.0}};

    std::size_t i, j;
    if (!core::CStringUtils::stringToType(loadedVersion, i) ||
        !core::CStringUtils::stringToType(currentVersion, j) ||
        i - 1 >= boost::size(HIGH_SCORE_UPGRADE_FACTOR) ||
        j - 1 >= boost::size(HIGH_SCORE_UPGRADE_FACTOR[0])) {
        LOG_ERROR(<< "Don't know how to upgrade quantiles from version "
                  << loadedVersion << " to version " << currentVersion);
        return false;
    }

    double highScoreUpgradeFactor = HIGH_SCORE_UPGRADE_FACTOR[i - 1][j - 1];
    double qDigestUpgradeFactor = Q_DIGEST_UPGRADE_FACTOR[i - 1][j - 1];

    LOG_INFO(<< "Upgrading quantiles from version " << loadedVersion << " to version "
             << currentVersion << " - will scale highest score by " << highScoreUpgradeFactor
             << " and Q digest min/max values by " << qDigestUpgradeFactor);

    // For the maximum score aging is equivalent to scaling.
    for (auto& element : m_MaxScores) {
        element.second.age(highScoreUpgradeFactor);
    }

    if (m_RawScoreQuantileSummary.scale(qDigestUpgradeFactor) == false) {
        LOG_ERROR(<< "Failed to scale raw score quantiles");
        return false;
    }

    if (m_RawScoreHighQuantileSummary.scale(qDigestUpgradeFactor) == false) {
        LOG_ERROR(<< "Failed to scale raw score high quantiles");
        return false;
    }

    return true;
}

void CAnomalyScore::CNormalizer::clear() {
    m_HighPercentileScore = std::numeric_limits<uint32_t>::max();
    m_HighPercentileCount = 0ull;

    for (auto& element : m_MaxScores) {
        element.second.clear();
    }

    m_RawScoreQuantileSummary.clear();
    m_RawScoreHighQuantileSummary.clear();
    m_TimeToQuantileDecay = QUANTILE_DECAY_TIME;
}

void CAnomalyScore::CNormalizer::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(HIGH_PERCENTILE_SCORE_TAG, m_HighPercentileScore);
    inserter.insertValue(HIGH_PERCENTILE_COUNT_TAG, m_HighPercentileCount);
    inserter.insertValue(MAX_SCORE_TAG, m_MaxScore.toDelimited());
    inserter.insertLevel(RAW_SCORE_QUANTILE_SUMMARY,
                         boost::bind(&maths::CQDigest::acceptPersistInserter,
                                     &m_RawScoreQuantileSummary, _1));
    inserter.insertLevel(RAW_SCORE_HIGH_QUANTILE_SUMMARY,
                         boost::bind(&maths::CQDigest::acceptPersistInserter,
                                     &m_RawScoreHighQuantileSummary, _1));
    inserter.insertValue(TIME_TO_QUANTILE_DECAY_TAG, m_TimeToQuantileDecay);

    core::CPersistUtils::persist(MAX_SCORES_PER_PARTITION_TAG, m_MaxScores, inserter);
}

bool CAnomalyScore::CNormalizer::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();

        // This used to be 64 bit but is now 32 bit, so may need adjusting
        // on restoration
        RESTORE_SETUP_TEARDOWN(
            HIGH_PERCENTILE_SCORE_TAG, uint64_t highPercentileScore64(0),
            core::CStringUtils::stringToType(traverser.value(), highPercentileScore64),
            m_HighPercentileScore = static_cast<uint32_t>(std::min(
                highPercentileScore64,
                static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))));
        RESTORE(HIGH_PERCENTILE_COUNT_TAG,
                core::CStringUtils::stringToType(traverser.value(), m_HighPercentileCount));

        // For BWC continue to restore the old-style job-scope maximum score if present
        RESTORE_NO_ERROR(MAX_SCORE_TAG, m_MaxScore.fromDelimited(traverser.value()));

        RESTORE(RAW_SCORE_QUANTILE_SUMMARY,
                traverser.traverseSubLevel(boost::bind(&maths::CQDigest::acceptRestoreTraverser,
                                                       &m_RawScoreQuantileSummary, _1)));
        RESTORE(RAW_SCORE_HIGH_QUANTILE_SUMMARY,
                traverser.traverseSubLevel(
                    boost::bind(&maths::CQDigest::acceptRestoreTraverser,
                                &m_RawScoreHighQuantileSummary, _1)));

        core::CPersistUtils::restore(MAX_SCORES_PER_PARTITION_TAG, m_MaxScores, traverser);

    } while (traverser.next());

    return true;
}

uint64_t CAnomalyScore::CNormalizer::checksum() const {
    uint64_t seed = static_cast<uint64_t>(m_NoisePercentile);
    seed = maths::CChecksum::calculate(seed, m_NoiseMultiplier);
    seed = maths::CChecksum::calculate(seed, m_NormalizedScoreKnotPoints);
    seed = maths::CChecksum::calculate(seed, m_MaximumNormalizedScore);
    seed = maths::CChecksum::calculate(seed, m_HighPercentileScore);
    seed = maths::CChecksum::calculate(seed, m_HighPercentileCount);
    seed = maths::CChecksum::calculate(seed, m_MaxScores);
    seed = maths::CChecksum::calculate(seed, m_BucketNormalizationFactor);
    seed = maths::CChecksum::calculate(seed, m_RawScoreQuantileSummary);
    seed = maths::CChecksum::calculate(seed, m_RawScoreHighQuantileSummary);
    seed = maths::CChecksum::calculate(seed, m_DecayRate);
    return maths::CChecksum::calculate(seed, m_TimeToQuantileDecay);
}

uint32_t CAnomalyScore::CNormalizer::discreteScore(double rawScore) const {
    return static_cast<uint32_t>(DISCRETIZATION_FACTOR * rawScore + 0.5);
}

double CAnomalyScore::CNormalizer::rawScore(uint32_t discreteScore) const {
    return static_cast<double>(discreteScore) / DISCRETIZATION_FACTOR;
}

const double CAnomalyScore::CNormalizer::DISCRETIZATION_FACTOR = 1000.0;
const double CAnomalyScore::CNormalizer::HIGH_PERCENTILE = 90.0;
const double CAnomalyScore::CNormalizer::QUANTILE_DECAY_TIME = 20.0;
const double CAnomalyScore::CNormalizer::BIG_CHANGE_FACTOR = 1.1;

// We use short field names to reduce the state size
namespace {
const std::string NORMALIZER_TAG("a");
}

const std::string& CAnomalyScore::normalizedScoreToSeverity(double normalizedScore) {
    if (normalizedScore < 25.0) {
        return WARNING_SEVERITY;
    }

    if (normalizedScore < 50.0) {
        return MINOR_SEVERITY;
    }

    if (normalizedScore < 75.0) {
        return MAJOR_SEVERITY;
    }

    return CRITICAL_SEVERITY;
}

bool CAnomalyScore::normalizerFromJson(const std::string& json, CNormalizer& normalizer) {
    std::istringstream iss(json);
    core::CJsonStateRestoreTraverser traverser(iss);

    return normalizerFromJson(traverser, normalizer);
}

bool CAnomalyScore::normalizerFromJson(core::CStateRestoreTraverser& traverser,
                                       CNormalizer& normalizer) {
    bool restoredNormalizer(false);
    std::string restoredVersion(MISSING_VERSION_FORMAT_VERSION);

    while (traverser.next()) {
        const std::string& name = traverser.name();

        if (name == MLVERSION_ATTRIBUTE) {
            restoredVersion = traverser.value();
            if (restoredVersion != CURRENT_FORMAT_VERSION) {
                if (normalizer.isUpgradable(restoredVersion, CURRENT_FORMAT_VERSION)) {
                    LOG_DEBUG(<< "Restored quantiles JSON version is " << restoredVersion
                              << "; current JSON version is " << CURRENT_FORMAT_VERSION
                              << " - will upgrade quantiles");
                } else {
                    // If the version has changed and the format is too different to
                    // even upgrade then start again from scratch - this counts as a
                    // successful load
                    LOG_INFO(<< "Restored quantiles JSON version is " << restoredVersion
                             << "; current JSON version is " << CURRENT_FORMAT_VERSION
                             << " - will restart quantiles from scratch");
                    return true;
                }
            }
        } else if (name == NORMALIZER_TAG) {
            restoredNormalizer = traverser.traverseSubLevel(boost::bind(
                &CAnomalyScore::CNormalizer::acceptRestoreTraverser, &normalizer, _1));
            if (!restoredNormalizer) {
                LOG_ERROR(<< "Unable to restore quantiles to the normaliser");
            }
        }
    }

    if (restoredNormalizer && restoredVersion != CURRENT_FORMAT_VERSION) {
        LOG_INFO(<< "Restored quantiles JSON version is " << restoredVersion << "; current JSON version is "
                 << CURRENT_FORMAT_VERSION << " - will attempt upgrade");

        if (normalizer.upgrade(restoredVersion, CURRENT_FORMAT_VERSION) == false) {
            LOG_ERROR(<< "Failed to upgrade quantiles from version " << restoredVersion
                      << " to version " << CURRENT_FORMAT_VERSION);
            return false;
        }
    }

    return restoredNormalizer;
}

void CAnomalyScore::normalizerToJson(const CNormalizer& normalizer,
                                     const std::string& searchKey,
                                     const std::string& cue,
                                     const std::string& description,
                                     core_t::TTime time,
                                     std::string& json) {
    std::ostringstream ss;

    // The JSON inserter will only close the object when it is destroyed
    {
        core::CJsonStatePersistInserter inserter(ss);

        // Important: Even though some of these fields appear to be unnecessary,
        // the CHierarchicalResultsNormalizer requires them to exist in this
        // order.  Do not change the fields or the ordering.
        inserter.insertValue(MLCUE_ATTRIBUTE, cue);
        inserter.insertValue(MLKEY_ATTRIBUTE, searchKey);
        inserter.insertValue(MLQUANTILESDESCRIPTION_ATTRIBUTE, description);
        inserter.insertValue(MLVERSION_ATTRIBUTE, CURRENT_FORMAT_VERSION);
        inserter.insertValue(TIME_ATTRIBUTE, core::CStringUtils::typeToString(time));

        inserter.insertLevel(NORMALIZER_TAG, boost::bind(&CNormalizer::acceptPersistInserter,
                                                         &normalizer, _1));
    }

    json = ss.str();
}
}
}
