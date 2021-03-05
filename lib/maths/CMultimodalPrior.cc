/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CMultimodalPrior.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CSmallVector.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/Constants.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/CClusterer.h>
#include <maths/CClustererStateSerialiser.h>
#include <maths/CKMeansOnline1d.h>
#include <maths/CMathsFuncs.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/CPriorStateSerialiser.h>
#include <maths/CRestoreParams.h>
#include <maths/CSampling.h>
#include <maths/CSetTools.h>
#include <maths/CSolvers.h>
#include <maths/CTools.h>
#include <maths/CToolsDetail.h>
#include <maths/MathsTypes.h>
#include <maths/ProbabilityAggregators.h>

#include <cmath>
#include <limits>
#include <set>

namespace ml {
namespace maths {

namespace {

using TSizeDoublePr = std::pair<std::size_t, double>;
using TSizeDoublePr2Vec = core::CSmallVector<TSizeDoublePr, 2>;
using TSizeDoublePr5Vec = core::CSmallVector<TSizeDoublePr, 5>;
using TDoubleDoublePr = std::pair<double, double>;
using TSizeSet = std::set<std::size_t>;
using TDouble1Vec = core::CSmallVector<double, 1>;
using maths_t::TDoubleWeightsAry;
using maths_t::TDoubleWeightsAry1Vec;
using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

//! \brief Wrapper to call the -log(c.d.f) of a prior object.
class CMinusLogJointCdf {
public:
    template<typename T>
    bool operator()(const T& prior,
                    const TDouble1Vec& samples,
                    const TDoubleWeightsAry1Vec& weights,
                    double& lowerBound,
                    double& upperBound) const {
        return prior->minusLogJointCdf(samples, weights, lowerBound, upperBound);
    }
};

//! \brief Wrapper to call the log(1 - c.d.f) of a prior object.
class CMinusLogJointCdfComplement {
public:
    template<typename T>
    bool operator()(const T& prior,
                    const TDouble1Vec& samples,
                    const TDoubleWeightsAry1Vec& weights,
                    double& lowerBound,
                    double& upperBound) const {
        return prior->minusLogJointCdfComplement(samples, weights, lowerBound, upperBound);
    }
};

//! \brief Wrapper of CMultimodalPrior::minusLogJointCdf function
//! for use with our solver.
class CLogCdf {
public:
    using result_type = double;

    enum EStyle { E_Lower, E_Upper, E_Mean };

public:
    CLogCdf(EStyle style, const CMultimodalPrior& prior, const TDoubleWeightsAry& weights)
        : m_Style(style), m_Prior(&prior), m_Weights(1, weights), m_X(1u, 0.0) {}

    double operator()(double x) const {
        m_X[0] = x;
        double lowerBound, upperBound;
        if (m_Prior->minusLogJointCdf(m_X, m_Weights, lowerBound, upperBound) == false) {
            throw std::runtime_error("Unable to compute c.d.f. at " +
                                     core::CStringUtils::typeToString(x));
        }
        switch (m_Style) {
        case E_Lower:
            return -lowerBound;
        case E_Upper:
            return -upperBound;
        case E_Mean:
            return -(lowerBound + upperBound) / 2.0;
        }
        return -(lowerBound + upperBound) / 2.0;
    }

private:
    EStyle m_Style;
    const CMultimodalPrior* m_Prior;
    TDoubleWeightsAry1Vec m_Weights;
    //! Avoids creating the vector argument to minusLogJointCdf
    //! more than once.
    mutable TDouble1Vec m_X;
};

const std::size_t MODE_SPLIT_NUMBER_SAMPLES(50u);
const std::size_t MODE_MERGE_NUMBER_SAMPLES(25u);

const core::TPersistenceTag CLUSTERER_TAG("a", "clusterer");
const core::TPersistenceTag SEED_PRIOR_TAG("b", "seed_prior");
const core::TPersistenceTag MODE_TAG("c", "mode");
const core::TPersistenceTag NUMBER_SAMPLES_TAG("d", "number_samples");
//const std::string MINIMUM_TAG("e"); No longer used
//const std::string MAXIMUM_TAG("f"); No longer used
const core::TPersistenceTag DECAY_RATE_TAG("g", "decay_rate");

const std::string EMPTY_STRING;
}

//////// CMultimodalPrior Implementation ////////

CMultimodalPrior::CMultimodalPrior(maths_t::EDataType dataType,
                                   const CClusterer1d& clusterer,
                                   const CPrior& seedPrior,
                                   double decayRate /*= 0.0*/)
    : CPrior(dataType, decayRate), m_Clusterer(clusterer.clone()),
      m_SeedPrior(seedPrior.clone()) {
    // Register the split and merge callbacks.
    m_Clusterer->splitFunc(CModeSplitCallback(*this));
    m_Clusterer->mergeFunc(CModeMergeCallback(*this));
}

CMultimodalPrior::CMultimodalPrior(maths_t::EDataType dataType,
                                   const TMeanVarAccumulatorVec& moments,
                                   double decayRate /*= 0.0*/)
    : CPrior(dataType, decayRate),
      m_SeedPrior(
          CNormalMeanPrecConjugate::nonInformativePrior(dataType, decayRate).clone()) {
    using TNormalVec = std::vector<CNormalMeanPrecConjugate>;

    TNormalVec normals;
    normals.reserve(moments.size());
    for (const auto& moments_ : moments) {
        normals.emplace_back(dataType, moments_, decayRate);
    }

    m_Clusterer = std::make_unique<CKMeansOnline1d>(normals);

    m_Modes.reserve(normals.size());
    for (std::size_t i = 0; i < normals.size(); ++i) {
        m_Modes.emplace_back(i, TPriorPtr(normals.back().clone()));
    }
}

CMultimodalPrior::CMultimodalPrior(maths_t::EDataType dataType, double decayRate, TPriorPtrVec& priors)
    : CPrior(dataType, decayRate) {
    m_Modes.reserve(priors.size());
    for (std::size_t i = 0; i < priors.size(); ++i) {
        m_Modes.emplace_back(i, std::move(priors[i]));
    }
}

CMultimodalPrior::CMultimodalPrior(const SDistributionRestoreParams& params,
                                   core::CStateRestoreTraverser& traverser)
    : CPrior(params.s_DataType, params.s_DecayRate) {
    traverser.traverseSubLevel(std::bind(&CMultimodalPrior::acceptRestoreTraverser, this,
                                         std::cref(params), std::placeholders::_1));
}

bool CMultimodalPrior::acceptRestoreTraverser(const SDistributionRestoreParams& params,
                                              core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE_SETUP_TEARDOWN(DECAY_RATE_TAG, double decayRate,
                               core::CStringUtils::stringToType(traverser.value(), decayRate),
                               this->decayRate(decayRate))
        RESTORE(CLUSTERER_TAG, traverser.traverseSubLevel(std::bind<bool>(
                                   CClustererStateSerialiser(), std::cref(params),
                                   std::ref(m_Clusterer), std::placeholders::_1)))
        RESTORE(SEED_PRIOR_TAG, traverser.traverseSubLevel(std::bind<bool>(
                                    CPriorStateSerialiser(), std::cref(params),
                                    std::ref(m_SeedPrior), std::placeholders::_1)))
        RESTORE_SETUP_TEARDOWN(MODE_TAG, TMode mode,
                               traverser.traverseSubLevel(std::bind(
                                   &TMode::acceptRestoreTraverser, &mode,
                                   std::cref(params), std::placeholders::_1)),
                               m_Modes.push_back(std::move(mode)))
        RESTORE_SETUP_TEARDOWN(NUMBER_SAMPLES_TAG, double numberSamples,
                               core::CStringUtils::stringToType(traverser.value(), numberSamples),
                               this->numberSamples(numberSamples))
    } while (traverser.next());

    if (m_Clusterer != nullptr) {
        // Register the split and merge callbacks.
        m_Clusterer->splitFunc(CModeSplitCallback(*this));
        m_Clusterer->mergeFunc(CModeMergeCallback(*this));
    }

    this->checkRestoredInvariants();

    return true;
}

void CMultimodalPrior::checkRestoredInvariants() const {
    VIOLATES_INVARIANT_NO_EVALUATION(m_SeedPrior, ==, nullptr);
}

CMultimodalPrior::CMultimodalPrior(const CMultimodalPrior& other)
    : CPrior(other.dataType(), other.decayRate()),
      m_Clusterer(other.m_Clusterer != nullptr ? other.m_Clusterer->clone() : nullptr),
      m_SeedPrior(other.m_SeedPrior != nullptr ? other.m_SeedPrior->clone() : nullptr) {
    // Register the split and merge callbacks.
    if (m_Clusterer != nullptr) {
        m_Clusterer->splitFunc(CModeSplitCallback(*this));
        m_Clusterer->mergeFunc(CModeMergeCallback(*this));
    }

    // Clone all the modes up front so we can implement strong exception safety.
    TModeVec modes;
    modes.reserve(other.m_Modes.size());
    for (const auto& mode : other.m_Modes) {
        modes.emplace_back(mode.s_Index, TPriorPtr(mode.s_Prior->clone()));
    }
    m_Modes.swap(modes);

    this->addSamples(other.numberSamples());
}

CMultimodalPrior& CMultimodalPrior::operator=(const CMultimodalPrior& rhs) {
    if (this != &rhs) {
        CMultimodalPrior copy(rhs);
        this->swap(copy);
    }
    return *this;
}

void CMultimodalPrior::swap(CMultimodalPrior& other) {
    this->CPrior::swap(other);

    std::swap(m_Clusterer, other.m_Clusterer);
    // The call backs for split and merge should point to the
    // appropriate priors (we don't swap the "this" pointers
    // after all). So we need to refresh them after swapping.
    if (m_Clusterer != nullptr) {
        m_Clusterer->splitFunc(CModeSplitCallback(*this));
        m_Clusterer->mergeFunc(CModeMergeCallback(*this));
    }
    if (other.m_Clusterer != nullptr) {
        other.m_Clusterer->splitFunc(CModeSplitCallback(other));
        other.m_Clusterer->mergeFunc(CModeMergeCallback(other));
    }

    std::swap(m_SeedPrior, other.m_SeedPrior);
    m_Modes.swap(other.m_Modes);
}

CMultimodalPrior::EPrior CMultimodalPrior::type() const {
    return E_Multimodal;
}

CMultimodalPrior* CMultimodalPrior::clone() const {
    return new CMultimodalPrior(*this);
}

void CMultimodalPrior::dataType(maths_t::EDataType value) {
    this->CPrior::dataType(value);
    m_Clusterer->dataType(value);
    for (const auto& mode : m_Modes) {
        mode.s_Prior->dataType(value);
    }
}

void CMultimodalPrior::decayRate(double value) {
    this->CPrior::decayRate(value);
    m_Clusterer->decayRate(value);
    for (const auto& mode : m_Modes) {
        mode.s_Prior->decayRate(value);
    }
    m_SeedPrior->decayRate(value);
}

void CMultimodalPrior::setToNonInformative(double /*offset*/, double decayRate) {
    m_Clusterer->clear();
    m_Modes.clear();
    this->decayRate(decayRate);
    this->numberSamples(0.0);
}

bool CMultimodalPrior::needsOffset() const {
    for (const auto& mode : m_Modes) {
        if (mode.s_Prior->needsOffset()) {
            return true;
        }
    }
    return false;
}

double CMultimodalPrior::adjustOffset(const TDouble1Vec& samples,
                                      const TDoubleWeightsAry1Vec& weights) {
    double result{0.0};
    if (this->needsOffset()) {
        TSizeDoublePr2Vec clusters;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            m_Clusterer->cluster(samples[i], clusters);
            for (const auto& cluster : clusters) {
                auto j = std::find_if(m_Modes.begin(), m_Modes.end(),
                                      CSetTools::CIndexInSet(cluster.first));
                if (j != m_Modes.end()) {
                    result += j->s_Prior->adjustOffset({samples[i]}, {weights[i]});
                }
            }
        }
    }
    return result;
}

double CMultimodalPrior::offset() const {
    double offset{0.0};
    for (const auto& mode : m_Modes) {
        offset = std::max(offset, mode.s_Prior->offset());
    }
    return offset;
}

void CMultimodalPrior::addSamples(const TDouble1Vec& samples,
                                  const TDoubleWeightsAry1Vec& weights) {
    if (samples.empty()) {
        return;
    }
    if (samples.size() != weights.size()) {
        LOG_ERROR(<< "Mismatch in samples '"
                  << core::CContainerPrinter::print(samples) << "' and weights '"
                  << core::CContainerPrinter::print(weights) << "'");
        return;
    }

    this->adjustOffset(samples, weights);

    // This uses a clustering methodology (defined by m_Clusterer)
    // to assign each sample to a cluster. Each cluster has its own
    // mode in the distribution, which comprises a weight and prior
    // which are updated based on the assignment.
    //
    // The idea in separating clustering from modeling is to enable
    // different clustering methodologies to be used as appropriate
    // to the particular data set under consideration and based on
    // the runtime available.
    //
    // It is intended that approximate Bayesian schemes for multimode
    // distributions, such as the variational treatment of a mixture
    // of Gaussians, are implemented as separate priors.
    //
    // Flow through this function is as follows:
    //   1) Assign the samples to clusters.
    //   2) Find or create corresponding modes as necessary.
    //   3) Update the mode's weight and prior with the samples
    //      assigned to it.

    // Declared outside the loop to minimize the number of times it
    // is initialized.
    TDouble1Vec sample(1);
    TDoubleWeightsAry1Vec weight(1);
    TSizeDoublePr2Vec clusters;

    try {
        bool hasSeasonalScale{this->isNonInformative() == false &&
                              maths_t::hasSeasonalVarianceScale(weights)};
        double mean{hasSeasonalScale ? this->marginalLikelihoodMean() : 0.0};

        for (std::size_t i = 0; i < samples.size(); ++i) {
            double x{samples[i]};
            if (CMathsFuncs::isFinite(x) == false) {
                LOG_ERROR(<< "Discarding " << x);
                continue;
            }
            if (hasSeasonalScale) {
                x = mean + (x - mean) /
                               std::sqrt(maths_t::seasonalVarianceScale(weights[i]));
            }

            sample[0] = x;
            weight[0] = weights[i];
            maths_t::setSeasonalVarianceScale(1.0, weight[0]);

            clusters.clear();
            m_Clusterer->add(x, clusters, maths_t::count(weight[0]));

            double Z{std::accumulate(
                m_Modes.begin(), m_Modes.end(), maths_t::count(weight[0]),
                [](double sum, const TMode& mode) { return sum + mode.weight(); })};

            double n{0.0};
            for (const auto& cluster : clusters) {
                auto k = std::find_if(m_Modes.begin(), m_Modes.end(),
                                      CSetTools::CIndexInSet(cluster.first));
                if (k == m_Modes.end()) {
                    LOG_TRACE(<< "Creating mode with index " << cluster.first);
                    m_Modes.emplace_back(cluster.first, m_SeedPrior);
                    k = m_Modes.end() - 1;
                }
                maths_t::setCount(cluster.second, weight[0]);
                if (maths_t::isWinsorised(weight)) {
                    double ww = maths_t::winsorisationWeight(weight[0]);
                    double f = (k->weight() + cluster.second) / Z;
                    maths_t::setWinsorisationWeight(
                        std::max(1.0 - (1.0 - ww) / f, ww * f), weight[0]);
                }
                k->s_Prior->addSamples(sample, weight);
                n += maths_t::countForUpdate(weight[0]);
            }
            this->addSamples(n);
        }
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to update likelihood: " << e.what());
    }
}

void CMultimodalPrior::propagateForwardsByTime(double time) {
    if (CMathsFuncs::isFinite(time) == false || time < 0.0) {
        LOG_ERROR(<< "Bad propagation time " << time);
        return;
    }
    if (this->isNonInformative()) {
        // Nothing to be done.
        return;
    }

    // We want to hold the probabilities constant. Since the i'th
    // probability:
    //   p(i) = w(i) / Sum_j{ w(j) }
    //
    // where w(i) is its weight we can achieve this by multiplying
    // all weights by some factor f in the range [0, 1].

    m_Clusterer->propagateForwardsByTime(time);
    for (const auto& mode : m_Modes) {
        mode.s_Prior->propagateForwardsByTime(time);
    }

    // Remove any mode which is non-informative.
    while (m_Modes.size() > 1) {
        // Calling remove with the mode's index triggers a callback
        // which also removes it from s_Modes, see CModeMergeCallback.
        auto i = std::find_if(m_Modes.begin(), m_Modes.end(), [](const auto& mode) {
            return mode.s_Prior->isNonInformative();
        });
        if (i == m_Modes.end() || m_Clusterer->remove(i->s_Index) == false) {
            break;
        }
    }

    this->numberSamples(this->numberSamples() * std::exp(-this->decayRate() * time));
    LOG_TRACE(<< "numberSamples = " << this->numberSamples());
}

TDoubleDoublePr CMultimodalPrior::marginalLikelihoodSupport() const {
    if (m_Modes.size() == 0) {
        return {-std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
    }
    if (m_Modes.size() == 1) {
        return m_Modes[0].s_Prior->marginalLikelihoodSupport();
    }

    TDoubleDoublePr result{-std::numeric_limits<double>::max(),
                           std::numeric_limits<double>::max()};

    // We define this is as the union of the mode supports.
    for (const auto& mode : m_Modes) {
        TDoubleDoublePr s{mode.s_Prior->marginalLikelihoodSupport()};
        result.first = std::min(result.first, s.first);
        result.second = std::max(result.second, s.second);
    }

    return result;
}

double CMultimodalPrior::marginalLikelihoodMean() const {
    if (m_Modes.size() == 0) {
        return 0.0;
    }
    if (m_Modes.size() == 1) {
        return m_Modes[0].s_Prior->marginalLikelihoodMean();
    }

    // By linearity we have that:
    //   Integral{ x * Sum_i{ w(i) * f(x | i) } }
    //     = Sum_i{ w(i) * Integral{ x * f(x | i) } }
    //     = Sum_i{ w(i) * mean(i) }

    TMeanAccumulator result;
    for (const auto& mode : m_Modes) {
        result.add(mode.s_Prior->marginalLikelihoodMean(), mode.weight());
    }
    return CBasicStatistics::mean(result);
}

double CMultimodalPrior::nearestMarginalLikelihoodMean(double value) const {

    if (m_Modes.empty()) {
        return 0.0;
    }

    double mean{m_Modes[0].s_Prior->marginalLikelihoodMean()};
    double distance{std::fabs(value - mean)};
    double result{mean};
    for (std::size_t i = 1; i < m_Modes.size(); ++i) {
        mean = m_Modes[i].s_Prior->marginalLikelihoodMean();
        if (std::fabs(value - mean) < distance) {
            distance = std::fabs(value - mean);
            result = mean;
        }
    }
    return result;
}

double CMultimodalPrior::marginalLikelihoodMode(const TDoubleWeightsAry& weights) const {
    if (m_Modes.size() == 0) {
        return 0.0;
    }
    if (m_Modes.size() == 1) {
        return m_Modes[0].s_Prior->marginalLikelihoodMode(weights);
    }

    using TMaxAccumulator = CBasicStatistics::SMax<double>::TAccumulator;

    // We'll approximate this as the maximum likelihood mode (mode).
    double result{0.0};

    double seasonalScale{std::sqrt(maths_t::seasonalVarianceScale(weights))};
    double countVarianceScale{maths_t::countVarianceScale(weights)};

    // Declared outside the loop to minimize number of times they
    // are created.
    TDouble1Vec distributionMode(1);
    TDoubleWeightsAry1Vec weight{maths_t::countVarianceScaleWeight(countVarianceScale)};

    TMaxAccumulator maxLikelihood;
    for (const auto& mode : m_Modes) {
        double w{mode.weight()};
        const auto& prior = mode.s_Prior;
        distributionMode[0] = prior->marginalLikelihoodMode(weight[0]);
        double likelihood;
        if (prior->jointLogMarginalLikelihood(distributionMode, weight, likelihood) &
            (maths_t::E_FpFailed | maths_t::E_FpOverflowed)) {
            continue;
        }
        if (maxLikelihood.add(std::log(w) + likelihood)) {
            result = distributionMode[0];
        }
    }

    if (maths_t::hasSeasonalVarianceScale(weights)) {
        double mean{this->marginalLikelihoodMean()};
        result = mean + seasonalScale * (result - mean);
    }

    return result;
}

CMultimodalPrior::TDouble1Vec
CMultimodalPrior::marginalLikelihoodModes(const TDoubleWeightsAry& weights) const {
    TDouble1Vec result(m_Modes.size());
    for (std::size_t i = 0; i < m_Modes.size(); ++i) {
        result[i] = m_Modes[i].s_Prior->marginalLikelihoodMode(weights);
    }
    return result;
}

double CMultimodalPrior::marginalLikelihoodVariance(const TDoubleWeightsAry& weights) const {
    if (m_Modes.size() == 0) {
        return std::numeric_limits<double>::max();
    }
    if (m_Modes.size() == 1) {
        return m_Modes[0].s_Prior->marginalLikelihoodVariance(weights);
    }

    // By linearity we have that:
    //   Integral{ (x - m)^2 * Sum_i{ w(i) * f(x | i) } }
    //     = Sum_i{ w(i) * (Integral{ x^2 * f(x | i) } - m^2) }
    //     = Sum_i{ w(i) * ((mi^2 + vi) - m^2) }

    double varianceScale{maths_t::seasonalVarianceScale(weights) *
                         maths_t::countVarianceScale(weights)};
    double mean{this->marginalLikelihoodMean()};

    TMeanAccumulator result;
    for (const auto& mode : m_Modes) {
        double w{mode.weight()};
        double mm{mode.s_Prior->marginalLikelihoodMean()};
        double mv{mode.s_Prior->marginalLikelihoodVariance()};
        result.add((mm - mean) * (mm + mean) + mv, w);
    }

    return std::max(varianceScale * CBasicStatistics::mean(result), 0.0);
}

TDoubleDoublePr
CMultimodalPrior::marginalLikelihoodConfidenceInterval(double percentage,
                                                       const TDoubleWeightsAry& weights) const {

    TDoubleDoublePr support{this->marginalLikelihoodSupport()};

    if (this->isNonInformative()) {
        return support;
    }

    if (m_Modes.size() == 1) {
        return m_Modes[0].s_Prior->marginalLikelihoodConfidenceInterval(percentage, weights);
    }

    percentage = CTools::truncate(percentage, 0.0, 100.0);
    if (percentage == 100.0) {
        return support;
    }

    using TMinMaxAccumulator = CBasicStatistics::CMinMax<double>;

    // The multimodal distribution confidence interval must lie between the
    // minimum and maximum of the corresponding modes' confidence intervals.
    TMinMaxAccumulator lower;
    TMinMaxAccumulator upper;
    for (const auto& mode : m_Modes) {
        auto interval = mode.s_Prior->marginalLikelihoodConfidenceInterval(percentage, weights);
        lower.add(interval.first);
        upper.add(interval.second);
    }

    percentage /= 100.0;
    double p1{maths_t::count(weights) * std::log((1.0 - percentage) / 2.0)};
    double p2{maths_t::count(weights) * std::log((1.0 + percentage) / 2.0)};

    double vs{maths_t::seasonalVarianceScale(weights)};
    double mean{vs != 1.0 ? this->marginalLikelihoodMean() : 0.0};

    CLogCdf fl{CLogCdf::E_Lower, *this, weights};
    CLogCdf fu{CLogCdf::E_Upper, *this, weights};

    auto computePercentile = [&](const CLogCdf& f, double p, const TMinMaxAccumulator& bounds) {
        auto fMinusp = [&f, p](double x) { return f(x) - p; };
        double a{bounds.min() < mean
                     ? mean - std::max(std::sqrt(vs), 1.0) * (mean - bounds.min())
                     : bounds.min()};
        double b{bounds.max() > mean
                     ? mean + std::max(std::sqrt(vs), 1.0) * (bounds.max() - mean)
                     : bounds.max()};
        double fa{fMinusp(a)};
        double fb{fMinusp(b)};
        // If we enter either of the following loops, since the c.d.f. is
        // monotonic, we know that the interval is either right or left of
        // the required root. We can therefore shift the original interval
        // since a and b are, respectively, upper and lower bounds for the
        // root. Each iteration also doubles the original interval width.
        // Note, we can only enter one or other of these loops so provided
        // we supply a few more than 10 iterations to the solver it'll have
        // enough iterations to comfortably converge.
        for (std::size_t i = 0; fa > 0.0 && i < 10; ++i) {
            std::tie(a, b) = std::make_pair(b - 3.0 * (b - a), a);
            std::tie(fa, fb) = std::make_pair(fMinusp(a), fa);
        }
        for (std::size_t i = 0; fb < 0.0 && i < 10; ++i) {
            std::tie(a, b) = std::make_pair(b, a + 3.0 * (b - a));
            std::tie(fa, fb) = std::make_pair(fb, fMinusp(b));
        }
        std::size_t maxIterations{25};
        CEqualWithTolerance<double> equal{CToleranceTypes::E_AbsoluteTolerance,
                                          std::min(std::numeric_limits<double>::epsilon() * b,
                                                   1e-3 * p / std::max(fa, fb))};
        double percentile;
        CSolvers::solve(a, b, fa, fb, fMinusp, maxIterations, equal, percentile);
        LOG_TRACE(<< "p1 = " << p << ", x = " << percentile << ", f(x) = " << f(percentile)
                  << " brackets = [" << a << "," << b << "]");
        return percentile;
    };

    TDoubleDoublePr result{mean - std::sqrt(vs) * (mean - lower.min()),
                           mean + std::sqrt(vs) * (upper.max() - mean)};
    try {
        result.first = computePercentile(fl, p1, lower);
        result.second = computePercentile(fu, p2, upper);
    } catch (const std::exception& e) {
        LOG_WARN(<< "Unable to compute percentiles: " << e.what()
                 << ", percentiles = [" << p1 << "," << p2 << "] "
                 << ", vs = " << vs);
    }
    return result;
}

maths_t::EFloatingPointErrorStatus
CMultimodalPrior::jointLogMarginalLikelihood(const TDouble1Vec& samples,
                                             const TDoubleWeightsAry1Vec& weights,
                                             double& result) const {

    result = 0.0;

    if (samples.empty()) {
        LOG_ERROR(<< "Can't compute likelihood for empty sample set");
        return maths_t::E_FpFailed;
    }
    if (samples.size() != weights.size()) {
        LOG_ERROR(<< "Mismatch in samples '"
                  << core::CContainerPrinter::print(samples) << "' and weights '"
                  << core::CContainerPrinter::print(weights) << "'");
        return maths_t::E_FpFailed;
    }
    if (this->isNonInformative()) {
        // The non-informative likelihood is improper and effectively
        // zero everywhere. We use minus max double because
        // log(0) = HUGE_VALUE, which causes problems for Windows.
        // Calling code is notified when the calculation overflows
        // and should avoid taking the exponential since this will
        // underflow and pollute the floating point environment. This
        // may cause issues for some library function implementations
        // (see fe*exceptflag for more details).
        result = -std::numeric_limits<double>::max();
        return maths_t::E_FpOverflowed;
    }

    if (m_Modes.size() == 1) {
        return m_Modes[0].s_Prior->jointLogMarginalLikelihood(samples, weights, result);
    }

    // The likelihood can be computed from the conditional likelihood
    // that a sample is from each mode. In particular, the likelihood
    // of a sample x is:
    //   L(x) = Sum_m{ L(x | m) * p(m) }
    //
    // where,
    //   L(x | m) is the likelihood the sample is from the m'th mode,
    //   p(m) is the probability a sample is from the m'th mode.
    //
    // We compute the combined likelihood by taking the product of the
    // individual likelihoods. Note, this brushes over the fact that the
    // joint marginal likelihood that a collection of samples is from
    // the i'th mode is not just the product of the likelihoods that the
    // individual samples are from the i'th mode since we're integrating
    // over a prior. Really, we should compute likelihoods over all
    // possible assignments of the samples to the modes and use the fact
    // that:
    //   P(a) = Product_i{ Sum_m{ p(m) * I{a(i) = m} } }
    //
    // where,
    //   P(a) is the probability of a given assignment,
    //   p(m) is the probability a sample is from the m'th mode,
    //   I{.} is the indicator function.
    //
    // The approximation is increasingly accurate as the prior distribution
    // on each mode narrows.

    result = 0.0;

    // Declared outside the loop to minimize number of times it is created.
    TDouble1Vec sample(1);
    TSizeDoublePr5Vec modeLogLikelihoods;
    modeLogLikelihoods.reserve(m_Modes.size());

    double mean{maths_t::hasSeasonalVarianceScale(weights) ? this->marginalLikelihoodMean()
                                                           : 0.0};
    TDoubleWeightsAry1Vec weight{TWeights::UNIT};
    try {
        for (std::size_t i = 0; i < samples.size(); ++i) {
            double n{maths_t::countForUpdate(weights[i])};
            double seasonalScale{std::sqrt(maths_t::seasonalVarianceScale(weights[i]))};
            double logSeasonalScale{seasonalScale != 1.0 ? std::log(seasonalScale) : 0.0};

            sample[0] = mean + (samples[i] - mean) / seasonalScale;
            maths_t::setCountVarianceScale(maths_t::countVarianceScale(weights[i]),
                                           weight[0]);

            // We re-normalize so that the maximum log likelihood is one
            // to avoid underflow.
            modeLogLikelihoods.clear();
            double maxLogLikelihood{-std::numeric_limits<double>::max()};

            for (std::size_t j = 0; j < m_Modes.size(); ++j) {
                double modeLogLikelihood;
                maths_t::EFloatingPointErrorStatus status{m_Modes[j].s_Prior->jointLogMarginalLikelihood(
                    sample, weight, modeLogLikelihood)};
                if (status & maths_t::E_FpFailed) {
                    // Logging handled at a lower level.
                    return status;
                }
                if ((status & maths_t::E_FpOverflowed) == false) {
                    modeLogLikelihoods.emplace_back(j, modeLogLikelihood);
                    maxLogLikelihood = std::max(maxLogLikelihood, modeLogLikelihood);
                }
            }

            if (modeLogLikelihoods.empty()) {
                // Technically, the marginal likelihood is zero here
                // so the log would be infinite. We use minus max
                // double because log(0) = HUGE_VALUE, which causes
                // problems for Windows. Calling code is notified
                // when the calculation overflows and should avoid
                // taking the exponential since this will underflow
                // and pollute the floating point environment. This
                // may cause issues for some library function
                // implementations (see fe*exceptflag for more details).
                result = -std::numeric_limits<double>::max();
                return maths_t::E_FpOverflowed;
            }

            LOG_TRACE(<< "modeLogLikelihoods = "
                      << core::CContainerPrinter::print(modeLogLikelihoods));

            double sampleLikelihood{0.0};
            double Z{0.0};

            for (const auto& modeLogLikelihood : modeLogLikelihoods) {
                double w{m_Modes[modeLogLikelihood.first].weight()};
                // Divide through by the largest value to avoid underflow.
                sampleLikelihood += w * std::exp(modeLogLikelihood.second - maxLogLikelihood);
                Z += w;
            }

            sampleLikelihood /= Z;
            double sampleLogLikelihood{n * (std::log(sampleLikelihood) + maxLogLikelihood)};

            LOG_TRACE(<< "sample = " << core::CContainerPrinter::print(sample)
                      << ", maxLogLikelihood = " << maxLogLikelihood
                      << ", sampleLogLikelihood = " << sampleLogLikelihood);

            result += sampleLogLikelihood - n * logSeasonalScale;
        }
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to compute likelihood: " << e.what());
        return maths_t::E_FpFailed;
    }

    maths_t::EFloatingPointErrorStatus status{CMathsFuncs::fpStatus(result)};
    if (status & maths_t::E_FpFailed) {
        LOG_ERROR(<< "Failed to compute likelihood (" << this->debugWeights() << ")");
        LOG_ERROR(<< "samples = " << core::CContainerPrinter::print(samples));
        LOG_ERROR(<< "weights = " << core::CContainerPrinter::print(weights));
    }
    LOG_TRACE(<< "Joint log likelihood = " << result);
    return status;
}

void CMultimodalPrior::sampleMarginalLikelihood(std::size_t numberSamples,
                                                TDouble1Vec& samples) const {

    samples.clear();

    if (numberSamples == 0 || this->numberSamples() == 0.0) {
        return;
    }

    samples.clear();

    if (m_Modes.size() == 1) {
        m_Modes[0].s_Prior->sampleMarginalLikelihood(numberSamples, samples);
        return;
    }

    // We sample each mode according to its weight.

    TDoubleVec normalizedWeights;
    normalizedWeights.reserve(m_Modes.size());
    double Z{0.0};

    for (const auto& mode : m_Modes) {
        double weight{mode.weight()};
        normalizedWeights.push_back(weight);
        Z += weight;
    }
    for (auto& weight : normalizedWeights) {
        weight /= Z;
    }

    CSampling::TSizeVec sampling;
    CSampling::weightedSample(numberSamples, normalizedWeights, sampling);
    LOG_TRACE(<< "normalizedWeights = " << core::CContainerPrinter::print(normalizedWeights)
              << ", sampling = " << core::CContainerPrinter::print(sampling));

    if (sampling.size() != m_Modes.size()) {
        LOG_ERROR(<< "Failed to sample marginal likelihood");
        return;
    }

    samples.reserve(numberSamples);
    TDouble1Vec modeSamples;
    for (std::size_t i = 0; i < m_Modes.size(); ++i) {
        m_Modes[i].s_Prior->sampleMarginalLikelihood(sampling[i], modeSamples);
        LOG_TRACE(<< "modeSamples = " << core::CContainerPrinter::print(modeSamples));
        std::copy(modeSamples.begin(), modeSamples.end(), std::back_inserter(samples));
    }
    LOG_TRACE(<< "samples = " << core::CContainerPrinter::print(samples));
}

bool CMultimodalPrior::minusLogJointCdf(const TDouble1Vec& samples,
                                        const TDoubleWeightsAry1Vec& weights,
                                        double& lowerBound,
                                        double& upperBound) const {
    return this->minusLogJointCdfImpl(CMinusLogJointCdf{}, samples, weights,
                                      lowerBound, upperBound);
}

bool CMultimodalPrior::minusLogJointCdfComplement(const TDouble1Vec& samples,
                                                  const TDoubleWeightsAry1Vec& weights,
                                                  double& lowerBound,
                                                  double& upperBound) const {

    return this->minusLogJointCdfImpl(CMinusLogJointCdfComplement{}, samples,
                                      weights, lowerBound, upperBound);
}

bool CMultimodalPrior::probabilityOfLessLikelySamples(maths_t::EProbabilityCalculation calculation,
                                                      const TDouble1Vec& samples,
                                                      const TDoubleWeightsAry1Vec& weights,
                                                      double& lowerBound,
                                                      double& upperBound,
                                                      maths_t::ETail& tail) const {
    lowerBound = upperBound = 1.0;
    tail = maths_t::E_UndeterminedTail;

    if (samples.empty()) {
        LOG_ERROR(<< "Can't compute distribution for empty sample set");
        return false;
    }
    if (this->isNonInformative()) {
        return true;
    }

    if (m_Modes.size() == 1) {
        return m_Modes[0].s_Prior->probabilityOfLessLikelySamples(
            calculation, samples, weights, lowerBound, upperBound, tail);
    }

    // Ideally we'd find the probability of the set of samples whose
    // total likelihood is less than or equal to that of the specified
    // samples, i.e. the probability of the set
    //   R = { y | L(y) < L(x) }
    //
    // where,
    //   x = {x(1), x(2), ..., x(n)} is the sample vector.
    //   y is understood to be a vector quantity.
    //
    // This is not *trivially* related to the probability that the
    // probabilities of the sets
    //   R(i) = { y | L(y) < L(x(i)) }
    //
    // since the joint conditional likelihood must be integrated over
    // priors for the parameters. However, we'll approximate this as
    // the joint probability (of a collection of standard normal R.Vs.)
    // having probabilities {P(R(i))}. This becomes increasingly accurate
    // as the prior distribution narrows.
    //
    // For the two sided calculation, we use the fact that the likelihood
    // function decreases monotonically away from the interval [a, b]
    // whose end points are the leftmost and rightmost modes' modes
    // since all component likelihoods decrease away from this interval.
    //
    // To evaluate the probability in the interval [a, b] we relax
    // the hard constraint that regions where f > f(x) contribute
    // zero probability. In particular, we note that we can write
    // the probability as:
    //   P = Integral{ I(f(s) < f(x)) * f(s) }ds
    //
    // and that:
    //   I(f(s) < f(x)) = lim_{k->inf}{ exp(-k * (f(s)/f(x) - 1))
    //                                  / (1 + exp(-k * (f(s)/f(x) - 1))) }
    //
    // We evaluate a smoother integral, i.e. smaller p, initially
    // to find out which regions contribute the most to P and then
    // re-evaluate those regions we need with higher resolution
    // using the fact that the maximum error in the approximation
    // of I(f(s) < f(x)) is 0.5.

    switch (calculation) {
    case maths_t::E_OneSidedBelow:
        if (this->minusLogJointCdf(samples, weights, upperBound, lowerBound) == false) {
            LOG_ERROR(<< "Failed computing probability of less likely samples: "
                      << core::CContainerPrinter::print(samples));
            return false;
        }
        lowerBound = std::exp(-lowerBound);
        upperBound = std::exp(-upperBound);
        tail = maths_t::E_LeftTail;
        break;

    case maths_t::E_TwoSided: {
        static const double EPS{1000.0 * std::numeric_limits<double>::epsilon()};
        static const std::size_t MAX_ITERATIONS{20};

        CJointProbabilityOfLessLikelySamples lowerBoundCalculator;
        CJointProbabilityOfLessLikelySamples upperBoundCalculator;

        TDoubleDoublePr support{this->marginalLikelihoodSupport()};
        support.first = (1.0 + (support.first > 0.0 ? EPS : -EPS)) * support.first;
        support.second = (1.0 + (support.first > 0.0 ? EPS : -EPS)) * support.second;
        bool hasSeasonalScale{maths_t::hasSeasonalVarianceScale(weights)};
        double mean{hasSeasonalScale ? this->marginalLikelihoodMean() : 0.0};

        double a{std::numeric_limits<double>::max()};
        double b{-std::numeric_limits<double>::max()};
        double Z{0.0};
        for (const auto& mode : m_Modes) {
            double m{mode.s_Prior->marginalLikelihoodMode()};
            a = std::min(a, m);
            b = std::max(b, m);
            Z += mode.weight();
        }
        a = CTools::truncate(a, support.first, support.second);
        b = CTools::truncate(b, support.first, support.second);
        LOG_TRACE(<< "a = " << a << ", b = " << b << ", Z = " << Z);

        // Declared outside the loop to minimize the number of times
        // it is created.
        TDoubleWeightsAry1Vec weight(1);

        int tail_{0};
        for (std::size_t i = 0; i < samples.size(); ++i) {
            double x{samples[i]};
            weight[0] = weights[i];
            if (hasSeasonalScale) {
                x = mean + (x - mean) /
                               std::sqrt(maths_t::seasonalVarianceScale(weight[0]));
                maths_t::setSeasonalVarianceScale(1.0, weight[0]);
            }

            double fx;
            maths_t::EFloatingPointErrorStatus status =
                this->jointLogMarginalLikelihood({x}, weight, fx);
            if (status & maths_t::E_FpFailed) {
                LOG_ERROR(<< "Unable to compute likelihood for " << x);
                return false;
            }
            if (status & maths_t::E_FpOverflowed) {
                lowerBound = upperBound = 0.0;
                return true;
            }
            LOG_TRACE(<< "x = " << x << ", f(x) = " << fx);

            CPrior::CLogMarginalLikelihood logLikelihood{*this, weight};

            CTools::CMixtureProbabilityOfLessLikelySample calculator{m_Modes.size(),
                                                                     x, fx, a, b};
            for (const auto& mode : m_Modes) {
                double w{mode.weight() / Z};
                double centre{mode.s_Prior->marginalLikelihoodMode(weight[0])};
                double spread{
                    std::sqrt(mode.s_Prior->marginalLikelihoodVariance(weight[0]))};
                calculator.addMode(w, centre, spread);
                tail_ = tail_ | (x < centre ? maths_t::E_LeftTail : maths_t::E_RightTail);
            }

            double sampleLowerBound{0.0};
            double sampleUpperBound{0.0};

            double lb, ub;

            double xl;
            CEqualWithTolerance<double> lequal{CToleranceTypes::E_AbsoluteTolerance,
                                               EPS * a};
            if (calculator.leftTail(logLikelihood, MAX_ITERATIONS, lequal, xl)) {
                this->minusLogJointCdf({xl}, weight, lb, ub);
                sampleLowerBound += std::exp(std::min(-lb, -ub));
                sampleUpperBound += std::exp(std::max(-lb, -ub));
            } else {
                this->minusLogJointCdf({xl}, weight, lb, ub);
                sampleUpperBound += std::exp(std::max(-lb, -ub));
            }

            double xr;
            CEqualWithTolerance<double> requal{CToleranceTypes::E_AbsoluteTolerance,
                                               EPS * b};
            if (calculator.rightTail(logLikelihood, MAX_ITERATIONS, requal, xr)) {
                this->minusLogJointCdfComplement({xr}, weight, lb, ub);
                sampleLowerBound += std::exp(std::min(-lb, -ub));
                sampleUpperBound += std::exp(std::max(-lb, -ub));
            } else {
                this->minusLogJointCdfComplement({xr}, weight, lb, ub);
                sampleUpperBound += std::exp(std::max(-lb, -ub));
            }

            double p{0.0};
            if (a < b) {
                p = calculator.calculate(logLikelihood, sampleLowerBound);
            }

            LOG_TRACE(<< "sampleLowerBound = " << sampleLowerBound
                      << ", sampleUpperBound = " << sampleUpperBound << " p = " << p);

            lowerBoundCalculator.add(CTools::truncate(sampleLowerBound + p, 0.0, 1.0));
            upperBoundCalculator.add(CTools::truncate(sampleUpperBound + p, 0.0, 1.0));
        }

        if (lowerBoundCalculator.calculate(lowerBound) == false ||
            upperBoundCalculator.calculate(upperBound) == false) {
            LOG_ERROR(<< "Couldn't compute probability of less likely samples:"
                      << " " << lowerBoundCalculator << " " << upperBoundCalculator);
            return false;
        }
        tail = static_cast<maths_t::ETail>(tail_);
    } break;

    case maths_t::E_OneSidedAbove:
        if (this->minusLogJointCdfComplement(samples, weights, upperBound, lowerBound) == false) {
            LOG_ERROR(<< "Failed computing probability of less likely samples: "
                      << core::CContainerPrinter::print(samples));
            return false;
        }
        lowerBound = std::exp(-lowerBound);
        upperBound = std::exp(-upperBound);
        tail = maths_t::E_RightTail;
        break;
    }

    return true;
}

bool CMultimodalPrior::isNonInformative() const {
    return m_Modes.empty() ||
           (m_Modes.size() == 1 && m_Modes[0].s_Prior->isNonInformative());
}

void CMultimodalPrior::print(const std::string& indent, std::string& result) const {
    result += "\n" + indent + "multimodal";
    if (this->isNonInformative()) {
        result += " non-informative";
        return;
    }

    double Z{0.0};
    for (const auto& mode : m_Modes) {
        Z += mode.weight();
    }
    result += ":";
    for (const auto& mode : m_Modes) {
        double weight{mode.weight() / Z};
        std::string indent_{indent + " weight " +
                            core::CStringUtils::typeToStringPretty(weight) + "  "};
        mode.s_Prior->print(indent_, result);
    }
}

std::string CMultimodalPrior::printJointDensityFunction() const {
    return "Not supported";
}

std::uint64_t CMultimodalPrior::checksum(std::uint64_t seed) const {
    seed = this->CPrior::checksum(seed);
    seed = CChecksum::calculate(seed, m_Clusterer);
    seed = CChecksum::calculate(seed, m_SeedPrior);
    return CChecksum::calculate(seed, m_Modes);
}

void CMultimodalPrior::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CMultimodalPrior");
    core::CMemoryDebug::dynamicSize("m_Clusterer", m_Clusterer, mem);
    core::CMemoryDebug::dynamicSize("m_SeedPrior", m_SeedPrior, mem);
    core::CMemoryDebug::dynamicSize("m_Modes", m_Modes, mem);
}

std::size_t CMultimodalPrior::memoryUsage() const {
    std::size_t mem = core::CMemory::dynamicSize(m_Clusterer);
    mem += core::CMemory::dynamicSize(m_SeedPrior);
    mem += core::CMemory::dynamicSize(m_Modes);
    return mem;
}

std::size_t CMultimodalPrior::staticSize() const {
    return sizeof(*this);
}

void CMultimodalPrior::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertLevel(CLUSTERER_TAG, std::bind<void>(CClustererStateSerialiser(),
                                                        std::cref(*m_Clusterer),
                                                        std::placeholders::_1));
    inserter.insertLevel(SEED_PRIOR_TAG, std::bind<void>(CPriorStateSerialiser(),
                                                         std::cref(*m_SeedPrior),
                                                         std::placeholders::_1));
    for (std::size_t i = 0; i < m_Modes.size(); ++i) {
        inserter.insertLevel(MODE_TAG, std::bind(&TMode::acceptPersistInserter,
                                                 &m_Modes[i], std::placeholders::_1));
    }
    inserter.insertValue(DECAY_RATE_TAG, this->decayRate(), core::CIEEE754::E_SinglePrecision);
    inserter.insertValue(NUMBER_SAMPLES_TAG, this->numberSamples(),
                         core::CIEEE754::E_SinglePrecision);
}

std::size_t CMultimodalPrior::numberModes() const {
    return m_Modes.size();
}

bool CMultimodalPrior::checkInvariants(const std::string& tag) const {

    bool result{true};

    if (m_Modes.size() != m_Clusterer->numberClusters()) {
        LOG_ERROR(<< tag << "# modes = " << m_Modes.size()
                  << ", # clusters = " << m_Clusterer->numberClusters());
        result = false;
    }

    double numberSamples{this->numberSamples()};
    double modeSamples{0.0};
    for (const auto& mode : m_Modes) {
        if (m_Clusterer->hasCluster(mode.s_Index) == false) {
            LOG_ERROR(<< tag << "Expected cluster for mode = " << mode.s_Index);
            result = false;
        }
        modeSamples += mode.s_Prior->numberSamples();
    }

    CEqualWithTolerance<double> equal{
        CToleranceTypes::E_AbsoluteTolerance | CToleranceTypes::E_RelativeTolerance, 1e-3};
    if (equal(modeSamples, numberSamples) == false) {
        LOG_ERROR(<< tag << "Sum mode samples = " << modeSamples
                  << ", total samples = " << numberSamples);
        result = false;
    }

    return result;
}

bool CMultimodalPrior::participatesInModelSelection() const {
    return m_Modes.size() > 1;
}

double CMultimodalPrior::unmarginalizedParameters() const {
    return std::max(static_cast<double>(m_Modes.size()), 1.0) - 1.0;
}

template<typename CDF>
bool CMultimodalPrior::minusLogJointCdfImpl(CDF minusLogCdf,
                                            const TDouble1Vec& samples,
                                            const TDoubleWeightsAry1Vec& weights,
                                            double& lowerBound,
                                            double& upperBound) const {
    lowerBound = upperBound = 0.0;

    if (samples.empty()) {
        LOG_ERROR(<< "Can't compute c.d.f. for empty sample set");
        return false;
    }

    if (m_Modes.size() == 1) {
        return minusLogCdf(m_Modes[0].s_Prior, samples, weights, lowerBound, upperBound);
    }

    using TMinAccumulator = CBasicStatistics::COrderStatisticsStack<double, 1>;

    // The c.d.f. of the marginal likelihood is the weighted sum
    // of the c.d.fs of each mode since:
    //   cdf(x) = Integral{ L(u) }du
    //          = Integral{ Sum_m{ L(u | m) p(m) } }du
    //          = Sum_m{ Integral{ L(u | m) ) p(m) }du }

    // Declared outside the loop to minimize the number of times
    // they are created.
    TDouble1Vec sample(1);
    TDoubleWeightsAry1Vec weight{TWeights::UNIT};
    TDoubleVec modeLowerBounds;
    TDoubleVec modeUpperBounds;
    modeLowerBounds.reserve(m_Modes.size());
    modeUpperBounds.reserve(m_Modes.size());

    try {
        double mean{maths_t::hasSeasonalVarianceScale(weights) ? this->marginalLikelihoodMean()
                                                               : 0.0};

        for (std::size_t i = 0; i < samples.size(); ++i) {
            double n{maths_t::count(weights[i])};
            double seasonalScale{std::sqrt(maths_t::seasonalVarianceScale(weights[i]))};

            if (this->isNonInformative()) {
                lowerBound -= n * std::log(CTools::IMPROPER_CDF);
                upperBound -= n * std::log(CTools::IMPROPER_CDF);
                continue;
            }

            sample[0] = mean + (samples[i] - mean) / seasonalScale;
            maths_t::setCountVarianceScale(maths_t::countVarianceScale(weights[i]),
                                           weight[0]);

            // We re-normalize so that the maximum log c.d.f. is one
            // to avoid underflow.
            TMinAccumulator minLowerBound;
            TMinAccumulator minUpperBound;
            modeLowerBounds.clear();
            modeUpperBounds.clear();

            for (const auto& mode : m_Modes) {
                double modeLowerBound;
                double modeUpperBound;
                if (minusLogCdf(mode.s_Prior, sample, weight, modeLowerBound,
                                modeUpperBound) == false) {
                    LOG_ERROR(<< "Unable to compute c.d.f. for "
                              << core::CContainerPrinter::print(samples));
                    return false;
                }
                minLowerBound.add(modeLowerBound);
                minUpperBound.add(modeUpperBound);
                modeLowerBounds.push_back(modeLowerBound);
                modeUpperBounds.push_back(modeUpperBound);
            }

            TMeanAccumulator sampleLowerBound;
            TMeanAccumulator sampleUpperBound;

            for (std::size_t j = 0; j < m_Modes.size(); ++j) {
                LOG_TRACE(<< "Mode -log(c.d.f.) = [" << modeLowerBounds[j]
                          << "," << modeUpperBounds[j] << "]");
                double w{m_Modes[j].weight()};
                // Divide through by the largest value to avoid underflow.
                // Remember we are working with minus logs so the largest
                // value corresponds to the smallest log.
                sampleLowerBound.add(std::exp(-(modeLowerBounds[j] - minLowerBound[0])), w);
                sampleUpperBound.add(std::exp(-(modeUpperBounds[j] - minUpperBound[0])), w);
            }

            lowerBound += n * std::max(minLowerBound[0] -
                                           std::log(CBasicStatistics::mean(sampleLowerBound)),
                                       0.0);
            upperBound += n * std::max(minUpperBound[0] -
                                           std::log(CBasicStatistics::mean(sampleUpperBound)),
                                       0.0);

            LOG_TRACE(<< "sample = " << core::CContainerPrinter::print(sample) << ", sample -log(c.d.f.) = ["
                      << sampleLowerBound << "," << sampleUpperBound << "]");
        }
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to calculate c.d.f.: " << e.what());
        return false;
    }

    LOG_TRACE(<< "Joint -log(c.d.f.) = [" << lowerBound << "," << upperBound << "]");

    return true;
}

std::string CMultimodalPrior::debugWeights() const {
    return TMode::debugWeights(m_Modes);
}

////////// CMultimodalPrior::CModeSplitCallback Implementation //////////

CMultimodalPrior::CModeSplitCallback::CModeSplitCallback(CMultimodalPrior& prior)
    : m_Prior(&prior) {
}

void CMultimodalPrior::CModeSplitCallback::operator()(std::size_t sourceIndex,
                                                      std::size_t leftSplitIndex,
                                                      std::size_t rightSplitIndex) const {

    LOG_TRACE(<< "Splitting mode with index " << sourceIndex);

    TModeVec& modes = m_Prior->m_Modes;

    // Remove the split mode.
    auto mode = std::find_if(modes.begin(), modes.end(), CSetTools::CIndexInSet(sourceIndex));
    double numberSamples{mode != modes.end() ? mode->weight() : 0.0};
    modes.erase(mode);

    double pLeft{m_Prior->m_Clusterer->probability(leftSplitIndex)};
    double pRight{m_Prior->m_Clusterer->probability(rightSplitIndex)};
    double Z = (pLeft + pRight);
    if (Z > 0.0) {
        pLeft /= Z;
        pRight /= Z;
    }
    LOG_TRACE(<< "# samples = " << numberSamples << ", pLeft = " << pLeft
              << ", pRight = " << pRight);

    // Create the child modes.

    LOG_TRACE(<< "Creating mode with index " << leftSplitIndex);
    modes.emplace_back(leftSplitIndex, TPriorPtr(m_Prior->m_SeedPrior->clone()));
    {
        TDoubleVec samples;
        if (m_Prior->m_Clusterer->sample(leftSplitIndex, MODE_SPLIT_NUMBER_SAMPLES,
                                         samples) == false) {
            LOG_ERROR(<< "Couldn't find cluster for " << leftSplitIndex);
        }
        LOG_TRACE(<< "samples = " << core::CContainerPrinter::print(samples));

        double wl{pLeft * numberSamples};
        double ws{std::min(wl, 4.0)};
        double n{static_cast<double>(samples.size())};
        LOG_TRACE(<< "# left = " << wl);

        TDoubleWeightsAry1Vec weights(samples.size(), maths_t::countWeight(ws / n));
        modes.back().s_Prior->addSamples(samples, weights);

        if (wl > ws) {
            weights.assign(weights.size(), maths_t::countWeight((wl - ws) / n));
            modes.back().s_Prior->addSamples(samples, weights);
            LOG_TRACE(<< modes.back().s_Prior->print());
        }
    }

    LOG_TRACE(<< "Creating mode with index " << rightSplitIndex);
    modes.emplace_back(rightSplitIndex, TPriorPtr(m_Prior->m_SeedPrior->clone()));
    {
        TDoubleVec samples;
        if (m_Prior->m_Clusterer->sample(rightSplitIndex, MODE_SPLIT_NUMBER_SAMPLES,
                                         samples) == false) {
            LOG_ERROR(<< "Couldn't find cluster for " << rightSplitIndex);
        }
        LOG_TRACE(<< "samples = " << core::CContainerPrinter::print(samples));

        double wr{pRight * numberSamples};
        double ws{std::min(wr, 4.0)};
        double n{static_cast<double>(samples.size())};
        LOG_TRACE(<< "# right = " << wr);

        TDoubleWeightsAry1Vec weights(samples.size(), maths_t::countWeight(ws / n));
        modes.back().s_Prior->addSamples(samples, weights);

        if (wr > ws) {
            weights.assign(weights.size(), maths_t::countWeight((wr - ws) / n));
            modes.back().s_Prior->addSamples(samples, weights);
            LOG_TRACE(<< modes.back().s_Prior->print());
        }
    }

    if (m_Prior->checkInvariants("SPLIT: ") == false) {
        LOG_ERROR(<< "# samples = " << numberSamples << ", # modes = " << modes.size()
                  << ", pLeft = " << pLeft << ", pRight = " << pRight);
    }

    LOG_TRACE(<< "Split mode");
}

////////// CMultimodalPrior::CModeMergeCallback Implementation //////////

CMultimodalPrior::CModeMergeCallback::CModeMergeCallback(CMultimodalPrior& prior)
    : m_Prior(&prior) {
}

void CMultimodalPrior::CModeMergeCallback::operator()(std::size_t leftMergeIndex,
                                                      std::size_t rightMergeIndex,
                                                      std::size_t targetIndex) const {

    LOG_TRACE(<< "Merging modes with indices " << leftMergeIndex << " " << rightMergeIndex);

    TModeVec& modes{m_Prior->m_Modes};

    // Create the new mode.
    TMode newMode(targetIndex, TPriorPtr(m_Prior->m_SeedPrior->clone()));

    double wl{0.0};
    double wr{0.0};
    double w{0.0};
    std::size_t nl{0};
    std::size_t nr{0};
    TDouble1Vec samples;

    auto leftMode = std::find_if(modes.begin(), modes.end(),
                                 CSetTools::CIndexInSet(leftMergeIndex));
    if (leftMode != modes.end()) {
        wl = leftMode->s_Prior->numberSamples();
        w += wl;
        TDouble1Vec leftSamples;
        leftMode->s_Prior->sampleMarginalLikelihood(MODE_MERGE_NUMBER_SAMPLES, leftSamples);
        nl = leftSamples.size();
        samples.insert(samples.end(), leftSamples.begin(), leftSamples.end());
    } else {
        LOG_ERROR(<< "Couldn't find mode for " << leftMergeIndex);
    }

    auto rightMode = std::find_if(modes.begin(), modes.end(),
                                  CSetTools::CIndexInSet(rightMergeIndex));
    if (rightMode != modes.end()) {
        wr = rightMode->s_Prior->numberSamples();
        w += wr;
        TDouble1Vec rightSamples;
        rightMode->s_Prior->sampleMarginalLikelihood(MODE_MERGE_NUMBER_SAMPLES, rightSamples);
        nr = rightSamples.size();
        samples.insert(samples.end(), rightSamples.begin(), rightSamples.end());
    } else {
        LOG_ERROR(<< "Couldn't find mode for " << rightMergeIndex);
    }

    if (w > 0.0) {
        double nl_{static_cast<double>(nl)};
        double nr_{static_cast<double>(nr)};
        double Z{(nl_ * wl + nr_ * wr) / (nl_ + nr_)};
        wl /= Z;
        wr /= Z;
    }

    LOG_TRACE(<< "samples = " << core::CContainerPrinter::print(samples));
    LOG_TRACE(<< "w = " << w << ", wl = " << wl << ", wr = " << wr);

    double ws{std::min(w, 4.0)};
    double n{static_cast<double>(samples.size())};

    TDoubleWeightsAry1Vec weights;
    weights.reserve(samples.size());
    weights.resize(nl, maths_t::countWeight(wl * ws / n));
    weights.resize(nl + nr, maths_t::countWeight(wr * ws / n));
    newMode.s_Prior->addSamples(samples, weights);

    if (w > ws) {
        weights.clear();
        weights.resize(nl, maths_t::countWeight(wl * (w - ws) / n));
        weights.resize(nl + nr, maths_t::countWeight(wr * (w - ws) / n));
        newMode.s_Prior->addSamples(samples, weights);
    }

    // Remove the merged modes.
    TSizeSet mergedIndices;
    mergedIndices.insert(leftMergeIndex);
    mergedIndices.insert(rightMergeIndex);
    modes.erase(std::remove_if(modes.begin(), modes.end(), CSetTools::CIndexInSet(mergedIndices)),
                modes.end());

    // Add the new mode.
    LOG_TRACE(<< "Creating mode with index " << targetIndex);
    modes.push_back(std::move(newMode));

    m_Prior->checkInvariants("MERGE: ");

    LOG_TRACE(<< "Merged modes");
}
}
}
