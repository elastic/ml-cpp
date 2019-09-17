/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CMultimodalPrior.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/Constants.h>
#include <core/RestoreMacros.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CChecksum.h>
#include <maths/CClusterer.h>
#include <maths/CClustererStateSerialiser.h>
#include <maths/CIntegration.h>
#include <maths/CKMeansOnline1d.h>
#include <maths/CMathsFuncs.h>
#include <maths/CMultimodalPriorUtils.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/COrderings.h>
#include <maths/CPriorStateSerialiser.h>
#include <maths/CRestoreParams.h>
#include <maths/CSampling.h>
#include <maths/CSetTools.h>
#include <maths/CSolvers.h>
#include <maths/CTools.h>
#include <maths/CToolsDetail.h>
#include <maths/ProbabilityAggregators.h>

#include <boost/numeric/conversion/bounds.hpp>
#include <boost/optional.hpp>

#include <cmath>
#include <set>

namespace ml {
namespace maths {

namespace {

using TSizeDoublePr = std::pair<std::size_t, double>;
using TSizeDoublePr2Vec = core::CSmallVector<TSizeDoublePr, 2>;
using TDoubleDoublePr = std::pair<double, double>;
using TSizeSet = std::set<std::size_t>;

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
    for (std::size_t i = 0u; i < normals.size(); ++i) {
        m_Modes.emplace_back(i, TPriorPtr(normals.back().clone()));
    }
}

CMultimodalPrior::CMultimodalPrior(maths_t::EDataType dataType, double decayRate, TPriorPtrVec& priors)
    : CPrior(dataType, decayRate) {
    m_Modes.reserve(priors.size());
    for (std::size_t i = 0u; i < priors.size(); ++i) {
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

    return true;
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
    double result = 0.0;

    if (this->needsOffset()) {
        TSizeDoublePr2Vec clusters;
        for (std::size_t i = 0u; i < samples.size(); ++i) {
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
    double offset = 0.0;
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
        bool hasSeasonalScale = !this->isNonInformative() &&
                                maths_t::hasSeasonalVarianceScale(weights);
        double mean = hasSeasonalScale ? this->marginalLikelihoodMean() : 0.0;

        for (std::size_t i = 0u; i < samples.size(); ++i) {
            double x = samples[i];
            if (!CMathsFuncs::isFinite(x)) {
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

            auto addModeWeight = [](double sum, const TMode& mode) {
                return sum + mode.weight();
            };
            double Z = std::accumulate(m_Modes.begin(), m_Modes.end(),
                                       maths_t::count(weight[0]), addModeWeight);

            double n = 0.0;
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
    if (!CMathsFuncs::isFinite(time) || time < 0.0) {
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

    this->numberSamples(this->numberSamples() * std::exp(-this->decayRate() * time));
    LOG_TRACE(<< "numberSamples = " << this->numberSamples());
}

TDoubleDoublePr CMultimodalPrior::marginalLikelihoodSupport() const {
    return CMultimodalPriorUtils::marginalLikelihoodSupport(m_Modes);
}

double CMultimodalPrior::marginalLikelihoodMean() const {
    return CMultimodalPriorUtils::marginalLikelihoodMean(m_Modes);
}

double CMultimodalPrior::nearestMarginalLikelihoodMean(double value) const {

    if (m_Modes.empty()) {
        return 0.0;
    }

    double mean = m_Modes[0].s_Prior->marginalLikelihoodMean();
    double distance = std::fabs(value - mean);
    double result = mean;
    for (std::size_t i = 1u; i < m_Modes.size(); ++i) {
        mean = m_Modes[i].s_Prior->marginalLikelihoodMean();
        if (std::fabs(value - mean) < distance) {
            distance = std::fabs(value - mean);
            result = mean;
        }
    }
    return result;
}

double CMultimodalPrior::marginalLikelihoodMode(const TDoubleWeightsAry& weights) const {
    return CMultimodalPriorUtils::marginalLikelihoodMode(m_Modes, weights);
}

CMultimodalPrior::TDouble1Vec
CMultimodalPrior::marginalLikelihoodModes(const TDoubleWeightsAry& weights) const {
    TDouble1Vec result(m_Modes.size());
    for (std::size_t i = 0u; i < m_Modes.size(); ++i) {
        result[i] = m_Modes[i].s_Prior->marginalLikelihoodMode(weights);
    }
    return result;
}

double CMultimodalPrior::marginalLikelihoodVariance(const TDoubleWeightsAry& weights) const {
    return CMultimodalPriorUtils::marginalLikelihoodVariance(m_Modes, weights);
}

TDoubleDoublePr
CMultimodalPrior::marginalLikelihoodConfidenceInterval(double percentage,
                                                       const TDoubleWeightsAry& weights) const {
    return CMultimodalPriorUtils::marginalLikelihoodConfidenceInterval(
        *this, m_Modes, percentage, weights);
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
        result = boost::numeric::bounds<double>::lowest();
        return maths_t::E_FpOverflowed;
    }

    return m_Modes.size() == 1
               ? m_Modes[0].s_Prior->jointLogMarginalLikelihood(samples, weights, result)
               : CMultimodalPriorUtils::jointLogMarginalLikelihood(
                     m_Modes, samples, weights, result);
}

void CMultimodalPrior::sampleMarginalLikelihood(std::size_t numberSamples,
                                                TDouble1Vec& samples) const {

    samples.clear();

    if (numberSamples == 0 || this->numberSamples() == 0.0) {
        return;
    }

    CMultimodalPriorUtils::sampleMarginalLikelihood(m_Modes, numberSamples, samples);
}

bool CMultimodalPrior::minusLogJointCdf(const TDouble1Vec& samples,
                                        const TDoubleWeightsAry1Vec& weights,
                                        double& lowerBound,
                                        double& upperBound) const {
    return CMultimodalPriorUtils::minusLogJointCdf(m_Modes, samples, weights,
                                                   lowerBound, upperBound);
}

bool CMultimodalPrior::minusLogJointCdfComplement(const TDouble1Vec& samples,
                                                  const TDoubleWeightsAry1Vec& weights,
                                                  double& lowerBound,
                                                  double& upperBound) const {
    return CMultimodalPriorUtils::minusLogJointCdfComplement(
        m_Modes, samples, weights, lowerBound, upperBound);
}

bool CMultimodalPrior::probabilityOfLessLikelySamples(maths_t::EProbabilityCalculation calculation,
                                                      const TDouble1Vec& samples,
                                                      const TDoubleWeightsAry1Vec& weights,
                                                      double& lowerBound,
                                                      double& upperBound,
                                                      maths_t::ETail& tail) const {
    return CMultimodalPriorUtils::probabilityOfLessLikelySamples(
        *this, m_Modes, calculation, samples, weights, lowerBound, upperBound, tail);
}

bool CMultimodalPrior::isNonInformative() const {
    return CMultimodalPriorUtils::isNonInformative(m_Modes);
}

void CMultimodalPrior::print(const std::string& indent, std::string& result) const {
    CMultimodalPriorUtils::print(m_Modes, indent, result);
}

std::string CMultimodalPrior::printJointDensityFunction() const {
    return "Not supported";
}

uint64_t CMultimodalPrior::checksum(uint64_t seed) const {
    seed = this->CPrior::checksum(seed);
    seed = CChecksum::calculate(seed, m_Clusterer);
    seed = CChecksum::calculate(seed, m_SeedPrior);
    return CChecksum::calculate(seed, m_Modes);
}

void CMultimodalPrior::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
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
    for (std::size_t i = 0u; i < m_Modes.size(); ++i) {
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

    bool result = true;

    if (m_Modes.size() != m_Clusterer->numberClusters()) {
        LOG_ERROR(<< tag << "# modes = " << m_Modes.size()
                  << ", # clusters = " << m_Clusterer->numberClusters());
        result = false;
    }

    double numberSamples = this->numberSamples();
    double modeSamples = 0.0;
    for (const auto& mode : m_Modes) {
        if (!m_Clusterer->hasCluster(mode.s_Index)) {
            LOG_ERROR(<< tag << "Expected cluster for mode = " << mode.s_Index);
            result = false;
        }
        modeSamples += mode.s_Prior->numberSamples();
    }

    CEqualWithTolerance<double> equal(
        CToleranceTypes::E_AbsoluteTolerance | CToleranceTypes::E_RelativeTolerance, 1e-3);
    if (!equal(modeSamples, numberSamples)) {
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
    double numberSamples = mode != modes.end() ? mode->weight() : 0.0;
    modes.erase(mode);

    double pLeft = m_Prior->m_Clusterer->probability(leftSplitIndex);
    double pRight = m_Prior->m_Clusterer->probability(rightSplitIndex);
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
        if (!m_Prior->m_Clusterer->sample(leftSplitIndex, MODE_SPLIT_NUMBER_SAMPLES, samples)) {
            LOG_ERROR(<< "Couldn't find cluster for " << leftSplitIndex);
        }
        LOG_TRACE(<< "samples = " << core::CContainerPrinter::print(samples));

        double wl = pLeft * numberSamples;
        double ws = std::min(wl, 4.0);
        double n = static_cast<double>(samples.size());
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
        if (!m_Prior->m_Clusterer->sample(rightSplitIndex, MODE_SPLIT_NUMBER_SAMPLES, samples)) {
            LOG_ERROR(<< "Couldn't find cluster for " << rightSplitIndex);
        }
        LOG_TRACE(<< "samples = " << core::CContainerPrinter::print(samples));

        double wr = pRight * numberSamples;
        double ws = std::min(wr, 4.0);
        double n = static_cast<double>(samples.size());
        LOG_TRACE(<< "# right = " << wr);

        TDoubleWeightsAry1Vec weights(samples.size(), maths_t::countWeight(ws / n));
        modes.back().s_Prior->addSamples(samples, weights);

        if (wr > ws) {
            weights.assign(weights.size(), maths_t::countWeight((wr - ws) / n));
            modes.back().s_Prior->addSamples(samples, weights);
            LOG_TRACE(<< modes.back().s_Prior->print());
        }
    }

    if (!m_Prior->checkInvariants("SPLIT: ")) {
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

    TModeVec& modes = m_Prior->m_Modes;

    // Create the new mode.
    TMode newMode(targetIndex, TPriorPtr(m_Prior->m_SeedPrior->clone()));

    double wl = 0.0;
    double wr = 0.0;
    double w = 0.0;
    std::size_t nl = 0;
    std::size_t nr = 0;
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
        double nl_ = static_cast<double>(nl);
        double nr_ = static_cast<double>(nr);
        double Z = (nl_ * wl + nr_ * wr) / (nl_ + nr_);
        wl /= Z;
        wr /= Z;
    }

    LOG_TRACE(<< "samples = " << core::CContainerPrinter::print(samples));
    LOG_TRACE(<< "w = " << w << ", wl = " << wl << ", wr = " << wr);

    double ws = std::min(w, 4.0);
    double n = static_cast<double>(samples.size());

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
