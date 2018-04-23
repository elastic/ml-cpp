/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CNaiveBayes.h>

#include <core/CContainerPrinter.h>
#include <core/CFunctional.h>
#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/CChecksum.h>
#include <maths/CPrior.h>
#include <maths/CPriorStateSerialiser.h>
#include <maths/CRestoreParams.h>
#include <maths/CTools.h>
#include <maths/MathsTypes.h>

#include <boost/bind.hpp>
#include <boost/ref.hpp>

#include <numeric>
#include <string>

namespace ml {
namespace maths {
namespace {
const std::string PRIOR_TAG{"a"};
const std::string CLASS_LABEL_TAG{"b"};
const std::string CLASS_MODEL_TAG{"c"};
const std::string MIN_MAX_LOG_LIKELIHOOD_TO_USE_FEATURE_TAG{"d"};
const std::string COUNT_TAG{"e"};
const std::string CONDITIONAL_DENSITY_FROM_PRIOR_TAG{"f"};
}

CNaiveBayesFeatureDensityFromPrior::CNaiveBayesFeatureDensityFromPrior(const CPrior& prior)
    : m_Prior(prior.clone()) {
}

void CNaiveBayesFeatureDensityFromPrior::add(const TDouble1Vec& x) {
    m_Prior->addSamples(x, maths_t::CUnitWeights::SINGLE_UNIT);
}

CNaiveBayesFeatureDensityFromPrior* CNaiveBayesFeatureDensityFromPrior::clone() const {
    return new CNaiveBayesFeatureDensityFromPrior(*m_Prior);
}

bool CNaiveBayesFeatureDensityFromPrior::acceptRestoreTraverser(
    const SDistributionRestoreParams& params,
    core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name{traverser.name()};
        RESTORE(PRIOR_TAG, traverser.traverseSubLevel(boost::bind<bool>(
                               CPriorStateSerialiser(), boost::cref(params),
                               boost::ref(m_Prior), _1)));
    } while (traverser.next());
    return true;
}

void CNaiveBayesFeatureDensityFromPrior::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertLevel(PRIOR_TAG, boost::bind<void>(CPriorStateSerialiser(),
                                                      boost::cref(*m_Prior), _1));
}

double CNaiveBayesFeatureDensityFromPrior::logValue(const TDouble1Vec& x) const {
    double result;
    if (m_Prior->jointLogMarginalLikelihood(x, maths_t::CUnitWeights::SINGLE_UNIT,
                                            result) != maths_t::E_FpNoErrors) {
        LOG_ERROR("Bad density value at " << x << " for " << m_Prior->print());
        return boost::numeric::bounds<double>::lowest();
    }
    return result;
}

double CNaiveBayesFeatureDensityFromPrior::logMaximumValue() const {
    double result;
    if (m_Prior->jointLogMarginalLikelihood({m_Prior->marginalLikelihoodMode()},
                                            maths_t::CUnitWeights::SINGLE_UNIT,
                                            result) != maths_t::E_FpNoErrors) {
        LOG_ERROR("Bad density value for " << m_Prior->print());
        return boost::numeric::bounds<double>::lowest();
    }
    return result;
}

void CNaiveBayesFeatureDensityFromPrior::dataType(maths_t::EDataType dataType) {
    m_Prior->dataType(dataType);
}

void CNaiveBayesFeatureDensityFromPrior::propagateForwardsByTime(double time) {
    m_Prior->propagateForwardsByTime(time);
}

void CNaiveBayesFeatureDensityFromPrior::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    return core::CMemoryDebug::dynamicSize("m_Prior", m_Prior, mem);
}

std::size_t CNaiveBayesFeatureDensityFromPrior::staticSize() const {
    return sizeof(*this);
}

std::size_t CNaiveBayesFeatureDensityFromPrior::memoryUsage() const {
    return core::CMemory::dynamicSize(m_Prior);
}

uint64_t CNaiveBayesFeatureDensityFromPrior::checksum(uint64_t seed) const {
    return CChecksum::calculate(seed, m_Prior);
}

std::string CNaiveBayesFeatureDensityFromPrior::print() const {
    std::string result;
    m_Prior->print("  ", result);
    return result;
}

CNaiveBayes::CNaiveBayes(const CNaiveBayesFeatureDensity& exemplar,
                         double decayRate,
                         TOptionalDouble minMaxLogLikelihoodToUseFeature)
    : m_MinMaxLogLikelihoodToUseFeature{minMaxLogLikelihoodToUseFeature},
      m_DecayRate{decayRate}, m_Exemplar{exemplar.clone()}, m_ClassConditionalDensities{2} {
}

CNaiveBayes::CNaiveBayes(const SDistributionRestoreParams& params,
                         core::CStateRestoreTraverser& traverser)
    : m_DecayRate{params.s_DecayRate}, m_ClassConditionalDensities{2} {
    traverser.traverseSubLevel(boost::bind(&CNaiveBayes::acceptRestoreTraverser,
                                           this, boost::cref(params), _1));
}

bool CNaiveBayes::acceptRestoreTraverser(const SDistributionRestoreParams& params,
                                         core::CStateRestoreTraverser& traverser) {
    std::size_t label;
    do {
        const std::string& name{traverser.name()};
        RESTORE_BUILT_IN(CLASS_LABEL_TAG, label)
        RESTORE_SETUP_TEARDOWN(CLASS_MODEL_TAG, SClass class_,
                               traverser.traverseSubLevel(boost::bind(
                                   &SClass::acceptRestoreTraverser,
                                   boost::ref(class_), boost::cref(params), _1)),
                               m_ClassConditionalDensities.emplace(label, class_))
        RESTORE_SETUP_TEARDOWN(MIN_MAX_LOG_LIKELIHOOD_TO_USE_FEATURE_TAG, double value,
                               core::CStringUtils::stringToType(traverser.value(), value),
                               m_MinMaxLogLikelihoodToUseFeature.reset(value))
    } while (traverser.next());
    return true;
}

void CNaiveBayes::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    using TSizeClassUMapCItr = TSizeClassUMap::const_iterator;
    using TSizeClassUMapCItrVec = std::vector<TSizeClassUMapCItr>;
    TSizeClassUMapCItrVec classes;
    classes.reserve(m_ClassConditionalDensities.size());
    for (auto i = m_ClassConditionalDensities.begin();
         i != m_ClassConditionalDensities.end(); ++i) {
        classes.push_back(i);
    }
    std::sort(classes.begin(), classes.end(),
              core::CFunctional::SDereference<COrderings::SFirstLess>());
    for (const auto& class_ : classes) {
        inserter.insertValue(CLASS_LABEL_TAG, class_->first);
        inserter.insertLevel(CLASS_MODEL_TAG,
                             boost::bind(&SClass::acceptPersistInserter,
                                         boost::ref(class_->second), _1));
    }
    if (m_MinMaxLogLikelihoodToUseFeature) {
        inserter.insertValue(MIN_MAX_LOG_LIKELIHOOD_TO_USE_FEATURE_TAG,
                             *m_MinMaxLogLikelihoodToUseFeature,
                             core::CIEEE754::E_SinglePrecision);
    }
}

void CNaiveBayes::swap(CNaiveBayes& other) {
    std::swap(m_DecayRate, other.m_DecayRate);
    m_Exemplar.swap(other.m_Exemplar);
    m_ClassConditionalDensities.swap(other.m_ClassConditionalDensities);
    std::swap(m_MinMaxLogLikelihoodToUseFeature, other.m_MinMaxLogLikelihoodToUseFeature);
}

bool CNaiveBayes::initialized() const {
    return m_ClassConditionalDensities.size() > 0;
}

void CNaiveBayes::initialClassCounts(const TDoubleSizePrVec& counts) {
    for (const auto& count : counts) {
        m_ClassConditionalDensities[count.second] = SClass{count.first, {}};
    }
}

void CNaiveBayes::addTrainingDataPoint(std::size_t label, const TDouble1VecVec& x) {
    if (!this->validate(x)) {
        return;
    }

    auto& class_ = m_ClassConditionalDensities[label];

    if (class_.s_ConditionalDensities.empty()) {
        class_.s_ConditionalDensities.reserve(x.size());
        std::generate_n(
            std::back_inserter(class_.s_ConditionalDensities), x.size(),
            [this]() { return TFeatureDensityPtr{m_Exemplar->clone()}; });
    }

    bool updateCount{false};
    for (std::size_t i = 0u; i < x.size(); ++i) {
        if (x[i].size() > 0) {
            class_.s_ConditionalDensities[i]->add(x[i]);
            updateCount = true;
        }
    }

    if (updateCount) {
        class_.s_Count += 1.0;
    } else {
        LOG_TRACE("Ignoring empty feature vector");
    }
}

void CNaiveBayes::dataType(maths_t::EDataType dataType) {
    for (auto& class_ : m_ClassConditionalDensities) {
        for (auto& density : class_.second.s_ConditionalDensities) {
            density->dataType(dataType);
        }
    }
}

void CNaiveBayes::propagateForwardsByTime(double time) {
    double factor{std::exp(-m_DecayRate * time)};
    for (auto& class_ : m_ClassConditionalDensities) {
        class_.second.s_Count *= factor;
        for (auto& density : class_.second.s_ConditionalDensities) {
            density->propagateForwardsByTime(time);
        }
    }
}

CNaiveBayes::TDoubleSizePrVec
CNaiveBayes::highestClassProbabilities(std::size_t n, const TDouble1VecVec& x) const {
    TDoubleSizePrVec p(this->classProbabilities(x));
    n = std::min(n, p.size());
    std::sort(p.begin(), p.begin() + n, std::greater<TDoubleSizePr>());
    return TDoubleSizePrVec{p.begin(), p.begin() + n};
}

double CNaiveBayes::classProbability(std::size_t label, const TDouble1VecVec& x) const {
    TDoubleSizePrVec p(this->classProbabilities(x));
    auto i = std::find_if(p.begin(), p.end(), [label](const TDoubleSizePr& p_) {
        return p_.second == label;
    });
    return i == p.end() ? 0.0 : i->first;
}

CNaiveBayes::TDoubleSizePrVec CNaiveBayes::classProbabilities(const TDouble1VecVec& x) const {
    if (!this->validate(x)) {
        return {};
    }
    if (m_ClassConditionalDensities.empty()) {
        LOG_ERROR("Trying to compute class probabilities without supplying training data");
        return {};
    }

    using TDoubleVec = std::vector<double>;
    using TMaxAccumulator = CBasicStatistics::SMax<double>::TAccumulator;

    TDoubleSizePrVec p;
    p.reserve(m_ClassConditionalDensities.size());
    for (const auto& class_ : m_ClassConditionalDensities) {
        p.emplace_back(CTools::fastLog(class_.second.s_Count), class_.first);
    }

    TDoubleVec logLikelihoods;
    for (std::size_t i = 0u; i < x.size(); ++i) {
        if (x[i].size() > 0) {
            TMaxAccumulator maxLogLikelihood;
            logLikelihoods.clear();
            for (const auto& class_ : m_ClassConditionalDensities) {
                const auto& density = class_.second.s_ConditionalDensities[i];
                double logLikelihood{density->logValue(x[i])};
                double logMaximumLikelihood{density->logMaximumValue()};
                maxLogLikelihood.add(logLikelihood - logMaximumLikelihood);
                logLikelihoods.push_back(logLikelihood);
            }
            double weight{1.0};
            if (m_MinMaxLogLikelihoodToUseFeature) {
                weight = CTools::logisticFunction(
                    (maxLogLikelihood[0] - *m_MinMaxLogLikelihoodToUseFeature) /
                        std::fabs(*m_MinMaxLogLikelihoodToUseFeature),
                    0.1);
            }
            for (std::size_t j = 0u; j < logLikelihoods.size(); ++j) {
                p[j].first += weight * logLikelihoods[j];
            }
        }
    }

    double scale{std::max_element(p.begin(), p.end())->first};
    double Z{0.0};
    for (auto& pc : p) {
        pc.first = std::exp(pc.first - scale);
        Z += pc.first;
    }
    for (auto& pc : p) {
        pc.first /= Z;
    }

    return p;
}

void CNaiveBayes::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    core::CMemoryDebug::dynamicSize("m_Exemplar", m_Exemplar, mem);
    core::CMemoryDebug::dynamicSize("m_ClassConditionalDensities",
                                    m_ClassConditionalDensities, mem);
}

std::size_t CNaiveBayes::memoryUsage() const {
    return core::CMemory::dynamicSize(m_Exemplar) +
           core::CMemory::dynamicSize(m_ClassConditionalDensities);
}

uint64_t CNaiveBayes::checksum(uint64_t seed) const {
    return CChecksum::calculate(seed, m_ClassConditionalDensities);
}

std::string CNaiveBayes::print() const {
    std::ostringstream result;
    result << "\n";
    for (const auto& class_ : m_ClassConditionalDensities) {
        result << "CLASS(" << class_.first << ")\n";
        for (const auto& density : class_.second.s_ConditionalDensities) {
            result << "---";
            result << density->print() << "\n";
        }
    }
    return result.str();
}

bool CNaiveBayes::validate(const TDouble1VecVec& x) const {
    auto class_ = m_ClassConditionalDensities.begin();
    if (class_ != m_ClassConditionalDensities.end() &&
        class_->second.s_ConditionalDensities.size() > 0 &&
        class_->second.s_ConditionalDensities.size() != x.size()) {
        LOG_ERROR("Unexpected feature vector: " << core::CContainerPrinter::print(x));
        return false;
    }
    return true;
}

bool CNaiveBayes::SClass::acceptRestoreTraverser(const SDistributionRestoreParams& params,
                                                 core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name{traverser.name()};
        RESTORE_BUILT_IN(COUNT_TAG, s_Count)
        RESTORE_SETUP_TEARDOWN(CONDITIONAL_DENSITY_FROM_PRIOR_TAG,
                               CNaiveBayesFeatureDensityFromPrior tmp,
                               traverser.traverseSubLevel(boost::bind(
                                   &CNaiveBayesFeatureDensityFromPrior::acceptRestoreTraverser,
                                   boost::ref(tmp), boost::cref(params), _1)),
                               s_ConditionalDensities.emplace_back(tmp.clone()))
        // Add other implementations' restore code here.
    } while (traverser.next());
    return true;
}

void CNaiveBayes::SClass::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(COUNT_TAG, s_Count, core::CIEEE754::E_SinglePrecision);
    for (const auto& density : s_ConditionalDensities) {
        if (dynamic_cast<const CNaiveBayesFeatureDensityFromPrior*>(density.get())) {
            inserter.insertLevel(CONDITIONAL_DENSITY_FROM_PRIOR_TAG,
                                 boost::bind(&CNaiveBayesFeatureDensity::acceptPersistInserter,
                                             density.get(), _1));
            continue;
        }
        // Add other implementations' persist code here.
    }
}

void CNaiveBayes::SClass::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    core::CMemoryDebug::dynamicSize("s_ConditionalDensities", s_ConditionalDensities, mem);
}

std::size_t CNaiveBayes::SClass::memoryUsage() const {
    return core::CMemory::dynamicSize(s_ConditionalDensities);
}

uint64_t CNaiveBayes::SClass::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, s_Count);
    return CChecksum::calculate(seed, s_ConditionalDensities);
}
}
}
