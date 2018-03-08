/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2018 Elasticsearch BV. All Rights Reserved.
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

#include <maths/CNaiveBayes.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CFunctional.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/CChecksum.h>
#include <maths/CPrior.h>
#include <maths/CPriorStateSerialiser.h>
#include <maths/CRestoreParams.h>
#include <maths/CTools.h>

#include <boost/bind.hpp>
#include <boost/ref.hpp>

#include <numeric>
#include <string>

namespace ml
{
namespace maths
{
namespace
{
const std::string PRIOR_TAG{"a"};
const std::string CLASS_LABEL_TAG{"b"};
const std::string CLASS_MODEL_TAG{"c"};
const std::string COUNT_TAG{"d"};
const std::string CONDITIONAL_DENSITY_FROM_PRIOR_TAG{"e"};
}

CNaiveBayesFeatureDensityFromPrior::CNaiveBayesFeatureDensityFromPrior(CPrior &prior) :
        m_Prior(prior.clone())
{}

void CNaiveBayesFeatureDensityFromPrior::add(const TDouble1Vec &x)
{
    m_Prior->addSamples(CConstantWeights::COUNT, x, CConstantWeights::SINGLE_UNIT);
}

CNaiveBayesFeatureDensityFromPrior *CNaiveBayesFeatureDensityFromPrior::clone() const
{
    return new CNaiveBayesFeatureDensityFromPrior(*m_Prior);
}

bool CNaiveBayesFeatureDensityFromPrior::acceptRestoreTraverser(const SDistributionRestoreParams &params,
                                                                core::CStateRestoreTraverser &traverser)
{
    do
    {
        const std::string &name{traverser.name()};
        RESTORE(PRIOR_TAG, traverser.traverseSubLevel(boost::bind<bool>(
                               CPriorStateSerialiser(),
                               boost::cref(params), boost::ref(m_Prior), _1)));
    }
    while (traverser.next());
    return true;
}

void CNaiveBayesFeatureDensityFromPrior::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertLevel(PRIOR_TAG, boost::bind<void>(CPriorStateSerialiser(),
                                                      boost::cref(*m_Prior), _1));
}

double CNaiveBayesFeatureDensityFromPrior::logValue(const TDouble1Vec &x) const
{
    double result;
    if (m_Prior->jointLogMarginalLikelihood(CConstantWeights::COUNT, x,
                                            CConstantWeights::SINGLE_UNIT,
                                            result) != maths_t::E_FpNoErrors)
    {
        LOG_ERROR("Bad value density value for " << x);
        return boost::numeric::bounds<double>::lowest();
    }
    return result;
}

void CNaiveBayesFeatureDensityFromPrior::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    return core::CMemoryDebug::dynamicSize("m_Prior", m_Prior, mem);
}

std::size_t CNaiveBayesFeatureDensityFromPrior::staticSize() const
{
    return sizeof(*this);
}

std::size_t CNaiveBayesFeatureDensityFromPrior::memoryUsage() const
{
    return core::CMemory::dynamicSize(m_Prior);
}

void CNaiveBayesFeatureDensityFromPrior::propagateForwardsByTime(double time)
{
   m_Prior->propagateForwardsByTime(time);
}

uint64_t CNaiveBayesFeatureDensityFromPrior::checksum(uint64_t seed) const
{
    return CChecksum::calculate(seed, m_Prior);
}


CNaiveBayes::CNaiveBayes(const CNaiveBayesFeatureDensity &exemplar, double decayRate) :
        m_DecayRate{decayRate},
        m_Exemplar{exemplar.clone()},
        m_ClassConditionalDensities{2}
{}

CNaiveBayes::CNaiveBayes(const SDistributionRestoreParams &params,
                         core::CStateRestoreTraverser &traverser) :
        m_DecayRate{params.s_DecayRate},
        m_ClassConditionalDensities{2}
{
    traverser.traverseSubLevel(boost::bind(&CNaiveBayes::acceptRestoreTraverser,
                                           this, boost::cref(params), _1));
}

bool CNaiveBayes::acceptRestoreTraverser(const SDistributionRestoreParams &params,
                                         core::CStateRestoreTraverser &traverser)
{
    std::size_t label;
    do
    {
        const std::string &name{traverser.name()};
        RESTORE_BUILT_IN(CLASS_LABEL_TAG, label)
        RESTORE_SETUP_TEARDOWN(CLASS_MODEL_TAG,
                               SClass class_,
                               traverser.traverseSubLevel(boost::bind(
                                             &SClass::acceptRestoreTraverser,
                                             boost::ref(class_), boost::cref(params), _1)),
                               m_ClassConditionalDensities.emplace(label, class_))
    }
    while (traverser.next());
    return true;
}

void CNaiveBayes::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    using TSizeClassUMapCItr = TSizeClassUMap::const_iterator;
    using TSizeClassUMapCItrVec = std::vector<TSizeClassUMapCItr>;
    TSizeClassUMapCItrVec classes;
    classes.reserve(m_ClassConditionalDensities.size());
    for (auto i = m_ClassConditionalDensities.begin(); i != m_ClassConditionalDensities.end(); ++i)
    {
        classes.push_back(i);
    }
    std::sort(classes.begin(), classes.end(), core::CFunctional::SDereference<COrderings::SFirstLess>());
    for (const auto &class_ : classes)
    {
        inserter.insertValue(CLASS_LABEL_TAG, class_->first);
        inserter.insertLevel(CLASS_MODEL_TAG, boost::bind(&SClass::acceptPersistInserter,
                                                          boost::ref(class_->second), _1));
    }
}

void CNaiveBayes::initialClassCounts(const TDoubleSizePrVec &counts)
{
    for (const auto &count : counts)
    {
        m_ClassConditionalDensities[count.second] = SClass{count.first, {}};
    }
}

void CNaiveBayes::addTrainingDataPoint(std::size_t label, const TDouble1VecVec &x)
{
    if (!this->validate(x))
    {
        return;
    }

    auto &class_ = m_ClassConditionalDensities[label];

    if (class_.s_ConditionalDensities.empty())
    {
        class_.s_ConditionalDensities.reserve(x.size());
        std::generate_n(std::back_inserter(class_.s_ConditionalDensities),
                        x.size(),
                        [this]() { return TFeatureDensityPtr{m_Exemplar->clone()}; });
    }

    bool updateCount{false};
    for (std::size_t i = 0u; i < x.size(); ++i)
    {
        if (x[i].size() > 0)
        {
            class_.s_ConditionalDensities[i]->add(x[i]);
            updateCount = true;
        }
    }

    if (updateCount)
    {
        class_.s_Count += 1.0;
    }
    else
    {
        LOG_TRACE("Ignoring empty feature vector");
    }
}

void CNaiveBayes::propagateForwardsByTime(double time)
{
    double factor{std::exp(-m_DecayRate * time)};
    for (auto &class_ : m_ClassConditionalDensities)
    {
        class_.second.s_Count *= factor;
        for (auto &density : class_.second.s_ConditionalDensities)
        {
            density->propagateForwardsByTime(time);
        }
    }
}

CNaiveBayes::TDoubleSizePrVec
CNaiveBayes::highestClassProbabilities(std::size_t n, const TDouble1VecVec &x) const
{
    if (!this->validate(x))
    {
        return {};
    }
    if (m_ClassConditionalDensities.empty())
    {
        LOG_ERROR("Trying to compute class probabilities without supplying training data");
        return {};
    }

    TDoubleSizePrVec p;
    p.reserve(m_ClassConditionalDensities.size());

    for (const auto &class_ : m_ClassConditionalDensities)
    {
        double f{CTools::fastLog(class_.second.s_Count)};
        for (std::size_t i = 0u; i < x.size(); ++i)
        {
            if (x[i].size() > 0)
            {
                f += class_.second.s_ConditionalDensities[i]->logValue(x[i]);
            }
        }
        p.emplace_back(f, class_.first);
    }

    double scale{std::max_element(p.begin(), p.end())->first};
    double Z{0.0};
    for (auto &pc : p)
    {
        pc.first = std::exp(pc.first - scale);
        Z += pc.first;
    }
    for (auto &pc : p)
    {
        pc.first /= Z;
    }

    n = std::min(n, p.size());
    std::sort(p.begin(), p.begin() + n, std::greater<TDoubleSizePr>());

    return TDoubleSizePrVec{p.begin(), p.begin() + n};
}

void CNaiveBayes::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    core::CMemoryDebug::dynamicSize("m_Exemplar", m_Exemplar, mem);
    core::CMemoryDebug::dynamicSize("m_ClassConditionalDensities",
                                    m_ClassConditionalDensities, mem);
}

std::size_t CNaiveBayes::memoryUsage() const
{
    return  core::CMemory::dynamicSize(m_Exemplar)
          + core::CMemory::dynamicSize(m_ClassConditionalDensities);
}

uint64_t CNaiveBayes::checksum(uint64_t seed) const
{
    return CChecksum::calculate(seed, m_ClassConditionalDensities);
}

bool CNaiveBayes::validate(const TDouble1VecVec &x) const
{
    auto class_ = m_ClassConditionalDensities.begin();
    if (   class_ != m_ClassConditionalDensities.end()
        && class_->second.s_ConditionalDensities.size() > 0
        && class_->second.s_ConditionalDensities.size() != x.size())
    {
        LOG_ERROR("Unexpected feature vector: " << core::CContainerPrinter::print(x));
        return false;
    }
    return true;
}

bool CNaiveBayes::SClass::acceptRestoreTraverser(const SDistributionRestoreParams &params,
                                                 core::CStateRestoreTraverser &traverser)
{
    do
    {
        const std::string &name{traverser.name()};
        RESTORE_BUILT_IN(COUNT_TAG, s_Count)
        RESTORE_SETUP_TEARDOWN(CONDITIONAL_DENSITY_FROM_PRIOR_TAG,
                               CNaiveBayesFeatureDensityFromPrior tmp,
                               traverser.traverseSubLevel(boost::bind(
                                       &CNaiveBayesFeatureDensityFromPrior::acceptRestoreTraverser,
                                       boost::ref(tmp), boost::cref(params), _1)),
                               s_ConditionalDensities.emplace_back(tmp.clone()))
        // Add other implementation's restore code here.
    }
    while (traverser.next());
    return true;
}

void CNaiveBayes::SClass::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertValue(COUNT_TAG, s_Count, core::CIEEE754::E_SinglePrecision);
    for (const auto &density : s_ConditionalDensities)
    {
        if (dynamic_cast<const CNaiveBayesFeatureDensityFromPrior*>(density.get()))
        {
            inserter.insertLevel(CONDITIONAL_DENSITY_FROM_PRIOR_TAG,
                                 boost::bind(&CNaiveBayesFeatureDensity::acceptPersistInserter,
                                             density.get(), _1));
            continue;
        }
        // Add other implementation's persist code here.
    }
}

void CNaiveBayes::SClass::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const
{
    core::CMemoryDebug::dynamicSize("s_ConditionalDensities", s_ConditionalDensities, mem);
}

std::size_t CNaiveBayes::SClass::memoryUsage() const
{
    return core::CMemory::dynamicSize(s_ConditionalDensities);
}

uint64_t CNaiveBayes::SClass::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, s_Count);
    return CChecksum::calculate(seed, s_ConditionalDensities);
}

}
}
