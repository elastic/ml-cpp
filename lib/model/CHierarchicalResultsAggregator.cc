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

#include <model/CHierarchicalResultsAggregator.h>

#include <core/CLogger.h>
#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/CMathsFuncs.h>
#include <maths/CTools.h>
#include <maths/Constants.h>
#include <maths/ProbabilityAggregators.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CAnomalyScore.h>
#include <model/CLimits.h>
#include <model/CProbabilityAndInfluenceCalculator.h>

#include <boost/bind.hpp>
#include <boost/container/flat_map.hpp>
#include <boost/container/flat_set.hpp>
#include <boost/range.hpp>
#include <boost/ref.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace ml {
namespace model {
namespace {

using TStoredStringPtr = CHierarchicalResults::TStoredStringPtr;
using TStoredStringPtrStoredStringPtrPr = CHierarchicalResults::TStoredStringPtrStoredStringPtrPr;
using TStoredStringPtrStoredStringPtrPrDoublePr =
    CHierarchicalResults::TStoredStringPtrStoredStringPtrPrDoublePr;
using TStoredStringPtrStoredStringPtrPrDoublePrVec =
    CHierarchicalResults::TStoredStringPtrStoredStringPtrPrDoublePrVec;

//! \brief Creates new detector equalizers.
class CDetectorEqualizerFactory {
public:
    CDetectorEqualizer make(const std::string& /*name1*/,
                            const std::string& /*name2*/,
                            const std::string& /*name3*/,
                            const std::string& /*name4*/) const {
        return CDetectorEqualizer();
    }

    CDetectorEqualizer make(const std::string& /*name1*/, const std::string& /*name2*/) const {
        return CDetectorEqualizer();
    }

    CDetectorEqualizer make(const std::string& /*name*/) const { return CDetectorEqualizer(); }
};

//! Check if the underlying strings are equal.
bool equal(const TStoredStringPtrStoredStringPtrPr& lhs,
           const TStoredStringPtrStoredStringPtrPr& rhs) {
    return *lhs.first == *rhs.first && *lhs.second == *rhs.second;
}

//! Compute the probability of \p influence.
bool influenceProbability(const TStoredStringPtrStoredStringPtrPrDoublePrVec& influences,
                          const TStoredStringPtr& influencerName,
                          const TStoredStringPtr& influencerValue,
                          double p,
                          double& result) {
    TStoredStringPtrStoredStringPtrPr influence(influencerName, influencerValue);

    std::size_t k{static_cast<std::size_t>(std::lower_bound(influences.begin(),
                                                            influences.end(),
                                                            influence,
                                                            maths::COrderings::SFirstLess()) -
                                           influences.begin())};

    if (k < influences.size() && equal(influences[k].first, influence)) {
        result = influences[k].second == 1.0
                     ? p
                     : std::exp(influences[k].second * maths::CTools::fastLog(p));
        return true;
    }

    return false;
}

const core::CHashing::CMurmurHash2String HASHER;
const std::string BUCKET_TAG("a");
const std::string INFLUENCER_BUCKET_TAG("b");
const std::string INFLUENCER_TAG("c");
const std::string PARTITION_TAG("d");
const std::string PERSON_TAG("e");
const std::string LEAF_TAG("f");

} // unnamed::

CHierarchicalResultsAggregator::CHierarchicalResultsAggregator(
    const CAnomalyDetectorModelConfig& modelConfig)
    : TBase(TDetectorEqualizer()),
      m_Job(E_NoOp),
      m_DecayRate(modelConfig.decayRate()),
      m_MaximumAnomalousProbability(modelConfig.maximumAnomalousProbability()) {
    this->refresh(modelConfig);
}

void CHierarchicalResultsAggregator::setJob(EJob job) {
    m_Job = job;
}

void CHierarchicalResultsAggregator::refresh(const CAnomalyDetectorModelConfig& modelConfig) {
    m_DecayRate = modelConfig.decayRate();
    m_MaximumAnomalousProbability = modelConfig.maximumAnomalousProbability();
    for (std::size_t i = 0u; i < model_t::NUMBER_AGGREGATION_STYLES; ++i) {
        for (std::size_t j = 0u; j < model_t::NUMBER_AGGREGATION_PARAMS; ++j) {
            m_Parameters[i][j] =
                modelConfig.aggregationStyleParam(static_cast<model_t::EAggregationStyle>(i),
                                                  static_cast<model_t::EAggregationParam>(j));
        }
    }
}

void CHierarchicalResultsAggregator::clear(void) {
    this->TBase::clear();
}

void CHierarchicalResultsAggregator::visit(const CHierarchicalResults& /*results*/,
                                           const TNode& node,
                                           bool pivot) {
    if (isLeaf(node)) {
        this->aggregateLeaf(node);
    } else {
        this->aggregateNode(node, pivot);
    }
}

void CHierarchicalResultsAggregator::propagateForwardByTime(double time) {
    if (time < 0.0) {
        LOG_ERROR("Can't propagate normalizer backwards in time");
        return;
    }
    double factor{
        std::exp(-m_DecayRate * CDetectorEqualizer::largestProbabilityToCorrect() * time)};
    this->age(boost::bind(&TDetectorEqualizer::age, _1, factor));
}

void CHierarchicalResultsAggregator::acceptPersistInserter(
    core::CStatePersistInserter& inserter) const {
    inserter.insertLevel(BUCKET_TAG,
                         boost::bind(&TDetectorEqualizer::acceptPersistInserter,
                                     boost::cref(this->bucketElement()),
                                     _1));
    core::CPersistUtils::persist(INFLUENCER_BUCKET_TAG, this->influencerBucketSet(), inserter);
    core::CPersistUtils::persist(INFLUENCER_TAG, this->influencerSet(), inserter);
    core::CPersistUtils::persist(PARTITION_TAG, this->partitionSet(), inserter);
    core::CPersistUtils::persist(PERSON_TAG, this->personSet(), inserter);
    core::CPersistUtils::persist(LEAF_TAG, this->leafSet(), inserter);
}

bool CHierarchicalResultsAggregator::acceptRestoreTraverser(
    core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE(BUCKET_TAG,
                traverser.traverseSubLevel(boost::bind(&TDetectorEqualizer::acceptRestoreTraverser,
                                                       boost::ref(this->bucketElement()),
                                                       _1)))
        RESTORE(INFLUENCER_BUCKET_TAG,
                core::CPersistUtils::restore(INFLUENCER_BUCKET_TAG,
                                             this->influencerBucketSet(),
                                             traverser));
        RESTORE(INFLUENCER_TAG,
                core::CPersistUtils::restore(INFLUENCER_TAG, this->influencerSet(), traverser));
        RESTORE(PARTITION_TAG,
                core::CPersistUtils::restore(PARTITION_TAG, this->partitionSet(), traverser));
        RESTORE(PERSON_TAG, core::CPersistUtils::restore(PERSON_TAG, this->personSet(), traverser));
        RESTORE(LEAF_TAG, core::CPersistUtils::restore(LEAF_TAG, this->leafSet(), traverser));
    } while (traverser.next());
    return true;
}

uint64_t CHierarchicalResultsAggregator::checksum(void) const {
    uint64_t seed = static_cast<uint64_t>(m_DecayRate);
    seed = maths::CChecksum::calculate(seed, m_Parameters);
    seed = maths::CChecksum::calculate(seed, m_MaximumAnomalousProbability);
    return this->TBase::checksum(seed);
}

void CHierarchicalResultsAggregator::aggregateLeaf(const TNode& node) {
    if (isSimpleCount(node)) {
        return;
    }

    int detector{node.s_Detector};
    double probability{node.probability()};
    if (!maths::CMathsFuncs::isFinite(probability)) {
        probability = 1.0;
    }
    probability = maths::CTools::truncate(probability, maths::CTools::smallestProbability(), 1.0);
    this->correctProbability(node, false, detector, probability);
    model_t::EAggregationStyle style{isAttribute(node) ? model_t::E_AggregateAttributes
                                                       : model_t::E_AggregatePeople};

    node.s_AnnotatedProbability.s_Probability = probability;
    node.s_AggregationStyle = style;
    node.s_SmallestChildProbability = probability;
    node.s_SmallestDescendantProbability = probability;
    node.s_RawAnomalyScore = maths::CTools::deviation(probability);
}

void CHierarchicalResultsAggregator::aggregateNode(const TNode& node, bool pivot) {
    LOG_TRACE("node = " << node.print() << ", pivot = " << pivot);

    std::size_t numberDetectors;
    TIntSizePrDouble1VecUMap partition[N];
    if (!this->partitionChildProbabilities(node, pivot, numberDetectors, partition)) {
        return;
    }
    LOG_TRACE("partition = " << core::CContainerPrinter::print(partition));

    int detector;
    int aggregation;
    TDouble1Vec detectorProbabilities;
    this->detectorProbabilities(node,
                                pivot,
                                numberDetectors,
                                partition,
                                detector,
                                aggregation,
                                detectorProbabilities);
    LOG_TRACE("detector = " << detector << ", aggregation = " << aggregation
                            << ", detector probabilities = " << detectorProbabilities);

    const double* params{m_Parameters[model_t::E_AggregateDetectors]};
    CAnomalyScore::compute(params[model_t::E_JointProbabilityWeight],
                           params[model_t::E_ExtremeProbabilityWeight],
                           static_cast<std::size_t>(params[model_t::E_MinExtremeSamples]),
                           static_cast<std::size_t>(params[model_t::E_MaxExtremeSamples]),
                           m_MaximumAnomalousProbability,
                           detectorProbabilities,
                           node.s_RawAnomalyScore,
                           node.s_AnnotatedProbability.s_Probability);
    node.s_Detector = detector;
    node.s_AggregationStyle = aggregation;
    LOG_TRACE("probability = " << node.probability());
}

bool CHierarchicalResultsAggregator::partitionChildProbabilities(
    const TNode& node,
    bool pivot,
    std::size_t& numberDetectors,
    TIntSizePrDouble1VecUMap (&partition)[N]) {
    using TSizeFSet = boost::container::flat_set<std::size_t>;
    using TMinAccumulator = maths::CBasicStatistics::SMin<double>::TAccumulator;

    for (std::size_t i = 0u; i < N; ++i) {
        partition[i].reserve(node.s_Children.size());
    }

    bool haveResult{false};
    TSizeFSet detectors;
    TMinAccumulator pMinChild;
    TMinAccumulator pMinDescendent;

    for (const auto& child : node.s_Children) {
        if (isSimpleCount(*child)) {
            continue;
        }

        double probability{child->probability()};
        std::size_t key{0};
        if (pivot && !isRoot(node) &&
            !influenceProbability(child->s_AnnotatedProbability.s_Influences,
                                  node.s_Spec.s_PersonFieldName,
                                  node.s_Spec.s_PersonFieldValue,
                                  probability,
                                  probability)) {
            LOG_ERROR("Couldn't find influence for " << child->print());
            continue;
        } else {
            key = this->hash(*child);
        }

        haveResult = true;
        pMinChild.add(probability);
        if (isTypeForWhichWeWriteResults(*child, pivot)) {
            pMinDescendent.add(probability);
        }
        pMinDescendent.add(child->s_SmallestDescendantProbability);

        model_t::EAggregationStyle style{
            static_cast<model_t::EAggregationStyle>(child->s_AggregationStyle)};
        switch (style) {
        case model_t::E_AggregatePeople:
        case model_t::E_AggregateAttributes:
            detectors.insert(child->s_Detector);
            partition[style][{child->s_Detector, key}].push_back(probability);
            break;
        case model_t::E_AggregateDetectors:
            LOG_ERROR("Unexpected aggregation style for " << child->print());
            continue;
        }
    }

    if (haveResult) {
        node.s_SmallestChildProbability =
            maths::CTools::truncate(pMinChild[0], maths::CTools::smallestProbability(), 1.0);
        node.s_SmallestDescendantProbability =
            maths::CTools::truncate(pMinDescendent[0], maths::CTools::smallestProbability(), 1.0);
    }
    numberDetectors = detectors.size();
    LOG_TRACE("detector = " << core::CContainerPrinter::print(detectors));

    return haveResult;
}

void CHierarchicalResultsAggregator::detectorProbabilities(
    const TNode& node,
    bool pivot,
    std::size_t numberDetectors,
    const TIntSizePrDouble1VecUMap (&partition)[N],
    int& detector,
    int& aggregation,
    TDouble1Vec& probabilities) {
    using TDouble1Vec = core::CSmallVector<double, 1>;
    using TIntDouble1VecFMap = boost::container::flat_map<int, TDouble1Vec>;

    int fallback{static_cast<int>(model_t::E_AggregatePeople)};
    detector = -3;
    aggregation =
        (pivot || isPartition(node) || (isPopulation(node) && isPerson(node))) ? fallback : -1;

    TIntDouble1VecFMap detectorProbabilities;
    detectorProbabilities.reserve(numberDetectors);
    for (int i = 0u; i < static_cast<int>(N); ++i) {
        const double* params{m_Parameters[i]};
        for (const auto& subset : partition[i]) {
            int detector_{subset.first.first};
            double probability;
            if (subset.second.size() == 1) {
                probability = subset.second[0];
            } else {
                double rawAnomalyScore;
                CAnomalyScore::compute(params[model_t::E_JointProbabilityWeight],
                                       params[model_t::E_ExtremeProbabilityWeight],
                                       static_cast<std::size_t>(
                                           params[model_t::E_MinExtremeSamples]),
                                       static_cast<std::size_t>(
                                           params[model_t::E_MaxExtremeSamples]),
                                       m_MaximumAnomalousProbability,
                                       subset.second,
                                       rawAnomalyScore,
                                       probability);
            }
            if (!maths::CMathsFuncs::isFinite(probability)) {
                probability = 1.0;
            }

            detectorProbabilities[detector_].push_back(probability);

            switch (detector) {
            case -3:
                detector = detector_; /*first value we've seen*/
                break;
            case -2: /*we have a mix of detectors*/
                break;
            default:
                detector = (detector != detector_ ? -2 : detector_);
                break;
            }
            switch (aggregation) {
            case -1:
                aggregation = i; /*first value we've seen*/
                break;
            default:
                aggregation = (aggregation != i ? fallback : i);
                break;
            }
        }
    }

    probabilities.reserve(numberDetectors);
    for (const auto& dp : detectorProbabilities) {
        double probability{dp.second[0]};
        if (dp.second.size() > 1) {
            const double* params{m_Parameters[model_t::E_AggregatePeople]};
            double rawAnomalyScore;
            CAnomalyScore::compute(params[model_t::E_JointProbabilityWeight],
                                   params[model_t::E_ExtremeProbabilityWeight],
                                   static_cast<std::size_t>(params[model_t::E_MinExtremeSamples]),
                                   static_cast<std::size_t>(params[model_t::E_MaxExtremeSamples]),
                                   m_MaximumAnomalousProbability,
                                   dp.second,
                                   rawAnomalyScore,
                                   probability);
        }
        probabilities.push_back(this->correctProbability(node, pivot, dp.first, probability));
    }
}

std::size_t CHierarchicalResultsAggregator::hash(const TNode& node) const {
    std::size_t result{HASHER(node.s_Spec.s_PartitionFieldValue)};
    if (node.s_Spec.s_IsPopulation) {
        boost::hash_combine(result, HASHER(node.s_Spec.s_PersonFieldValue));
    }
    return result;
}

double CHierarchicalResultsAggregator::correctProbability(const TNode& node,
                                                          bool pivot,
                                                          int detector,
                                                          double probability) {
    using TMaxAccumulator = maths::CBasicStatistics::SMax<double>::TAccumulator;

    if (probability < CDetectorEqualizer::largestProbabilityToCorrect()) {
        CDetectorEqualizerFactory factory;
        TDetectorEqualizerPtrVec equalizers;
        this->elements(node, pivot, factory, equalizers);
        TMaxAccumulator corrected;
        for (auto&& equalizer : equalizers) {
            switch (m_Job) {
            case E_UpdateAndCorrect:
                equalizer->add(detector, probability);
                corrected.add(equalizer->correct(detector, probability));
                break;
            case E_Correct:
                corrected.add(equalizer->correct(detector, probability));
                break;
            case E_NoOp:
                break;
            }
        }
        if (corrected.count() > 0) {
            probability = corrected[0];
        }
    }

    return probability;
}
}
}
