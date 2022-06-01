/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <model/CHierarchicalResultsNormalizer.h>

#include <core/CBase64Filter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CStringUtils.h>

#include <maths/common/COrderings.h>
#include <maths/common/CTools.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/FunctionTypes.h>

#include <rapidjson/document.h>

#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>

#include <algorithm>
#include <sstream>

namespace ml {
namespace model {

namespace {
// The bucket cue is "sysChange" for historical reasons.  Do NOT tidy this up
// unless you fully understand the implications.  In particular existing
// persisted quantiles will all need to be regenerated.
const std::string BUCKET_CUE("sysChange");
const std::string INFLUENCER_BUCKET_CUE_PREFIX("inflb");
const std::string INFLUENCER_CUE_PREFIX("infl");
const std::string PARTITION_CUE_PREFIX("part");
const std::string PERSON_CUE_PREFIX("per");
const std::string LEAF_CUE_PREFIX("leaf");
const std::string EMPTY_STRING;
}

namespace hierarchical_results_normalizer_detail {

SNormalizer::SNormalizer(const std::string& description, const TNormalizerPtr& normalizer)
    : s_Description(description), s_Normalizer(normalizer) {
}

void SNormalizer::clear() {
    s_Normalizer->clear();
}

void SNormalizer::propagateForwardByTime(double time) {
    s_Normalizer->propagateForwardByTime(time);
}

std::uint64_t SNormalizer::checksum() const {
    std::uint64_t seed = maths::common::CChecksum::calculate(0, s_Description);
    return maths::common::CChecksum::calculate(seed, s_Normalizer);
}
}

CHierarchicalResultsNormalizer::CHierarchicalResultsNormalizer(const CAnomalyDetectorModelConfig& modelConfig)
    : TBase(TNormalizer(std::string(),
                        std::make_shared<CAnomalyScore::CNormalizer>(modelConfig))),
      m_Job(E_NoOp), m_ModelConfig(modelConfig), m_HasLastUpdateCausedBigChange(false) {
}

void CHierarchicalResultsNormalizer::setJob(EJob job) {
    m_Job = job;
}

void CHierarchicalResultsNormalizer::clear() {
    this->TBase::clear();
    m_HasLastUpdateCausedBigChange = false;
}

void CHierarchicalResultsNormalizer::resetBigChange() {
    m_HasLastUpdateCausedBigChange = false;
}

void CHierarchicalResultsNormalizer::visit(const CHierarchicalResults& /*results*/,
                                           const TNode& node,
                                           bool pivot) {
    TNormalizerPtrVec normalizers;
    this->elements(node, pivot, CNormalizerFactory{m_ModelConfig}, normalizers);

    if (normalizers.empty()) {
        return;
    }

    // We need to reset this flag if the normalizer is to be used
    // for results for members of a population analysis.

    // This has to use the deviation of the probability rather than
    // the anomaly score stored on the bucket because the later is
    // scaled so that it sums to the bucket anomaly score.
    double score = node.probability() > m_ModelConfig.maximumAnomalousProbability()
                       ? 0.0
                       : maths::common::CTools::anomalyScore(node.probability());

    CAnomalyScore::CNormalizer::CMaximumScoreScope scope{
        dereferenceOrEmpty(node.s_Spec.s_PartitionFieldName),
        dereferenceOrEmpty(node.s_Spec.s_PartitionFieldValue),
        dereferenceOrEmpty(node.s_Spec.s_PersonFieldName),
        dereferenceOrEmpty(node.s_Spec.s_PersonFieldValue)};

    switch (m_Job) {
    case E_RefreshSettings:
        for (auto& normalizer : normalizers) {
            normalizer->s_Normalizer->isForMembersOfPopulation(isMemberOfPopulation(node));
        }
        break;
    case E_UpdateQuantiles:
        if (node.s_AnnotatedProbability.s_ShouldUpdateQuantiles == false) {
            LOG_TRACE(<< "NOT Updating quantiles for " << node.s_Spec.print()
                      << ", scope = '" << scope.print() << "', score = " << score);
            return;
        }
        for (auto& normalizer : normalizers) {
            m_HasLastUpdateCausedBigChange |=
                normalizer->s_Normalizer->updateQuantiles(scope, score);
        }
        break;
    case E_NormalizeScores:
        // Normalize with the lowest suitable normalizer.
        if (normalizers[0]->s_Normalizer->normalize(scope, score) == false) {
            LOG_ERROR(<< "Failed to normalize " << score << " for "
                      << node.s_Spec.print());
        }
        node.s_NormalizedAnomalyScore = score;
        break;
    case E_NoOp:
        LOG_ERROR(<< "Calling normalize without setting job");
        break;
    }
}

void CHierarchicalResultsNormalizer::propagateForwardByTime(double time) {
    if (time < 0.0) {
        LOG_ERROR(<< "Can't propagate normalizer backwards in time");
        return;
    }
    this->age(std::bind(&TNormalizer::propagateForwardByTime, std::placeholders::_1, time));
}

bool CHierarchicalResultsNormalizer::hasLastUpdateCausedBigChange() const {
    return m_HasLastUpdateCausedBigChange;
}

void CHierarchicalResultsNormalizer::toJson(core_t::TTime time,
                                            const std::string& key,
                                            std::string& json,
                                            bool makeArray) const {
    std::ostringstream compressedStream;
    using TFilteredOutput = boost::iostreams::filtering_stream<boost::iostreams::output>;
    {
        TFilteredOutput outFilter;
        outFilter.push(boost::iostreams::gzip_compressor());
        outFilter.push(core::CBase64Encoder());
        outFilter.push(compressedStream);
        if (makeArray) {
            outFilter << '[';
        }

        for (std::size_t i = 0; i < this->leafSet().size(); ++i) {
            const TWord& word = this->leafSet()[i].first;
            const TNormalizer& normalizer = this->leafSet()[i].second;
            CAnomalyScore::normalizerToJson(*normalizer.s_Normalizer, key, leafCue(word),
                                            normalizer.s_Description, time, outFilter);
            outFilter << ',';
        }

        for (std::size_t i = 0; i < this->personSet().size(); ++i) {
            const TWord& word = this->personSet()[i].first;
            const TNormalizer& normalizer = this->personSet()[i].second;
            CAnomalyScore::normalizerToJson(*normalizer.s_Normalizer, key,
                                            personCue(word), normalizer.s_Description,
                                            time, outFilter);
            outFilter << ',';
        }

        for (std::size_t i = 0; i < this->partitionSet().size(); ++i) {
            const TWord& word = this->partitionSet()[i].first;
            const TNormalizer& normalizer = this->partitionSet()[i].second;
            CAnomalyScore::normalizerToJson(*normalizer.s_Normalizer, key,
                                            partitionCue(word), normalizer.s_Description,
                                            time, outFilter);
            outFilter << ',';
        }

        for (std::size_t i = 0; i < this->influencerSet().size(); ++i) {
            const TWord& word = this->influencerSet()[i].first;
            const TNormalizer& normalizer = this->influencerSet()[i].second;
            CAnomalyScore::normalizerToJson(*normalizer.s_Normalizer, key,
                                            influencerCue(word),
                                            normalizer.s_Description, time, outFilter);
            outFilter << ',';
        }

        for (std::size_t i = 0; i < this->influencerBucketSet().size(); ++i) {
            const TWord& word = this->influencerBucketSet()[i].first;
            const TNormalizer& normalizer = this->influencerBucketSet()[i].second;
            CAnomalyScore::normalizerToJson(*normalizer.s_Normalizer, key,
                                            influencerBucketCue(word),
                                            normalizer.s_Description, time, outFilter);
            outFilter << ',';
        }

        // Put the bucket normalizer last so that incomplete restorations can be
        // detected by checking whether the bucket normalizer is restored
        CAnomalyScore::normalizerToJson(*this->bucketElement().s_Normalizer, key,
                                        bucketCue(), "root", time, outFilter);
        if (makeArray) {
            outFilter << ']';
        }
    }
    json = compressedStream.str();
}

CHierarchicalResultsNormalizer::ERestoreOutcome
CHierarchicalResultsNormalizer::fromJsonStream(std::istream& inputStream) {
    bool isBucketNormalizerRestored = false;

    this->TBase::clear();

    // The state may be compressed or uncompressed. We can distinguish because
    // the first character of a base64 encoded zlib compressed stream will never
    // be a bracket.
    using TFilteredInput = boost::iostreams::filtering_stream<boost::iostreams::input>;
    TFilteredInput filteredInput;
    std::istream* streamToTraverse = nullptr;
    switch (inputStream.peek()) {
    case EOF:
        LOG_DEBUG(<< "No normalizer state to restore");
        // this is not an error
        return E_Ok;
    case '[':
    case '{':
        LOG_DEBUG(<< "Detected uncompressed normalizer state");
        streamToTraverse = &inputStream;
        break;
    default:
        LOG_DEBUG(<< "Detected compressed normalizer state");
        filteredInput.push(boost::iostreams::gzip_decompressor());
        filteredInput.push(core::CBase64Decoder{});
        filteredInput.push(inputStream);
        streamToTraverse = &filteredInput;
        break;
    }

    core::CJsonStateRestoreTraverser traverser(*streamToTraverse);

    do {
        // Call name() to prime the traverser if it hasn't started
        if (traverser.name().empty()) {
            if (traverser.isEof()) {
                LOG_DEBUG(<< "No normalizer state to restore");
                // this is not an error
                return E_Ok;
            }
        }

        // The MLCUE_ATTRIBUTE should always be the first field
        if (traverser.name() != CAnomalyScore::MLCUE_ATTRIBUTE) {
            if (!traverser.next()) {
                LOG_INFO(<< "No normalizer state to restore");
                return E_Ok;
            }

            LOG_ERROR(<< "Expected " << CAnomalyScore::MLCUE_ATTRIBUTE << " field in quantiles JSON got "
                      << traverser.name() << " = " << traverser.value());
            return E_Corrupt;
        }

        const std::string cue(traverser.value());

        if (cue == BUCKET_CUE) {
            if (CAnomalyScore::normalizerFromJson(
                    traverser, *this->bucketElement().s_Normalizer) == false) {
                LOG_ERROR(<< "Unable to restore bucket normalizer");
                return E_Corrupt;
            }

            isBucketNormalizerRestored = true;
        } else {
            TWordNormalizerPrVec* normalizerVec = nullptr;
            TDictionary::TUInt64Array hashArray;
            if (!this->parseCue(cue, normalizerVec, hashArray)) {
                return E_Corrupt;
            }

            if (normalizerVec != nullptr) {
                if (!traverser.next()) {
                    LOG_ERROR(<< "Cannot restore hierarchical normalizer - end of object reached when "
                              << CAnomalyScore::MLKEY_ATTRIBUTE << " was expected");
                    return E_Corrupt;
                }

                if (!traverser.next()) {
                    LOG_ERROR(<< "Cannot restore hierarchical normalizer - end of object reached when "
                              << CAnomalyScore::MLQUANTILESDESCRIPTION_ATTRIBUTE
                              << " was expected");
                    return E_Corrupt;
                }

                if (traverser.name() != CAnomalyScore::MLQUANTILESDESCRIPTION_ATTRIBUTE) {
                    LOG_ERROR(<< "Cannot restore hierarchical normalizer - "
                              << CAnomalyScore::MLQUANTILESDESCRIPTION_ATTRIBUTE << " element expected but found "
                              << traverser.name() << '=' << traverser.value());
                    return E_Corrupt;
                }

                std::string quantileDesc(traverser.value());

                auto normalizer = std::make_shared<CAnomalyScore::CNormalizer>(m_ModelConfig);
                normalizerVec->emplace_back(TWord(hashArray),
                                            TNormalizer(quantileDesc, normalizer));
                if (CAnomalyScore::normalizerFromJson(traverser, *normalizer) == false) {
                    LOG_ERROR(<< "Unable to restore normalizer with cue " << cue);
                    return E_Corrupt;
                }
            }
        }
    } while (traverser.nextObject());

    this->sort();

    LOG_DEBUG(<< this->influencerBucketSet().size() << " influencer bucket normalizers, "
              << this->influencerSet().size() << " influencer normalizers, "
              << this->partitionSet().size() << " partition normalizers, "
              << this->personSet().size() << " person normalizers and "
              << this->leafSet().size() << " leaf normalizers restored from JSON stream");

    return isBucketNormalizerRestored ? E_Ok : E_Incomplete;
}

const CAnomalyScore::CNormalizer& CHierarchicalResultsNormalizer::bucketNormalizer() const {
    return *this->bucketElement().s_Normalizer;
}

const CAnomalyScore::CNormalizer*
CHierarchicalResultsNormalizer::influencerBucketNormalizer(const std::string& influencerFieldName) const {
    const TNormalizer* normalizer = this->influencerBucketElement(influencerFieldName);
    return normalizer ? normalizer->s_Normalizer.get() : nullptr;
}

const CAnomalyScore::CNormalizer*
CHierarchicalResultsNormalizer::influencerNormalizer(const std::string& influencerFieldName) const {
    const TNormalizer* normalizer = this->influencerElement(influencerFieldName);
    return normalizer ? normalizer->s_Normalizer.get() : nullptr;
}

const CAnomalyScore::CNormalizer*
CHierarchicalResultsNormalizer::partitionNormalizer(const std::string& partitionFieldName) const {
    const TNormalizer* normalizer = this->partitionElement(partitionFieldName);
    return normalizer ? normalizer->s_Normalizer.get() : nullptr;
}

const CAnomalyScore::CNormalizer*
CHierarchicalResultsNormalizer::leafNormalizer(const std::string& partitionFieldName,
                                               const std::string& personFieldName,
                                               const std::string& functionName,
                                               const std::string& valueFieldName) const {
    const TNormalizer* normalizer = this->leafElement(
        partitionFieldName, personFieldName, functionName, valueFieldName);
    return normalizer ? normalizer->s_Normalizer.get() : nullptr;
}

bool CHierarchicalResultsNormalizer::isMemberOfPopulation(const TNode& node,
                                                          std::function<bool(const TNode&)> test) {
    if (isPopulation(node)) {
        return true;
    }

    // Check whether node has a distinct person field name and value
    // and if it does whether these match any of its descendant results
    // which are from a population analysis. Note that test is only null
    // the first time this function is invoked for the node under test.
    // So the person field name and value in the lambda are set to the
    // values for this node.

    if (test == nullptr) {
        const std::string& personName = *node.s_Spec.s_PersonFieldName;
        const std::string& personValue = *node.s_Spec.s_PersonFieldValue;
        if (personName.empty() || personValue.empty()) {
            return false;
        }
        test = [&personName, &personValue](const TNode& child) {
            return isPopulation(child) && *child.s_Spec.s_PersonFieldName == personName &&
                   *child.s_Spec.s_PersonFieldValue == personValue;
        };
    }

    for (const auto& child : node.s_Children) {
        if (test(*child) || (isLeaf(*child) == false && isMemberOfPopulation(*child))) {
            return true;
        }
    }
    return false;
}

bool CHierarchicalResultsNormalizer::parseCue(const std::string& cue,
                                              TWordNormalizerPrVec*& normalizers,
                                              TDictionary::TUInt64Array& hashArray) {
    normalizers = nullptr;
    std::size_t hashStartPos = 0;

    if (cue.compare(0, INFLUENCER_BUCKET_CUE_PREFIX.length(),
                    INFLUENCER_BUCKET_CUE_PREFIX) == 0) {
        normalizers = &this->influencerBucketSet();
        hashStartPos = INFLUENCER_BUCKET_CUE_PREFIX.length();
    } else if (cue.compare(0, INFLUENCER_CUE_PREFIX.length(), INFLUENCER_CUE_PREFIX) == 0) {
        normalizers = &this->influencerSet();
        hashStartPos = INFLUENCER_CUE_PREFIX.length();
    } else if (cue.compare(0, PARTITION_CUE_PREFIX.length(), PARTITION_CUE_PREFIX) == 0) {
        normalizers = &this->partitionSet();
        hashStartPos = PARTITION_CUE_PREFIX.length();
    } else if (cue.compare(0, PERSON_CUE_PREFIX.length(), PERSON_CUE_PREFIX) == 0) {
        normalizers = &this->personSet();
        hashStartPos = PERSON_CUE_PREFIX.length();
    } else if (cue.compare(0, LEAF_CUE_PREFIX.length(), LEAF_CUE_PREFIX) == 0) {
        normalizers = &this->leafSet();
        hashStartPos = LEAF_CUE_PREFIX.length();
    } else {
        LOG_WARN(<< "Did not understand normalizer cue " << cue);
        return true;
    }

    LOG_TRACE(<< "cue = " << cue << ", hash = " << cue.substr(hashStartPos));

    if (core::CStringUtils::stringToType(cue.substr(hashStartPos), hashArray[0]) == false) {
        LOG_ERROR(<< "Unable to parse normalizer hash from cue " << cue
                  << " starting at position " << hashStartPos);
        return false;
    }

    return true;
}

const std::string& CHierarchicalResultsNormalizer::bucketCue() {
    return BUCKET_CUE;
}

std::string CHierarchicalResultsNormalizer::influencerBucketCue(const TWord& word) {
    return INFLUENCER_BUCKET_CUE_PREFIX + core::CStringUtils::typeToString(word.hash64());
}

std::string CHierarchicalResultsNormalizer::influencerCue(const TWord& word) {
    return INFLUENCER_CUE_PREFIX + core::CStringUtils::typeToString(word.hash64());
}

std::string CHierarchicalResultsNormalizer::partitionCue(const TWord& word) {
    return PARTITION_CUE_PREFIX + core::CStringUtils::typeToString(word.hash64());
}

std::string CHierarchicalResultsNormalizer::personCue(const TWord& word) {
    return PERSON_CUE_PREFIX + core::CStringUtils::typeToString(word.hash64());
}

std::string CHierarchicalResultsNormalizer::leafCue(const TWord& word) {
    return LEAF_CUE_PREFIX + core::CStringUtils::typeToString(word.hash64());
}

const std::string&
CHierarchicalResultsNormalizer::dereferenceOrEmpty(const core::CStoredStringPtr& stringPtr) {
    return stringPtr ? *stringPtr : EMPTY_STRING;
}

CHierarchicalResultsNormalizer::CNormalizerFactory::CNormalizerFactory(const CAnomalyDetectorModelConfig& modelConfig)
    : m_ModelConfig{modelConfig} {
}

CHierarchicalResultsNormalizer::TNormalizer
CHierarchicalResultsNormalizer::CNormalizerFactory::make(const TNode& node, bool) const {
    return {*node.s_Spec.s_PartitionFieldName + ' ' + *node.s_Spec.s_PersonFieldName +
                ' ' + *node.s_Spec.s_FunctionName + ' ' + *node.s_Spec.s_ValueFieldName,
            std::make_shared<CAnomalyScore::CNormalizer>(m_ModelConfig)};
}
}
}
