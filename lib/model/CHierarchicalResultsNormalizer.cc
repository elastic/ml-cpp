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

#include <model/CHierarchicalResultsNormalizer.h>

#include <core/CJsonStateRestoreTraverser.h>
#include <core/CStringUtils.h>

#include <maths/COrderings.h>
#include <maths/CTools.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/FunctionTypes.h>

#include <boost/bind.hpp>
#include <boost/make_shared.hpp>

#include <rapidjson/document.h>

#include <algorithm>

namespace ml
{
namespace model
{

namespace
{

//! \brief Creates new normalizer instances.
class CNormalizerFactory
{
    public:
        typedef CHierarchicalResultsNormalizer::TNormalizer TNormalizer;

        CNormalizerFactory(const CAnomalyDetectorModelConfig &modelConfig) : m_ModelConfig(modelConfig) {}

        TNormalizer make(const std::string &name1,
                         const std::string &name2,
                         const std::string &name3,
                         const std::string &name4) const
        {
            return make(name1 + ' ' + name2 + ' ' + name3 + ' ' + name4);
        }

        TNormalizer make(const std::string &name1, const std::string &name2) const
        {
            return make(name1 + ' ' + name2);
        }

        TNormalizer make(const std::string &name) const
        {
            return TNormalizer(name, boost::make_shared<CAnomalyScore::CNormalizer>(m_ModelConfig));
        }

    private:
        //! The model configuration file.
        const CAnomalyDetectorModelConfig &m_ModelConfig;
};

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

namespace hierarchical_results_normalizer_detail
{

SNormalizer::SNormalizer(const std::string &description, const TNormalizerPtr &normalizer) :
        s_Description(description),
        s_Normalizer(normalizer)
{
}

void SNormalizer::clear(void)
{
    s_Normalizer->clear();
}

void SNormalizer::propagateForwardByTime(double time)
{
    s_Normalizer->propagateForwardByTime(time);
}

uint64_t SNormalizer::checksum(void) const
{
    uint64_t seed = maths::CChecksum::calculate(0, s_Description);
    return maths::CChecksum::calculate(seed, s_Normalizer);
}

}

CHierarchicalResultsNormalizer::CHierarchicalResultsNormalizer(const CAnomalyDetectorModelConfig &modelConfig) :
        TBase(TNormalizer(std::string(), boost::make_shared<CAnomalyScore::CNormalizer>(modelConfig))),
        m_Job(E_NoOp),
        m_ModelConfig(modelConfig),
        m_HasLastUpdateCausedBigChange(false)
{
}

void CHierarchicalResultsNormalizer::setJob(EJob job)
{
    m_Job = job;
}

void CHierarchicalResultsNormalizer::clear(void)
{
    this->TBase::clear();
    m_HasLastUpdateCausedBigChange = false;
}

void CHierarchicalResultsNormalizer::resetBigChange()
{
    m_HasLastUpdateCausedBigChange = false;
}

void CHierarchicalResultsNormalizer::visit(const CHierarchicalResults &/*results*/,
                                           const TNode &node,
                                           bool pivot)
{
    CNormalizerFactory factory(m_ModelConfig);
    TNormalizerPtrVec normalizers;
    this->elements(node, pivot, factory, normalizers, m_ModelConfig.perPartitionNormalization());

    if (normalizers.empty())
    {
        return;
    }

    // This has to use the deviation of the probability rather than
    // the anomaly score stored on the bucket because the later is
    // scaled so that it sums to the bucket anomaly score.
    double score = node.probability() > m_ModelConfig.maximumAnomalousProbability() ?
                   0.0 :
                   maths::CTools::anomalyScore(node.probability());

    switch (m_Job)
    {
    case E_Update:
        for (std::size_t i = 0u; i < normalizers.size(); ++i)
        {
            m_HasLastUpdateCausedBigChange |= normalizers[i]->s_Normalizer->updateQuantiles(score);
        }
        break;
    case E_Normalize:
        // Normalize with the lowest suitable normalizer.
        if (!normalizers[0]->s_Normalizer->normalize(score))
        {
            LOG_ERROR("Failed to normalize " << score << " for " << node.s_Spec.print());
        }
        node.s_NormalizedAnomalyScore = score;
        break;
    case E_NoOp:
        LOG_ERROR("Calling normalize without setting job");
        break;
    }
}

void CHierarchicalResultsNormalizer::propagateForwardByTime(double time)
{
    if (time < 0.0)
    {
        LOG_ERROR("Can't propagate normalizer backwards in time");
        return;
    }
    this->age(boost::bind(&TNormalizer::propagateForwardByTime, _1, time));
}

bool CHierarchicalResultsNormalizer::hasLastUpdateCausedBigChange(void) const
{
    return m_HasLastUpdateCausedBigChange;
}

void CHierarchicalResultsNormalizer::toJson(core_t::TTime time,
                                            const std::string &key,
                                            std::string &json,
                                            bool makeArray) const
{
    TStrVec jsonVec(  1 // m_RootNormalizer
                    + this->influencerBucketSet().size()
                    + this->influencerSet().size()
                    + this->partitionSet().size()
                    + this->personSet().size()
                    + this->leafSet().size());
    std::size_t index = 0;

    for (std::size_t i = 0; i < this->leafSet().size(); ++i)
    {
        const TWord &word = this->leafSet()[i].first;
        const TNormalizer &normalizer = this->leafSet()[i].second;
        CAnomalyScore::normalizerToJson(*normalizer.s_Normalizer,
                                        key,
                                        leafCue(word),
                                        normalizer.s_Description,
                                        time,
                                        jsonVec[index++]);
    }

    for (std::size_t i = 0; i < this->personSet().size(); ++i)
    {
        const TWord &word = this->personSet()[i].first;
        const TNormalizer &normalizer = this->personSet()[i].second;
        CAnomalyScore::normalizerToJson(*normalizer.s_Normalizer,
                                        key,
                                        personCue(word),
                                        normalizer.s_Description,
                                        time,
                                        jsonVec[index++]);
    }

    for (std::size_t i = 0; i < this->partitionSet().size(); ++i)
    {
        const TWord &word = this->partitionSet()[i].first;
        const TNormalizer &normalizer = this->partitionSet()[i].second;
        CAnomalyScore::normalizerToJson(*normalizer.s_Normalizer,
                                        key,
                                        partitionCue(word),
                                        normalizer.s_Description,
                                        time,
                                        jsonVec[index++]);
    }

    for (std::size_t i = 0; i < this->influencerSet().size(); ++i)
    {
        const TWord &word = this->influencerSet()[i].first;
        const TNormalizer &normalizer = this->influencerSet()[i].second;
        CAnomalyScore::normalizerToJson(*normalizer.s_Normalizer,
                                        key,
                                        influencerCue(word),
                                        normalizer.s_Description,
                                        time,
                                        jsonVec[index++]);
    }

    for (std::size_t i = 0; i < this->influencerBucketSet().size(); ++i)
    {
        const TWord &word = this->influencerBucketSet()[i].first;
        const TNormalizer &normalizer = this->influencerBucketSet()[i].second;
        CAnomalyScore::normalizerToJson(*normalizer.s_Normalizer,
                                        key,
                                        influencerBucketCue(word),
                                        normalizer.s_Description,
                                        time,
                                        jsonVec[index++]);
    }

    // Put the bucket normalizer last so that incomplete restorations can be
    // detected by checking whether the bucket normalizer is restored
    CAnomalyScore::normalizerToJson(*this->bucketElement().s_Normalizer,
                                    key,
                                    bucketCue(),
                                    "root",
                                    time,
                                    jsonVec[index++]);

    json = core::CStringUtils::join(jsonVec, ",");
    if (makeArray)
    {
        json.insert(size_t(0), 1, '[');
        json += ']';
    }
}

CHierarchicalResultsNormalizer::ERestoreOutcome
    CHierarchicalResultsNormalizer::fromJsonStream(std::istream &inputStream)
{
    bool isBucketNormalizerRestored = false;

    this->TBase::clear();

    core::CJsonStateRestoreTraverser traverser(inputStream);

    do
    {
        // Call name() to prime the traverser if it hasn't started
        if (traverser.name() == EMPTY_STRING)
        {
            if (traverser.isEof())
            {
                LOG_DEBUG("No normalizer state to restore");
                // this is not an error
                return E_Ok;
            }
        }

        // The MLCUE_ATTRIBUTE should always be the first field
        if (traverser.name() != CAnomalyScore::MLCUE_ATTRIBUTE)
        {
            if (!traverser.next())
            {
                LOG_INFO("No normalizer state to restore");
                return E_Ok;
            }

            LOG_ERROR("Expected " << CAnomalyScore::MLCUE_ATTRIBUTE <<
                      " field in quantiles JSON got " << traverser.name() <<
                      " = " << traverser.value());
            return E_Corrupt;
        }

        const std::string cue(traverser.value());

        if (cue == BUCKET_CUE)
        {
            if (CAnomalyScore::normalizerFromJson(traverser,
                                                  *this->bucketElement().s_Normalizer) == false)
            {
                LOG_ERROR("Unable to restore bucket normalizer");
                return E_Corrupt;
            }

            isBucketNormalizerRestored = true;
        }
        else
        {
            TWordNormalizerPrVec *normalizerVec = 0;
            TDictionary::TUInt64Array hashArray;
            if (!this->parseCue(cue, normalizerVec, hashArray))
            {
                return E_Corrupt;
            }

            if (normalizerVec != 0)
            {
                if (!traverser.next())
                {
                    LOG_ERROR("Cannot restore hierarchical normalizer - end of object reached when " <<
                              CAnomalyScore::MLKEY_ATTRIBUTE << " was expected");
                    return E_Corrupt;
                }

                if (!traverser.next())
                {
                    LOG_ERROR("Cannot restore hierarchical normalizer - end of object reached when " <<
                              CAnomalyScore::MLQUANTILESDESCRIPTION_ATTRIBUTE << " was expected");
                    return E_Corrupt;
                }

                if (traverser.name() != CAnomalyScore::MLQUANTILESDESCRIPTION_ATTRIBUTE)
                {
                    LOG_ERROR("Cannot restore hierarchical normalizer - " <<
                              CAnomalyScore::MLQUANTILESDESCRIPTION_ATTRIBUTE <<
                              " element expected but found " << traverser.name() << '=' << traverser.value());
                    return E_Corrupt;
                }

                std::string quantileDesc(traverser.value());

                boost::shared_ptr<CAnomalyScore::CNormalizer> normalizer =
                        boost::make_shared<CAnomalyScore::CNormalizer>(m_ModelConfig);
                normalizerVec->emplace_back(TWord(hashArray), TNormalizer(quantileDesc, normalizer));
                if (CAnomalyScore::normalizerFromJson(traverser, *normalizer) == false)
                {
                    LOG_ERROR("Unable to restore normalizer with cue " << cue);
                    return E_Corrupt;
                }
            }
        }
    }
    while (traverser.nextObject());

    this->sort();

    LOG_DEBUG(this->influencerBucketSet().size() << " influencer bucket normalizers, " <<
              this->influencerSet().size() << " influencer normalizers, " <<
              this->partitionSet().size() << " partition normalizers, " <<
              this->personSet().size() << " person normalizers and " <<
              this->leafSet().size() << " leaf normalizers restored from JSON stream");

    return isBucketNormalizerRestored ? E_Ok : E_Incomplete;
}

const CAnomalyScore::CNormalizer &CHierarchicalResultsNormalizer::bucketNormalizer(void) const
{
    return *this->bucketElement().s_Normalizer;
}

const CAnomalyScore::CNormalizer *
    CHierarchicalResultsNormalizer::influencerBucketNormalizer(const std::string &influencerFieldName) const
{
    const TNormalizer *normalizer = this->influencerBucketElement(influencerFieldName);
    return normalizer ? normalizer->s_Normalizer.get() : 0;
}

const CAnomalyScore::CNormalizer *
    CHierarchicalResultsNormalizer::influencerNormalizer(const std::string &influencerFieldName) const
{
    const TNormalizer *normalizer = this->influencerElement(influencerFieldName);
    return normalizer ? normalizer->s_Normalizer.get() : 0;
}

const CAnomalyScore::CNormalizer *
    CHierarchicalResultsNormalizer::partitionNormalizer(const std::string &partitionFieldName) const
{
    const TNormalizer *normalizer = this->partitionElement(partitionFieldName);
    return normalizer ? normalizer->s_Normalizer.get() : 0;
}

const CAnomalyScore::CNormalizer *
    CHierarchicalResultsNormalizer::personNormalizer(const std::string &partitionFieldName,
                                                     const std::string &personFieldName) const
{
    const TNormalizer *normalizer = this->personElement(partitionFieldName, personFieldName);
    return normalizer ? normalizer->s_Normalizer.get() : 0;
}

const CAnomalyScore::CNormalizer *
    CHierarchicalResultsNormalizer::leafNormalizer(const std::string &partitionFieldName,
                                                   const std::string &personFieldName,
                                                   const std::string &functionName,
                                                   const std::string &valueFieldName) const
{
    const TNormalizer *normalizer = this->leafElement(partitionFieldName,
                                                      personFieldName,
                                                      functionName,
                                                      valueFieldName);
    return normalizer ? normalizer->s_Normalizer.get() : 0;
}

bool CHierarchicalResultsNormalizer::parseCue(const std::string &cue,
                                              TWordNormalizerPrVec *&normalizers,
                                              TDictionary::TUInt64Array &hashArray)
{
    normalizers = 0;
    std::size_t hashStartPos = 0;

    if (cue.compare(0, INFLUENCER_BUCKET_CUE_PREFIX.length(), INFLUENCER_BUCKET_CUE_PREFIX) == 0)
    {
        normalizers = &this->influencerBucketSet();
        hashStartPos = INFLUENCER_BUCKET_CUE_PREFIX.length();
    }
    else if (cue.compare(0, INFLUENCER_CUE_PREFIX.length(), INFLUENCER_CUE_PREFIX) == 0)
    {
        normalizers = &this->influencerSet();
        hashStartPos = INFLUENCER_CUE_PREFIX.length();
    }
    else if (cue.compare(0, PARTITION_CUE_PREFIX.length(), PARTITION_CUE_PREFIX) == 0)
    {
        normalizers = &this->partitionSet();
        hashStartPos = PARTITION_CUE_PREFIX.length();
    }
    else if (cue.compare(0, PERSON_CUE_PREFIX.length(), PERSON_CUE_PREFIX) == 0)
    {
        normalizers = &this->personSet();
        hashStartPos = PERSON_CUE_PREFIX.length();
    }
    else if (cue.compare(0, LEAF_CUE_PREFIX.length(), LEAF_CUE_PREFIX) == 0)
    {
        normalizers = &this->leafSet();
        hashStartPos = LEAF_CUE_PREFIX.length();
    }
    else
    {
        LOG_WARN("Did not understand normalizer cue " << cue);
        return true;
    }

    LOG_TRACE("cue = " << cue << ", hash = " << cue.substr(hashStartPos));

    if (core::CStringUtils::stringToType(cue.substr(hashStartPos), hashArray[0]) == false)
    {
        LOG_ERROR("Unable to parse normalizer hash from cue " << cue
                  << " starting at position " << hashStartPos);
        return false;
    }

    return true;
}

const std::string &CHierarchicalResultsNormalizer::bucketCue(void)
{
    return BUCKET_CUE;
}

std::string CHierarchicalResultsNormalizer::influencerBucketCue(const TWord &word)
{
    return INFLUENCER_BUCKET_CUE_PREFIX + core::CStringUtils::typeToString(word.hash64());
}

std::string CHierarchicalResultsNormalizer::influencerCue(const TWord &word)
{
    return INFLUENCER_CUE_PREFIX + core::CStringUtils::typeToString(word.hash64());
}

std::string CHierarchicalResultsNormalizer::partitionCue(const TWord &word)
{
    return PARTITION_CUE_PREFIX + core::CStringUtils::typeToString(word.hash64());
}

std::string CHierarchicalResultsNormalizer::personCue(const TWord &word)
{
    return PERSON_CUE_PREFIX + core::CStringUtils::typeToString(word.hash64());
}

std::string CHierarchicalResultsNormalizer::leafCue(const TWord &word)
{
    return LEAF_CUE_PREFIX + core::CStringUtils::typeToString(word.hash64());
}

}
}
