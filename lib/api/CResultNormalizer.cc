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
#include <api/CResultNormalizer.h>

#include <core/CStringUtils.h>

#include <maths/CTools.h>

#include <fstream>

namespace ml {
namespace api {

// Initialise statics
const std::string CResultNormalizer::LEVEL("level");
const std::string CResultNormalizer::PARTITION_FIELD_NAME("partition_field_name");
const std::string CResultNormalizer::PARTITION_FIELD_VALUE("partition_field_value");
const std::string CResultNormalizer::PERSON_FIELD_NAME("person_field_name");
const std::string CResultNormalizer::FUNCTION_NAME("function_name");
const std::string CResultNormalizer::VALUE_FIELD_NAME("value_field_name");
const std::string CResultNormalizer::PROBABILITY_NAME("probability");
const std::string CResultNormalizer::NORMALIZED_SCORE_NAME("normalized_score");
const std::string CResultNormalizer::ROOT_LEVEL("root");
const std::string CResultNormalizer::LEAF_LEVEL("leaf");
const std::string CResultNormalizer::PARTITION_LEVEL("part");
const std::string CResultNormalizer::BUCKET_INFLUENCER_LEVEL("inflb");
const std::string CResultNormalizer::INFLUENCER_LEVEL("infl");
const std::string CResultNormalizer::ZERO("0");

CResultNormalizer::CResultNormalizer(const model::CAnomalyDetectorModelConfig& modelConfig, COutputHandler& outputHandler)
    : m_ModelConfig(modelConfig),
      m_OutputHandler(outputHandler),
      m_WriteFieldNames(true),
      m_OutputFieldNormalizedScore(m_OutputFields[NORMALIZED_SCORE_NAME]),
      m_Normalizer(m_ModelConfig) {
}

bool CResultNormalizer::initNormalizer(const std::string& stateFileName) {
    std::ifstream inputStream(stateFileName.c_str());
    model::CHierarchicalResultsNormalizer::ERestoreOutcome outcome(m_Normalizer.fromJsonStream(inputStream));
    if (outcome != model::CHierarchicalResultsNormalizer::E_Ok) {
        LOG_ERROR("Failed to restore JSON state for quantiles");
        return false;
    }
    return true;
}

bool CResultNormalizer::handleRecord(const TStrStrUMap& dataRowFields) {
    if (m_WriteFieldNames) {
        TStrVec fieldNames;
        fieldNames.reserve(dataRowFields.size());
        for (const auto& entry : dataRowFields) {
            fieldNames.push_back(entry.first);
        }

        TStrVec extraFieldNames;
        extraFieldNames.push_back(NORMALIZED_SCORE_NAME);

        if (m_OutputHandler.fieldNames(fieldNames, extraFieldNames) == false) {
            LOG_ERROR("Unable to set field names for output");
            return false;
        }
        m_WriteFieldNames = false;
    }

    std::string level;
    std::string partition;
    std::string partitionValue;
    std::string person;
    std::string function;
    std::string valueFieldName;
    double probability(0.0);

    bool isValidRecord(false);
    if (m_ModelConfig.perPartitionNormalization()) {
        isValidRecord = parseDataFields(dataRowFields, level, partition, partitionValue, person, function, valueFieldName, probability);
    } else {
        isValidRecord = parseDataFields(dataRowFields, level, partition, person, function, valueFieldName, probability);
    }

    std::string partitionKey = m_ModelConfig.perPartitionNormalization() ? partition + partitionValue : partition;

    if (isValidRecord) {
        const model::CAnomalyScore::CNormalizer* levelNormalizer = nullptr;
        double score = probability > m_ModelConfig.maximumAnomalousProbability() ? 0.0 : maths::CTools::deviation(probability);
        if (level == ROOT_LEVEL) {
            levelNormalizer = &m_Normalizer.bucketNormalizer();
        } else if (level == LEAF_LEVEL) {
            levelNormalizer = m_Normalizer.leafNormalizer(partitionKey, person, function, valueFieldName);
        } else if (level == PARTITION_LEVEL) {
            levelNormalizer = m_Normalizer.partitionNormalizer(partitionKey);
        } else if (level == BUCKET_INFLUENCER_LEVEL) {
            levelNormalizer = m_Normalizer.influencerBucketNormalizer(person);
        } else if (level == INFLUENCER_LEVEL) {
            levelNormalizer = m_Normalizer.influencerNormalizer(person);
        } else {
            LOG_ERROR("Unexpected   : " << level);
        }
        if (levelNormalizer != nullptr) {
            if (levelNormalizer->canNormalize() && levelNormalizer->normalize(score) == false) {
                LOG_ERROR("Failed to normalize score " << score << " at level " << level << " with partition field name " << partition
                                                       << " and person field name " << person);
            }
        } else {
            LOG_ERROR("No normalizer available"
                      " at level '"
                      << level << "' with partition field name '" << partition << "' and person field name '" << person << "'");
        }

        m_OutputFieldNormalizedScore = (score > 0.0) ? core::CStringUtils::typeToStringPretty(score) : ZERO;
    } else {
        m_OutputFieldNormalizedScore.clear();
    }

    if (m_OutputHandler.writeRow(dataRowFields, m_OutputFields) == false) {
        LOG_ERROR("Unable to write normalized output");
        return false;
    }

    return true;
}

bool CResultNormalizer::parseDataFields(const TStrStrUMap& dataRowFields,
                                        std::string& level,
                                        std::string& partition,
                                        std::string& person,
                                        std::string& function,
                                        std::string& valueFieldName,
                                        double& probability) {
    return this->parseDataField(dataRowFields, LEVEL, level) && this->parseDataField(dataRowFields, PARTITION_FIELD_NAME, partition) &&
           this->parseDataField(dataRowFields, PERSON_FIELD_NAME, person) && this->parseDataField(dataRowFields, FUNCTION_NAME, function) &&
           this->parseDataField(dataRowFields, VALUE_FIELD_NAME, valueFieldName) &&
           this->parseDataField(dataRowFields, PROBABILITY_NAME, probability);
}

bool CResultNormalizer::parseDataFields(const TStrStrUMap& dataRowFields,
                                        std::string& level,
                                        std::string& partition,
                                        std::string& partitionValue,
                                        std::string& person,
                                        std::string& function,
                                        std::string& valueFieldName,
                                        double& probability) {
    return this->parseDataField(dataRowFields, LEVEL, level) && this->parseDataField(dataRowFields, PARTITION_FIELD_NAME, partition) &&
           this->parseDataField(dataRowFields, PARTITION_FIELD_VALUE, partitionValue) &&
           this->parseDataField(dataRowFields, PERSON_FIELD_NAME, person) && this->parseDataField(dataRowFields, FUNCTION_NAME, function) &&
           this->parseDataField(dataRowFields, VALUE_FIELD_NAME, valueFieldName) &&
           this->parseDataField(dataRowFields, PROBABILITY_NAME, probability);
}
}
}
