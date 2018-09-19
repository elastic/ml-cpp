/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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
const std::string CResultNormalizer::PERSON_FIELD_VALUE("person_field_value");
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

CResultNormalizer::CResultNormalizer(const model::CAnomalyDetectorModelConfig& modelConfig,
                                     COutputHandler& outputHandler)
    : m_ModelConfig(modelConfig), m_OutputHandler(outputHandler),
      m_WriteFieldNames(true),
      m_OutputFieldNormalizedScore(m_OutputFields[NORMALIZED_SCORE_NAME]),
      m_Normalizer(m_ModelConfig) {
}

bool CResultNormalizer::initNormalizer(const std::string& stateFileName) {
    std::ifstream inputStream(stateFileName.c_str());
    model::CHierarchicalResultsNormalizer::ERestoreOutcome outcome(
        m_Normalizer.fromJsonStream(inputStream));
    if (outcome != model::CHierarchicalResultsNormalizer::E_Ok) {
        LOG_ERROR(<< "Failed to restore JSON state for quantiles");
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
            LOG_ERROR(<< "Unable to set field names for output");
            return false;
        }
        m_WriteFieldNames = false;
    }

    std::string level;
    std::string partitionName;
    std::string partitionValue;
    std::string personName;
    std::string personValue;
    std::string function;
    std::string valueFieldName;
    double probability(0.0);

    // As of version 6.5 the 'personValue' field is required for (re)normalization to succeed.
    // In the case of renormalization the 'personValue' field must be included in the set of
    // parameters sent from the Java side ML plugin to Elasticsearch.
    // In production the version of the native code application and the java application it is communicating directly
    // with will always match so supporting BWC with versions prior to 6.5 is not necessary.
    bool isNormalizable = this->parseDataFields(
        dataRowFields, level, partitionName, partitionValue, personName,
        personValue, function, valueFieldName, probability);

    LOG_TRACE(<< "level='" << level << "', partitionName='" << partitionName << "', partitionValue='"
              << partitionValue << "', personName='" << personName << "', personValue='"
              << personValue << "', function='" << function << "', valueFieldName='"
              << valueFieldName << "', probability='" << probability << "'");

    if (isNormalizable) {
        const model::CAnomalyScore::CNormalizer* levelNormalizer = nullptr;
        double score = probability > m_ModelConfig.maximumAnomalousProbability()
                           ? 0.0
                           : maths::CTools::anomalyScore(probability);
        if (level == ROOT_LEVEL) {
            levelNormalizer = &m_Normalizer.bucketNormalizer();
        } else if (level == LEAF_LEVEL) {
            levelNormalizer = m_Normalizer.leafNormalizer(
                partitionName, personName, function, valueFieldName);
        } else if (level == PARTITION_LEVEL) {
            levelNormalizer = m_Normalizer.partitionNormalizer(partitionName);
        } else if (level == BUCKET_INFLUENCER_LEVEL) {
            levelNormalizer = m_Normalizer.influencerBucketNormalizer(personName);
        } else if (level == INFLUENCER_LEVEL) {
            levelNormalizer = m_Normalizer.influencerNormalizer(personName);
        } else {
            LOG_ERROR(<< "Unexpected   : " << level);
        }
        if (levelNormalizer != nullptr) {
            if (levelNormalizer->canNormalize()) {
                model::CAnomalyScore::CNormalizer::CMaximumScoreScope scope{
                    partitionName, partitionValue, personName, personValue};
                if (levelNormalizer->normalize(scope, score) == false) {
                    LOG_ERROR(<< "Failed to normalize score " << score << " at level \""
                              << level << "\" using scope " << scope.print());
                }
            }
        } else {
            LOG_ERROR(<< "No normalizer available at level '" << level
                      << "' with partition field name '" << partitionName
                      << "' and person field name '" << personName << "'");
        }

        m_OutputFieldNormalizedScore =
            (score > 0.0) ? core::CStringUtils::typeToStringPretty(score) : ZERO;
    } else {
        m_OutputFieldNormalizedScore.clear();
    }

    if (m_OutputHandler.writeRow(dataRowFields, m_OutputFields) == false) {
        LOG_ERROR(<< "Unable to write normalized output");
        return false;
    }

    return true;
}

bool CResultNormalizer::parseDataFields(const TStrStrUMap& dataRowFields,
                                        std::string& level,
                                        std::string& partitionName,
                                        std::string& partitionValue,
                                        std::string& personName,
                                        std::string& personValue,
                                        std::string& function,
                                        std::string& valueFieldName,
                                        double& probability) {
    return this->parseDataField(dataRowFields, LEVEL, level) &&
           this->parseDataField(dataRowFields, PARTITION_FIELD_NAME, partitionName) &&
           this->parseDataField(dataRowFields, PARTITION_FIELD_VALUE, partitionValue) &&
           this->parseDataField(dataRowFields, PERSON_FIELD_NAME, personName) &&
           this->parseDataField(dataRowFields, PERSON_FIELD_VALUE, personValue) &&
           this->parseDataField(dataRowFields, FUNCTION_NAME, function) &&
           this->parseDataField(dataRowFields, VALUE_FIELD_NAME, valueFieldName) &&
           this->parseDataField(dataRowFields, PROBABILITY_NAME, probability);
}
}
}
