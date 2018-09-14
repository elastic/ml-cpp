/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CResultNormalizer_h
#define INCLUDED_ml_api_CResultNormalizer_h

#include <core/CLogger.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CHierarchicalResultsNormalizer.h>

#include <api/CDataProcessor.h>
#include <api/COutputHandler.h>
#include <api/ImportExport.h>

#include <boost/unordered_map.hpp>

#include <set>
#include <string>
#include <vector>

namespace ml {
namespace api {

//! \brief
//! Create normalized anomaly scores
//!
//! DESCRIPTION:\n
//! Calculate the normalized score for the different types
//! of results. The input is expected to contain the values
//! to be normalized and all the necessary information to
//! choose the correct normalizer for each result
//! (e.g. partitionFieldName, personFieldName, functionName, etc.).
//! The expected values are:
//!   1) raw anomaly score for root nodes
//!   2) probabilities for all others
//!
//! The state required to initialize the normalizers is a JSON document
//! as created by model::CHierarchicalResultsNormalizer::toJson().
//!
//! IMPLEMENTATION DECISIONS:\n
//! Does not support processor chaining functionality as it is unlikely
//! that this class would ever be chained to another data processor.
//!
class API_EXPORT CResultNormalizer {
public:
    //! Field names used in records to be normalised
    static const std::string LEVEL;
    static const std::string PARTITION_FIELD_NAME;
    static const std::string PARTITION_FIELD_VALUE;
    static const std::string PERSON_FIELD_NAME;
    static const std::string PERSON_FIELD_VALUE;
    static const std::string FUNCTION_NAME;
    static const std::string VALUE_FIELD_NAME;
    static const std::string PROBABILITY_NAME;
    static const std::string NORMALIZED_SCORE_NAME;

    //! Normalisation level values
    static const std::string ROOT_LEVEL;
    static const std::string PARTITION_LEVEL;
    static const std::string LEAF_LEVEL;
    static const std::string BUCKET_INFLUENCER_LEVEL;
    static const std::string INFLUENCER_LEVEL;

    static const std::string ZERO;

public:
    using TStrVec = std::vector<std::string>;
    using TStrVecItr = TStrVec::iterator;
    using TStrVecCItr = TStrVec::const_iterator;

    using TStrStrUMap = boost::unordered_map<std::string, std::string>;
    using TStrStrUMapItr = TStrStrUMap::iterator;
    using TStrStrUMapCItr = TStrStrUMap::const_iterator;

public:
    CResultNormalizer(const model::CAnomalyDetectorModelConfig& modelConfig,
                      COutputHandler& outputHandler);

    //! Initialise the system change normalizer
    bool initNormalizer(const std::string& stateFileName);

    //! Handle a record to be normalized
    bool handleRecord(const TStrStrUMap& dataRowFields);

private:
    bool parseDataFields(const TStrStrUMap& dataRowFields,
                         std::string& level,
                         std::string& partition,
                         std::string& partitionValue,
                         std::string& person,
                         std::string& personValue,
                         std::string& function,
                         std::string& valueFieldName,
                         double& probability);

    template<typename T>
    bool parseDataField(const TStrStrUMap& dataRowFields,
                        const std::string& fieldName,
                        T& result) const {
        TStrStrUMapCItr iter = dataRowFields.find(fieldName);
        if (iter == dataRowFields.end() ||
            core::CStringUtils::stringToType(iter->second, result) == false) {
            LOG_ERROR(<< "Cannot interpret " << fieldName << " field in record:\n"
                      << CDataProcessor::debugPrintRecord(dataRowFields));
            return false;
        }
        return true;
    }

private:
    //! Reference to model config
    const model::CAnomalyDetectorModelConfig& m_ModelConfig;

    //! Object to which the output is passed
    COutputHandler& m_OutputHandler;

    //! Do we need to tell the output handler what our fieldnames are?
    bool m_WriteFieldNames;

    //! Map holding fields to write to the output
    TStrStrUMap m_OutputFields;

    //! References to specific entries in the map to save repeatedly
    //! searching for them
    std::string& m_OutputFieldNormalizedScore;

    //! The hierarchical results normalizer
    model::CHierarchicalResultsNormalizer m_Normalizer;
};
}
}

#endif // INCLUDED_ml_api_CResultNormalizer_h
