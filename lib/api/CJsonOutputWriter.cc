/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CJsonOutputWriter.h>

#include <core/CScopedRapidJsonPoolAllocator.h>
#include <core/CStringUtils.h>
#include <core/CTimeUtils.h>

#include <model/CHierarchicalResultsNormalizer.h>
#include <model/ModelTypes.h>

#include <api/CModelSizeStatsJsonWriter.h>
#include <api/CModelSnapshotJsonWriter.h>

#include <algorithm>
#include <ostream>

namespace ml {
namespace api {

namespace {

// JSON field names
const std::string JOB_ID("job_id");
const std::string TIMESTAMP("timestamp");
const std::string BUCKET("bucket");
const std::string DETECTOR_INDEX("detector_index");
const std::string RECORDS("records");
const std::string EVENT_COUNT("event_count");
const std::string IS_INTERIM("is_interim");
const std::string PROBABILITY("probability");
const std::string RAW_ANOMALY_SCORE("raw_anomaly_score");
const std::string ANOMALY_SCORE("anomaly_score");
const std::string RECORD_SCORE("record_score");
const std::string INITIAL_RECORD_SCORE("initial_record_score");
const std::string INFLUENCER_SCORE("influencer_score");
const std::string INITIAL_INFLUENCER_SCORE("initial_influencer_score");
const std::string FIELD_NAME("field_name");
const std::string BY_FIELD_NAME("by_field_name");
const std::string BY_FIELD_VALUE("by_field_value");
const std::string CORRELATED_BY_FIELD_VALUE("correlated_by_field_value");
const std::string TYPICAL("typical");
const std::string ACTUAL("actual");
const std::string CAUSES("causes");
const std::string FUNCTION("function");
const std::string FUNCTION_DESCRIPTION("function_description");
const std::string OVER_FIELD_NAME("over_field_name");
const std::string OVER_FIELD_VALUE("over_field_value");
const std::string PARTITION_FIELD_NAME("partition_field_name");
const std::string PARTITION_FIELD_VALUE("partition_field_value");
const std::string INITIAL_SCORE("initial_anomaly_score");
const std::string INFLUENCER_FIELD_NAME("influencer_field_name");
const std::string INFLUENCER_FIELD_VALUE("influencer_field_value");
const std::string INFLUENCER_FIELD_VALUES("influencer_field_values");
const std::string BUCKET_INFLUENCERS("bucket_influencers");
const std::string INFLUENCERS("influencers");
const std::string FLUSH("flush");
const std::string ID("id");
const std::string LAST_FINALIZED_BUCKET_END("last_finalized_bucket_end");
const std::string CATEGORY_ID("category_id");
const std::string CATEGORY_DEFINITION("category_definition");
const std::string TERMS("terms");
const std::string REGEX("regex");
const std::string MAX_MATCHING_LENGTH("max_matching_length");
const std::string EXAMPLES("examples");
const std::string BUCKET_SPAN("bucket_span");
const std::string PROCESSING_TIME("processing_time_ms");
const std::string TIME_INFLUENCER("bucket_time");
const std::string SCHEDULED_EVENTS("scheduled_events");
const std::string QUANTILES("quantiles");

//! Get a numeric field from a JSON document.
//! Assumes the document contains the field.
//! The caller is responsible for ensuring this, and a
//! program crash is likely if this requirement is not met.
double doubleFromDocument(const CJsonOutputWriter::TDocumentWeakPtr& weakDoc,
                          const std::string& field) {
    CJsonOutputWriter::TDocumentPtr docPtr = weakDoc.lock();
    if (!docPtr) {
        LOG_ERROR(<< "Inconsistent program state. JSON document unavailable.");
        return 0.0;
    }
    return (*docPtr)[field].GetDouble();
}

//! Sort rapidjson documents by the probability lowest to highest
class CProbabilityLess {
public:
    bool operator()(const CJsonOutputWriter::TDocumentWeakPtrIntPr& lhs,
                    const CJsonOutputWriter::TDocumentWeakPtrIntPr& rhs) const {
        return doubleFromDocument(lhs.first, PROBABILITY) <
               doubleFromDocument(rhs.first, PROBABILITY);
    }
};

const CProbabilityLess PROBABILITY_LESS = CProbabilityLess();

//! Sort rapidjson documents by detector name first then probability lowest to highest
class CDetectorThenProbabilityLess {
public:
    bool operator()(const CJsonOutputWriter::TDocumentWeakPtrIntPr& lhs,
                    const CJsonOutputWriter::TDocumentWeakPtrIntPr& rhs) const {
        if (lhs.second == rhs.second) {
            return doubleFromDocument(lhs.first, PROBABILITY) <
                   doubleFromDocument(rhs.first, PROBABILITY);
        }
        return lhs.second < rhs.second;
    }
};

const CDetectorThenProbabilityLess DETECTOR_PROBABILITY_LESS = CDetectorThenProbabilityLess();

//! Sort influences from highes to lowest
class CInfluencesLess {
public:
    bool operator()(const std::pair<const char*, double>& lhs,
                    const std::pair<const char*, double>& rhs) const {
        return lhs.second > rhs.second;
    }
};

const CInfluencesLess INFLUENCE_LESS = CInfluencesLess();

//! Sort influencer from highest to lowest by score
class CInfluencerGreater {
public:
    CInfluencerGreater(const std::string& field) : m_Field(field) {}

    bool operator()(const CJsonOutputWriter::TDocumentWeakPtr& lhs,
                    const CJsonOutputWriter::TDocumentWeakPtr& rhs) const {
        return doubleFromDocument(lhs, m_Field) > doubleFromDocument(rhs, m_Field);
    }

private:
    const std::string& m_Field;
};

const CInfluencerGreater INFLUENCER_GREATER = CInfluencerGreater(INITIAL_INFLUENCER_SCORE);
const CInfluencerGreater BUCKET_INFLUENCER_GREATER = CInfluencerGreater(INITIAL_SCORE);
}

CJsonOutputWriter::CJsonOutputWriter(const std::string& jobId,
                                     core::CJsonOutputStreamWrapper& strmOut)
    : m_JobId(jobId), m_Writer(strmOut), m_LastNonInterimBucketTime(0),
      m_Finalised(false), m_RecordOutputLimit(0) {
    // Don't write any output in the constructor because, the way things work at
    // the moment, the output stream might be redirected after construction
}

CJsonOutputWriter::~CJsonOutputWriter() {
    finalise();
}

void CJsonOutputWriter::finalise() {
    if (m_Finalised) {
        return;
    }

    // Flush the output This ensures that any buffers are written out
    // Note: This is still asynchronous.
    m_Writer.flush();
    m_Finalised = true;
}

bool CJsonOutputWriter::acceptResult(const CHierarchicalResultsWriter::TResults& results) {
    SBucketData& bucketData = m_BucketDataByTime[results.s_BucketStartTime];

    if (results.s_ResultType == CHierarchicalResultsWriter::E_SimpleCountResult) {
        if (!results.s_CurrentRate) {
            LOG_ERROR(<< "Simple count detector has no current rate");
            return false;
        }

        bucketData.s_InputEventCount = *results.s_CurrentRate;
        bucketData.s_BucketSpan = results.s_BucketSpan;
        bucketData.s_ScheduledEventDescriptions = results.s_ScheduledEventDescriptions;
        return true;
    }

    TDocumentWeakPtr newDoc;
    if (!results.s_IsOverallResult) {
        newDoc = m_Writer.makeStorableDoc();
        this->addPopulationCauseFields(results, newDoc);
        m_NestedDocs.push_back(newDoc);

        return true;
    }

    ++bucketData.s_RecordCount;

    TDocumentWeakPtrIntPrVec& detectorDocumentsToWrite = bucketData.s_DocumentsToWrite;

    bool makeHeap(false);
    // If a max number of records to output has not been set or we haven't
    // reached that limit yet just append the new document to the array
    if (m_RecordOutputLimit == 0 || bucketData.s_RecordCount <= m_RecordOutputLimit) {
        newDoc = m_Writer.makeStorableDoc();
        detectorDocumentsToWrite.push_back(TDocumentWeakPtrIntPr(newDoc, results.s_Identifier));

        // the document array is now full, make a max heap
        makeHeap = bucketData.s_RecordCount == m_RecordOutputLimit;
    } else {
        // Have reached the limit of records to write so compare the new doc
        // to the highest probability anomaly doc and replace if more anomalous
        if (results.s_Probability >= bucketData.s_HighestProbability) {
            // Discard any associated nested docs
            m_NestedDocs.clear();
            return true;
        }

        newDoc = m_Writer.makeStorableDoc();
        // remove the highest prob doc and insert new one
        std::pop_heap(detectorDocumentsToWrite.begin(),
                      detectorDocumentsToWrite.end(), PROBABILITY_LESS);
        detectorDocumentsToWrite.pop_back();

        detectorDocumentsToWrite.push_back(TDocumentWeakPtrIntPr(newDoc, results.s_Identifier));

        makeHeap = true;
    }

    // The check for population results must come first because some population
    // results are also metrics
    if (results.s_ResultType == CHierarchicalResultsWriter::E_PopulationResult) {
        this->addPopulationFields(results, newDoc);
    } else if (results.s_IsMetric) {
        this->addMetricFields(results, newDoc);
    } else {
        this->addEventRateFields(results, newDoc);
    }

    this->addInfluences(results.s_Influences, newDoc);

    if (makeHeap) {
        std::make_heap(detectorDocumentsToWrite.begin(),
                       detectorDocumentsToWrite.end(), PROBABILITY_LESS);

        bucketData.s_HighestProbability =
            doubleFromDocument(detectorDocumentsToWrite.front().first, PROBABILITY);
        makeHeap = false;
    }

    return true;
}

bool CJsonOutputWriter::acceptInfluencer(core_t::TTime time,
                                         const model::CHierarchicalResults::TNode& node,
                                         bool isBucketInfluencer) {
    TDocumentWeakPtr newDoc = m_Writer.makeStorableDoc();
    SBucketData& bucketData = m_BucketDataByTime[time];
    TDocumentWeakPtrVec& documents = (isBucketInfluencer)
                                         ? bucketData.s_BucketInfluencerDocuments
                                         : bucketData.s_InfluencerDocuments;

    bool isLimitedWrite(m_RecordOutputLimit > 0);

    if (isLimitedWrite && documents.size() == m_RecordOutputLimit) {
        double& lowestScore = (isBucketInfluencer)
                                  ? bucketData.s_LowestBucketInfluencerScore
                                  : bucketData.s_LowestInfluencerScore;

        if (node.s_NormalizedAnomalyScore < lowestScore) {
            //  Don't write this influencer
            return true;
        }

        // need to remove the lowest score record
        documents.pop_back();
    }

    this->addInfluencerFields(isBucketInfluencer, node, newDoc);
    documents.push_back(newDoc);

    bool sortVectorAfterWritingDoc = isLimitedWrite && documents.size() >= m_RecordOutputLimit;

    if (sortVectorAfterWritingDoc) {
        std::sort(documents.begin(), documents.end(),
                  isBucketInfluencer ? BUCKET_INFLUENCER_GREATER : INFLUENCER_GREATER);
    }

    if (isBucketInfluencer) {
        bucketData.s_MaxBucketInfluencerNormalizedAnomalyScore =
            std::max(bucketData.s_MaxBucketInfluencerNormalizedAnomalyScore,
                     node.s_NormalizedAnomalyScore);

        bucketData.s_LowestBucketInfluencerScore =
            std::min(bucketData.s_LowestBucketInfluencerScore,
                     doubleFromDocument(documents.back(), INITIAL_SCORE));
    } else {
        bucketData.s_LowestInfluencerScore =
            std::min(bucketData.s_LowestInfluencerScore,
                     doubleFromDocument(documents.back(), INITIAL_INFLUENCER_SCORE));
    }

    return true;
}

void CJsonOutputWriter::acceptBucketTimeInfluencer(core_t::TTime time,
                                                   double probability,
                                                   double rawAnomalyScore,
                                                   double normalizedAnomalyScore) {
    SBucketData& bucketData = m_BucketDataByTime[time];
    if (bucketData.s_RecordCount == 0) {
        return;
    }

    TDocumentWeakPtr doc = m_Writer.makeStorableDoc();
    TDocumentPtr newDoc = doc.lock();
    if (!newDoc) {
        LOG_ERROR(<< "Failed to create new JSON document");
        return;
    }
    m_Writer.addStringFieldCopyToObj(INFLUENCER_FIELD_NAME, TIME_INFLUENCER, *newDoc);
    m_Writer.addDoubleFieldToObj(PROBABILITY, probability, *newDoc);
    m_Writer.addDoubleFieldToObj(RAW_ANOMALY_SCORE, rawAnomalyScore, *newDoc);
    m_Writer.addDoubleFieldToObj(INITIAL_SCORE, normalizedAnomalyScore, *newDoc);
    m_Writer.addDoubleFieldToObj(ANOMALY_SCORE, normalizedAnomalyScore, *newDoc);

    bucketData.s_MaxBucketInfluencerNormalizedAnomalyScore = std::max(
        bucketData.s_MaxBucketInfluencerNormalizedAnomalyScore, normalizedAnomalyScore);
    bucketData.s_BucketInfluencerDocuments.push_back(doc);
}

bool CJsonOutputWriter::endOutputBatch(bool isInterim, uint64_t bucketProcessingTime) {
    for (TTimeBucketDataMapItr iter = m_BucketDataByTime.begin();
         iter != m_BucketDataByTime.end(); ++iter) {
        this->writeBucket(isInterim, iter->first, iter->second, bucketProcessingTime);
        if (!isInterim) {
            m_LastNonInterimBucketTime = iter->first;
        }
    }

    // After writing the buckets clear all the bucket data so that we don't
    // accumulate memory.
    m_BucketDataByTime.clear();
    m_NestedDocs.clear();

    return true;
}

bool CJsonOutputWriter::fieldNames(const TStrVec& /*fieldNames*/,
                                   const TStrVec& /*extraFieldNames*/) {
    return true;
}

bool CJsonOutputWriter::writeRow(const TStrStrUMap& dataRowFields,
                                 const TStrStrUMap& overrideDataRowFields) {
    using TScopedAllocator =
        core::CScopedRapidJsonPoolAllocator<core::CRapidJsonConcurrentLineWriter>;

    TScopedAllocator scopedAllocator("CJsonOutputWriter::writeRow", m_Writer);

    rapidjson::Document doc = m_Writer.makeDoc();

    // Write all the fields to the document as strings
    // No need to copy the strings as the doc is written straight away
    for (TStrStrUMapCItr fieldValueIter = dataRowFields.begin();
         fieldValueIter != dataRowFields.end(); ++fieldValueIter) {
        const std::string& name = fieldValueIter->first;
        const std::string& value = fieldValueIter->second;

        // Only output fields that aren't overridden
        if (overrideDataRowFields.find(name) == overrideDataRowFields.end()) {
            m_Writer.addMemberRef(name, value, doc);
        }
    }

    for (TStrStrUMapCItr fieldValueIter = overrideDataRowFields.begin();
         fieldValueIter != overrideDataRowFields.end(); ++fieldValueIter) {
        const std::string& name = fieldValueIter->first;
        const std::string& value = fieldValueIter->second;

        m_Writer.addMemberRef(name, value, doc);
    }

    m_Writer.write(doc);

    return true;
}

void CJsonOutputWriter::writeBucket(bool isInterim,
                                    core_t::TTime bucketTime,
                                    SBucketData& bucketData,
                                    uint64_t bucketProcessingTime) {
    // Write records
    if (!bucketData.s_DocumentsToWrite.empty()) {
        // Sort the results so they are grouped by detector and
        // ordered by probability
        std::sort(bucketData.s_DocumentsToWrite.begin(),
                  bucketData.s_DocumentsToWrite.end(), DETECTOR_PROBABILITY_LESS);

        m_Writer.StartObject();
        m_Writer.String(RECORDS);
        m_Writer.StartArray();

        // Iterate over the different detectors that we have results for
        for (TDocumentWeakPtrIntPrVecItr detectorIter =
                 bucketData.s_DocumentsToWrite.begin();
             detectorIter != bucketData.s_DocumentsToWrite.end(); ++detectorIter) {
            // Write the document, adding some extra fields as we go
            int detectorIndex = detectorIter->second;
            TDocumentWeakPtr weakDoc = detectorIter->first;
            TDocumentPtr docPtr = weakDoc.lock();
            if (!docPtr) {
                LOG_ERROR(<< "Inconsistent program state. JSON document unavailable.");
                continue;
            }

            m_Writer.addIntFieldToObj(DETECTOR_INDEX, detectorIndex, *docPtr);
            m_Writer.addIntFieldToObj(BUCKET_SPAN, bucketData.s_BucketSpan, *docPtr);
            m_Writer.addStringFieldCopyToObj(JOB_ID, m_JobId, *docPtr);
            m_Writer.addTimeFieldToObj(TIMESTAMP, bucketTime, *docPtr);

            if (isInterim) {
                m_Writer.addBoolFieldToObj(IS_INTERIM, isInterim, *docPtr);
            }
            m_Writer.write(*docPtr);
        }
        m_Writer.EndArray();
        m_Writer.EndObject();
    }

    // Write influencers
    if (!bucketData.s_InfluencerDocuments.empty()) {
        m_Writer.StartObject();
        m_Writer.String(INFLUENCERS);
        m_Writer.StartArray();
        for (TDocumentWeakPtrVecItr influencerIter =
                 bucketData.s_InfluencerDocuments.begin();
             influencerIter != bucketData.s_InfluencerDocuments.end(); ++influencerIter) {
            TDocumentWeakPtr weakDoc = *influencerIter;
            TDocumentPtr docPtr = weakDoc.lock();
            if (!docPtr) {
                LOG_ERROR(<< "Inconsistent program state. JSON document unavailable.");
                continue;
            }

            m_Writer.addStringFieldCopyToObj(JOB_ID, m_JobId, *docPtr);
            m_Writer.addTimeFieldToObj(TIMESTAMP, bucketTime, *docPtr);
            if (isInterim) {
                m_Writer.addBoolFieldToObj(IS_INTERIM, isInterim, *docPtr);
            }
            m_Writer.addIntFieldToObj(BUCKET_SPAN, bucketData.s_BucketSpan, *docPtr);
            m_Writer.write(*docPtr);
        }
        m_Writer.EndArray();
        m_Writer.EndObject();
    }

    // Write bucket at the end, as some of its values need to iterate over records, etc.
    m_Writer.StartObject();
    m_Writer.String(BUCKET);

    m_Writer.StartObject();
    m_Writer.String(JOB_ID);
    m_Writer.String(m_JobId);
    m_Writer.String(TIMESTAMP);
    m_Writer.Time(bucketTime);

    m_Writer.String(ANOMALY_SCORE);
    m_Writer.Double(bucketData.s_MaxBucketInfluencerNormalizedAnomalyScore);
    m_Writer.String(INITIAL_SCORE);
    m_Writer.Double(bucketData.s_MaxBucketInfluencerNormalizedAnomalyScore);
    m_Writer.String(EVENT_COUNT);
    m_Writer.Uint64(bucketData.s_InputEventCount);
    if (isInterim) {
        m_Writer.String(IS_INTERIM);
        m_Writer.Bool(isInterim);
    }
    m_Writer.String(BUCKET_SPAN);
    m_Writer.Int64(bucketData.s_BucketSpan);

    if (!bucketData.s_BucketInfluencerDocuments.empty()) {
        // Write the array of influencers
        m_Writer.String(BUCKET_INFLUENCERS);
        m_Writer.StartArray();
        for (TDocumentWeakPtrVecItr influencerIter =
                 bucketData.s_BucketInfluencerDocuments.begin();
             influencerIter != bucketData.s_BucketInfluencerDocuments.end();
             ++influencerIter) {
            TDocumentWeakPtr weakDoc = *influencerIter;
            TDocumentPtr docPtr = weakDoc.lock();
            if (!docPtr) {
                LOG_ERROR(<< "Inconsistent program state. JSON document unavailable.");
                continue;
            }

            m_Writer.addStringFieldCopyToObj(JOB_ID, m_JobId, *docPtr);
            m_Writer.addTimeFieldToObj(TIMESTAMP, bucketTime, *docPtr);
            m_Writer.addIntFieldToObj(BUCKET_SPAN, bucketData.s_BucketSpan, *docPtr);
            if (isInterim) {
                m_Writer.addBoolFieldToObj(IS_INTERIM, isInterim, *docPtr);
            }
            m_Writer.write(*docPtr);
        }
        m_Writer.EndArray();
    }

    m_Writer.String(PROCESSING_TIME);
    m_Writer.Uint64(bucketProcessingTime);

    if (bucketData.s_ScheduledEventDescriptions.empty() == false) {
        m_Writer.String(SCHEDULED_EVENTS);
        m_Writer.StartArray();
        for (const auto& it : bucketData.s_ScheduledEventDescriptions) {
            m_Writer.String(it);
        }
        m_Writer.EndArray();
    }

    m_Writer.EndObject();
    m_Writer.EndObject();
}

void CJsonOutputWriter::addMetricFields(const CHierarchicalResultsWriter::TResults& results,
                                        TDocumentWeakPtr weakDoc) {
    TDocumentPtr docPtr = weakDoc.lock();
    if (!docPtr) {
        LOG_ERROR(<< "Inconsistent program state. JSON document unavailable.");
        return;
    }

    // record_score, probability, fieldName, byFieldName, byFieldValue, partitionFieldName,
    // partitionFieldValue, function, typical, actual. influences?
    m_Writer.addDoubleFieldToObj(INITIAL_RECORD_SCORE,
                                 results.s_NormalizedAnomalyScore, *docPtr);
    m_Writer.addDoubleFieldToObj(RECORD_SCORE, results.s_NormalizedAnomalyScore, *docPtr);
    m_Writer.addDoubleFieldToObj(PROBABILITY, results.s_Probability, *docPtr);
    m_Writer.addStringFieldCopyToObj(FIELD_NAME, results.s_MetricValueField, *docPtr);
    if (!results.s_ByFieldName.empty()) {
        m_Writer.addStringFieldCopyToObj(BY_FIELD_NAME, results.s_ByFieldName, *docPtr);
        // If name is present then force output of value too, even when empty
        m_Writer.addStringFieldCopyToObj(BY_FIELD_VALUE, results.s_ByFieldValue,
                                         *docPtr, true);
        // But allow correlatedByFieldValue to be unset if blank
        m_Writer.addStringFieldCopyToObj(CORRELATED_BY_FIELD_VALUE,
                                         results.s_CorrelatedByFieldValue, *docPtr);
    }
    if (!results.s_PartitionFieldName.empty()) {
        m_Writer.addStringFieldCopyToObj(PARTITION_FIELD_NAME,
                                         results.s_PartitionFieldName, *docPtr);
        // If name is present then force output of value too, even when empty
        m_Writer.addStringFieldCopyToObj(
            PARTITION_FIELD_VALUE, results.s_PartitionFieldValue, *docPtr, true);
    }
    m_Writer.addStringFieldCopyToObj(FUNCTION, results.s_FunctionName, *docPtr);
    m_Writer.addStringFieldCopyToObj(FUNCTION_DESCRIPTION,
                                     results.s_FunctionDescription, *docPtr);
    m_Writer.addDoubleArrayFieldToObj(TYPICAL, results.s_BaselineMean, *docPtr);
    m_Writer.addDoubleArrayFieldToObj(ACTUAL, results.s_CurrentMean, *docPtr);
}

void CJsonOutputWriter::addPopulationFields(const CHierarchicalResultsWriter::TResults& results,
                                            TDocumentWeakPtr weakDoc) {
    TDocumentPtr docPtr = weakDoc.lock();
    if (!docPtr) {
        LOG_ERROR(<< "Inconsistent program state. JSON document unavailable.");
        return;
    }

    // record_score, probability, fieldName, byFieldName,
    // overFieldName, overFieldValue, partitionFieldName, partitionFieldValue,
    // function, causes, influences?
    m_Writer.addDoubleFieldToObj(INITIAL_RECORD_SCORE,
                                 results.s_NormalizedAnomalyScore, *docPtr);
    m_Writer.addDoubleFieldToObj(RECORD_SCORE, results.s_NormalizedAnomalyScore, *docPtr);
    m_Writer.addDoubleFieldToObj(PROBABILITY, results.s_Probability, *docPtr);
    m_Writer.addStringFieldCopyToObj(FIELD_NAME, results.s_MetricValueField, *docPtr);
    // There are no by field values at this level for population
    // results - they're in the "causes" object
    m_Writer.addStringFieldCopyToObj(BY_FIELD_NAME, results.s_ByFieldName, *docPtr);
    if (!results.s_OverFieldName.empty()) {
        m_Writer.addStringFieldCopyToObj(OVER_FIELD_NAME, results.s_OverFieldName, *docPtr);
        // If name is present then force output of value too, even when empty
        m_Writer.addStringFieldCopyToObj(OVER_FIELD_VALUE,
                                         results.s_OverFieldValue, *docPtr, true);
    }
    if (!results.s_PartitionFieldName.empty()) {
        m_Writer.addStringFieldCopyToObj(PARTITION_FIELD_NAME,
                                         results.s_PartitionFieldName, *docPtr);
        // If name is present then force output of value too, even when empty
        m_Writer.addStringFieldCopyToObj(
            PARTITION_FIELD_VALUE, results.s_PartitionFieldValue, *docPtr, true);
    }
    m_Writer.addStringFieldCopyToObj(FUNCTION, results.s_FunctionName, *docPtr);
    m_Writer.addStringFieldCopyToObj(FUNCTION_DESCRIPTION,
                                     results.s_FunctionDescription, *docPtr);

    // Add nested causes
    if (m_NestedDocs.size() > 0) {
        rapidjson::Value causeArray = m_Writer.makeArray(m_NestedDocs.size());
        for (size_t index = 0; index < m_NestedDocs.size(); ++index) {
            TDocumentWeakPtr nwDocPtr = m_NestedDocs[index];
            TDocumentPtr nDocPtr = nwDocPtr.lock();
            if (!nDocPtr) {
                LOG_ERROR(<< "Inconsistent program state. JSON document unavailable.");
                continue;
            }
            rapidjson::Value& docAsValue = *nDocPtr;

            m_Writer.pushBack(docAsValue, causeArray);
        }
        m_Writer.addMember(CAUSES, causeArray, *docPtr);

        m_NestedDocs.clear();
    } else {
        LOG_WARN(<< "Expected some causes for a population anomaly but got none");
    }
}

void CJsonOutputWriter::addPopulationCauseFields(const CHierarchicalResultsWriter::TResults& results,
                                                 TDocumentWeakPtr weakDoc) {
    TDocumentPtr docPtr = weakDoc.lock();
    if (!docPtr) {
        LOG_ERROR(<< "Inconsistent program state. JSON document unavailable.");
        return;
    }

    // probability, fieldName, byFieldName, byFieldValue,
    // overFieldName, overFieldValue, partitionFieldName, partitionFieldValue,
    // function, typical, actual, influences
    m_Writer.addDoubleFieldToObj(PROBABILITY, results.s_Probability, *docPtr);
    m_Writer.addStringFieldCopyToObj(FIELD_NAME, results.s_MetricValueField, *docPtr);
    if (!results.s_ByFieldName.empty()) {
        m_Writer.addStringFieldCopyToObj(BY_FIELD_NAME, results.s_ByFieldName, *docPtr);
        // If name is present then force output of value too, even when empty
        m_Writer.addStringFieldCopyToObj(BY_FIELD_VALUE, results.s_ByFieldValue,
                                         *docPtr, true);
        // But allow correlatedByFieldValue to be unset if blank
        m_Writer.addStringFieldCopyToObj(CORRELATED_BY_FIELD_VALUE,
                                         results.s_CorrelatedByFieldValue, *docPtr);
    }
    if (!results.s_OverFieldName.empty()) {
        m_Writer.addStringFieldCopyToObj(OVER_FIELD_NAME, results.s_OverFieldName, *docPtr);
        // If name is present then force output of value too, even when empty
        m_Writer.addStringFieldCopyToObj(OVER_FIELD_VALUE,
                                         results.s_OverFieldValue, *docPtr, true);
    }
    if (!results.s_PartitionFieldName.empty()) {
        m_Writer.addStringFieldCopyToObj(PARTITION_FIELD_NAME,
                                         results.s_PartitionFieldName, *docPtr);
        // If name is present then force output of value too, even when empty
        m_Writer.addStringFieldCopyToObj(
            PARTITION_FIELD_VALUE, results.s_PartitionFieldValue, *docPtr, true);
    }
    m_Writer.addStringFieldCopyToObj(FUNCTION, results.s_FunctionName, *docPtr);
    m_Writer.addStringFieldCopyToObj(FUNCTION_DESCRIPTION,
                                     results.s_FunctionDescription, *docPtr);
    m_Writer.addDoubleArrayFieldToObj(TYPICAL, results.s_PopulationAverage, *docPtr);
    m_Writer.addDoubleArrayFieldToObj(ACTUAL, results.s_FunctionValue, *docPtr);
}

void CJsonOutputWriter::addInfluences(const CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPrDoublePrVec& influenceResults,
                                      TDocumentWeakPtr weakDoc) {
    if (influenceResults.empty()) {
        return;
    }

    TDocumentPtr docPtr = weakDoc.lock();
    if (!docPtr) {
        LOG_ERROR(<< "Inconsistent program state. JSON document unavailable.");
        return;
    }

    //! This function takes the raw c_str pointers of the string objects in
    //! influenceResults. These strings must exist up to the time the results
    //! are written

    using TCharPtrDoublePr = std::pair<const char*, double>;
    using TCharPtrDoublePrVec = std::vector<TCharPtrDoublePr>;
    using TCharPtrDoublePrVecIter = TCharPtrDoublePrVec::iterator;
    using TCharPtrCharPtrDoublePrVecPr = std::pair<const char*, TCharPtrDoublePrVec>;
    using TStrCharPtrCharPtrDoublePrVecPrUMap =
        boost::unordered_map<std::string, TCharPtrCharPtrDoublePrVecPr>;
    using TStrCharPtrCharPtrDoublePrVecPrUMapIter = TStrCharPtrCharPtrDoublePrVecPrUMap::iterator;

    TStrCharPtrCharPtrDoublePrVecPrUMap influences;

    // group by influence field
    for (const auto& influenceResult : influenceResults) {
        TCharPtrCharPtrDoublePrVecPr infResult(influenceResult.first.first->c_str(),
                                               TCharPtrDoublePrVec());
        auto insertResult = influences.emplace(*influenceResult.first.first, infResult);

        insertResult.first->second.second.emplace_back(
            influenceResult.first.second->c_str(), influenceResult.second);
    }

    // Order by influence
    for (TStrCharPtrCharPtrDoublePrVecPrUMapIter iter = influences.begin();
         iter != influences.end(); ++iter) {
        std::sort(iter->second.second.begin(), iter->second.second.end(), INFLUENCE_LESS);
    }

    rapidjson::Value influencesDoc = m_Writer.makeArray(influences.size());

    for (TStrCharPtrCharPtrDoublePrVecPrUMapIter iter = influences.begin();
         iter != influences.end(); ++iter) {
        rapidjson::Value influenceDoc(rapidjson::kObjectType);

        rapidjson::Value values = m_Writer.makeArray(influences.size());
        for (TCharPtrDoublePrVecIter arrayIter = iter->second.second.begin();
             arrayIter != iter->second.second.end(); ++arrayIter) {
            m_Writer.pushBack(arrayIter->first, values);
        }

        m_Writer.addMember(INFLUENCER_FIELD_NAME, iter->second.first, influenceDoc);
        m_Writer.addMember(INFLUENCER_FIELD_VALUES, values, influenceDoc);
        m_Writer.pushBack(influenceDoc, influencesDoc);
    }

    // Note influences are written using the field name "influencers"
    m_Writer.addMember(INFLUENCERS, influencesDoc, *docPtr);
}

void CJsonOutputWriter::addEventRateFields(const CHierarchicalResultsWriter::TResults& results,
                                           TDocumentWeakPtr weakDoc) {
    TDocumentPtr docPtr = weakDoc.lock();
    if (!docPtr) {
        LOG_ERROR(<< "Inconsistent program state. JSON document unavailable.");
        return;
    }

    // record_score, probability, fieldName, byFieldName, byFieldValue, partitionFieldName,
    // partitionFieldValue, functionName, typical, actual, influences?

    m_Writer.addDoubleFieldToObj(INITIAL_RECORD_SCORE,
                                 results.s_NormalizedAnomalyScore, *docPtr);
    m_Writer.addDoubleFieldToObj(RECORD_SCORE, results.s_NormalizedAnomalyScore, *docPtr);
    m_Writer.addDoubleFieldToObj(PROBABILITY, results.s_Probability, *docPtr);
    m_Writer.addStringFieldCopyToObj(FIELD_NAME, results.s_MetricValueField, *docPtr);
    if (!results.s_ByFieldName.empty()) {
        m_Writer.addStringFieldCopyToObj(BY_FIELD_NAME, results.s_ByFieldName, *docPtr);
        // If name is present then force output of value too, even when empty
        m_Writer.addStringFieldCopyToObj(BY_FIELD_VALUE, results.s_ByFieldValue,
                                         *docPtr, true);
        // But allow correlatedByFieldValue to be unset if blank
        m_Writer.addStringFieldCopyToObj(CORRELATED_BY_FIELD_VALUE,
                                         results.s_CorrelatedByFieldValue, *docPtr);
    }
    if (!results.s_PartitionFieldName.empty()) {
        m_Writer.addStringFieldCopyToObj(PARTITION_FIELD_NAME,
                                         results.s_PartitionFieldName, *docPtr);
        // If name is present then force output of value too, even when empty
        m_Writer.addStringFieldCopyToObj(
            PARTITION_FIELD_VALUE, results.s_PartitionFieldValue, *docPtr, true);
    }
    m_Writer.addStringFieldCopyToObj(FUNCTION, results.s_FunctionName, *docPtr);
    m_Writer.addStringFieldCopyToObj(FUNCTION_DESCRIPTION,
                                     results.s_FunctionDescription, *docPtr);
    m_Writer.addDoubleArrayFieldToObj(TYPICAL, results.s_BaselineMean, *docPtr);
    m_Writer.addDoubleArrayFieldToObj(ACTUAL, results.s_CurrentMean, *docPtr);
}

void CJsonOutputWriter::addInfluencerFields(bool isBucketInfluencer,
                                            const model::CHierarchicalResults::TNode& node,
                                            TDocumentWeakPtr weakDoc) {
    TDocumentPtr docPtr = weakDoc.lock();
    if (!docPtr) {
        LOG_ERROR(<< "Inconsistent program state. JSON document unavailable.");
        return;
    }

    m_Writer.addDoubleFieldToObj(PROBABILITY, node.probability(), *docPtr);
    m_Writer.addDoubleFieldToObj(isBucketInfluencer ? INITIAL_SCORE : INITIAL_INFLUENCER_SCORE,
                                 node.s_NormalizedAnomalyScore, *docPtr);
    m_Writer.addDoubleFieldToObj(isBucketInfluencer ? ANOMALY_SCORE : INFLUENCER_SCORE,
                                 node.s_NormalizedAnomalyScore, *docPtr);
    const std::string& personFieldName = *node.s_Spec.s_PersonFieldName;
    m_Writer.addStringFieldCopyToObj(INFLUENCER_FIELD_NAME, personFieldName, *docPtr);
    if (isBucketInfluencer) {
        m_Writer.addDoubleFieldToObj(RAW_ANOMALY_SCORE, node.s_RawAnomalyScore, *docPtr);
    } else {
        if (!personFieldName.empty()) {
            // If name is present then force output of value too, even when empty
            m_Writer.addStringFieldCopyToObj(
                INFLUENCER_FIELD_VALUE, *node.s_Spec.s_PersonFieldValue, *docPtr, true);
        }
    }
}

void CJsonOutputWriter::limitNumberRecords(size_t count) {
    m_RecordOutputLimit = count;
}

size_t CJsonOutputWriter::limitNumberRecords() const {
    return m_RecordOutputLimit;
}

void CJsonOutputWriter::persistNormalizer(const model::CHierarchicalResultsNormalizer& normalizer,
                                          core_t::TTime& persistTime) {
    std::string quantilesState;
    normalizer.toJson(m_LastNonInterimBucketTime, "api", quantilesState, true);

    m_Writer.StartObject();
    m_Writer.String(QUANTILES);
    // No need to copy the strings as the doc is written straight away
    CModelSnapshotJsonWriter::writeQuantileState(
        m_JobId, quantilesState, m_LastNonInterimBucketTime, m_Writer);
    m_Writer.EndObject();

    persistTime = core::CTimeUtils::now();
    LOG_DEBUG(<< "Wrote quantiles state at " << persistTime);
}

void CJsonOutputWriter::pushAllocator(const std::string& allocatorName) {
    m_Writer.pushAllocator(allocatorName);
}

void CJsonOutputWriter::popAllocator() {
    m_Writer.popAllocator();
}

void CJsonOutputWriter::reportMemoryUsage(const model::CResourceMonitor::SResults& results) {
    m_Writer.StartObject();
    CModelSizeStatsJsonWriter::write(m_JobId, results, m_Writer);
    m_Writer.EndObject();

    LOG_TRACE(<< "Wrote memory usage results");
}

void CJsonOutputWriter::acknowledgeFlush(const std::string& flushId,
                                         core_t::TTime lastFinalizedBucketEnd) {
    m_Writer.StartObject();
    m_Writer.String(FLUSH);
    m_Writer.StartObject();

    m_Writer.String(ID);
    m_Writer.String(flushId);
    m_Writer.String(LAST_FINALIZED_BUCKET_END);
    m_Writer.Time(lastFinalizedBucketEnd);

    m_Writer.EndObject();
    m_Writer.EndObject();

    // this shouldn't hang in buffers, so flush
    m_Writer.flush();
    LOG_TRACE(<< "Wrote flush with ID " << flushId);
}

void CJsonOutputWriter::writeCategoryDefinition(int categoryId,
                                                const std::string& terms,
                                                const std::string& regex,
                                                std::size_t maxMatchingFieldLength,
                                                const TStrSet& examples) {
    m_Writer.StartObject();
    m_Writer.String(CATEGORY_DEFINITION);
    m_Writer.StartObject();
    m_Writer.String(JOB_ID);
    m_Writer.String(m_JobId);
    m_Writer.String(CATEGORY_ID);
    m_Writer.Int(categoryId);
    m_Writer.String(TERMS);
    m_Writer.String(terms);
    m_Writer.String(REGEX);
    m_Writer.String(regex);
    m_Writer.String(MAX_MATCHING_LENGTH);
    m_Writer.Uint64(maxMatchingFieldLength);
    m_Writer.String(EXAMPLES);
    m_Writer.StartArray();
    for (TStrSetCItr itr = examples.begin(); itr != examples.end(); ++itr) {
        const std::string& example = *itr;
        m_Writer.String(example);
    }
    m_Writer.EndArray();
    m_Writer.EndObject();
    m_Writer.EndObject();
}

CJsonOutputWriter::SBucketData::SBucketData()
    : s_MaxBucketInfluencerNormalizedAnomalyScore(0.0), s_InputEventCount(0),
      s_RecordCount(0), s_BucketSpan(0), s_HighestProbability(-1),
      s_LowestInfluencerScore(101.0), s_LowestBucketInfluencerScore(101.0) {
}
}
}
