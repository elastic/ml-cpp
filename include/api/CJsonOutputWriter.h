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
#ifndef INCLUDED_ml_api_CJsonOutputWriter_h
#define INCLUDED_ml_api_CJsonOutputWriter_h

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CMutex.h>
#include <core/CoreTypes.h>
#include <core/CRapidJsonConcurrentLineWriter.h>
#include <core/CSmallVector.h>

#include <model/CHierarchicalResults.h>
#include <model/CResourceMonitor.h>

#include <api/CCategoryExamplesCollector.h>
#include <api/CHierarchicalResultsWriter.h>
#include <api/COutputHandler.h>
#include <api/ImportExport.h>

#include <rapidjson/document.h>

#include <boost/optional.hpp>
#include <boost/shared_ptr.hpp>

#include <iosfwd>
#include <map>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>


namespace ml {
namespace model {
class CHierarchicalResultsNormalizer;
}
namespace core {
template <typename> class CScopedRapidJsonPoolAllocator;
}
namespace api {

//! \brief
//! Write output data in JSON format
//!
//! DESCRIPTION:\n
//! Outputs the anomaly detector results.
//!
//! The fields written for each detector type are hardcoded.  If new fields
//! are passed then this class needs updating.
//!
//! Use limitNumberRecords() to limit the number of records and influencers
//! written for each bucket.  If set to a non-zero value only the top N
//! least probable records, the top N influencers and top N bucket influencers
//! are written. This useful in situations where a very
//! large number of field combinations are being analysed.  Setting the limit
//! to zero implies all results are to be kept.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The writer buffers results until endOutputBatch() is called.  All results
//! for a given bucket must be passed to the writer between calls to
//! endOutputBatch().  Once endOutputBatch() has been called, no further
//! results can be added for any buckets for which results were previously
//! added.
//!
//! The simple count detector is not output.  Instead, its "actual" value
//! is set as the bucket's eventCount field.
//!
//! Empty string fields are not written to the output.
//!
//! Memory for values added to the output documents is allocated from a pool (to
//! reduce allocation cost and memory fragmentation).  This pool is cleared
//! between buckets to avoid excessive accumulation of memory over long periods.
//! It is crucial that the pool is only cleared when no documents reference
//! memory within it.  This is achieved by only clearing the pool
//! (m_JsonPoolAllocator) when the m_BucketDataByTime vector is empty.  Care
//! must be taken if the memory pool is used to allocate memory for long-lived
//! documents that are not stored within m_BucketDataByTime.  (It might be
//! better to have a separate pool if this situation ever arises in the future.)
//!
//! Population anomalies consist of overall results and breakdown results.
//! There is an assumption that the overall result for a population anomaly
//! immediately follows the corresponding breakdown results.  If this assumption
//! ever becomes invalid then the way this class nests breakdown results inside
//! overall results will need changing.
//!
//! Unusual scores and anomaly scores are generated for output bucket/record
//! using the latest quantiles.  The idea is to reduce the processing load
//! by avoiding the need for the Java process to request re-normalisation of
//! every single result as soon as it's written to Elasticsearch.  Quantiles
//! are only written to the output when they change a lot, when 2 hours of
//! wall-clock time elapses, or when the writer is finalised.  (When the
//! Java process receives these updated quantiles it will request
//! re-normalisation of previous results using the normalize
//! process, so it's best that this doesn't happen too often.)
//!
class API_EXPORT CJsonOutputWriter : public COutputHandler {
    public:
        typedef boost::shared_ptr<rapidjson::Document>      TDocumentPtr;
        typedef boost::weak_ptr<rapidjson::Document>        TDocumentWeakPtr;
        typedef std::vector<TDocumentWeakPtr>               TDocumentWeakPtrVec;
        typedef TDocumentWeakPtrVec::iterator               TDocumentWeakPtrVecItr;
        typedef TDocumentWeakPtrVec::const_iterator         TDocumentWeakPtrVecCItr;

        typedef std::pair<TDocumentWeakPtr, int>            TDocumentWeakPtrIntPr;
        typedef std::vector<TDocumentWeakPtrIntPr>          TDocumentWeakPtrIntPrVec;
        typedef TDocumentWeakPtrIntPrVec::iterator          TDocumentWeakPtrIntPrVecItr;
        typedef std::map<std::string, TDocumentWeakPtrVec>  TStrDocumentPtrVecMap;

        typedef std::vector<std::string>                    TStrVec;
        typedef core::CSmallVector<std::string, 1>          TStr1Vec;
        typedef std::vector<core_t::TTime>                  TTimeVec;
        typedef std::vector<double>                         TDoubleVec;
        typedef std::pair<double, double>                   TDoubleDoublePr;
        typedef std::vector<TDoubleDoublePr>                TDoubleDoublePrVec;
        typedef std::pair<double, TDoubleDoublePr>          TDoubleDoubleDoublePrPr;
        typedef std::vector<TDoubleDoubleDoublePrPr>        TDoubleDoubleDoublePrPrVec;
        typedef std::pair<std::string, double>              TStringDoublePr;
        typedef std::vector<TStringDoublePr>                TStringDoublePrVec;

        typedef boost::shared_ptr<rapidjson::Value>         TValuePtr;

        //! Structure to buffer up information about each bucket that we have
        //! unwritten results for
        struct SBucketData {
            SBucketData(void);

            //! The max normalized anomaly score of the bucket influencers
            double s_MaxBucketInfluencerNormalizedAnomalyScore;

            //! Count of input events for the bucket
            size_t s_InputEventCount;

            //! Count of result records in the bucket for which results are
            //! being built up
            size_t s_RecordCount;

            //! The bucketspan of this bucket
            core_t::TTime s_BucketSpan;

            //! The result record documents to be written, in a vector keyed on
            //! detector index
            TDocumentWeakPtrIntPrVec s_DocumentsToWrite;

            //! Bucket Influencer documents
            TDocumentWeakPtrVec s_BucketInfluencerDocuments;

            //! Influencer documents
            TDocumentWeakPtrVec s_InfluencerDocuments;

            // The highest probability of all the records stored
            // in the s_DocumentsToWrite array. Used for filtering
            // new records with a higher probability
            double s_HighestProbability;

            // Used for filtering new influencers
            // when the number to write is limited
            double s_LowestInfluencerScore;

            // Used for filtering new bucket influencers
            // when the number to write is limited
            double s_LowestBucketInfluencerScore;

            //! Partition scores
            TDocumentWeakPtrVec s_PartitionScoreDocuments;

            //! scheduled event descriptions
            TStr1Vec s_ScheduledEventDescriptions;
        };

        typedef std::map<core_t::TTime, SBucketData>   TTimeBucketDataMap;
        typedef TTimeBucketDataMap::iterator           TTimeBucketDataMapItr;
        typedef TTimeBucketDataMap::const_iterator     TTimeBucketDataMapCItr;


        static const std::string JOB_ID;
        static const std::string TIMESTAMP;
        static const std::string BUCKET;
        static const std::string LOG_TIME;
        static const std::string DETECTOR_INDEX;
        static const std::string RECORDS;
        static const std::string EVENT_COUNT;
        static const std::string IS_INTERIM;
        static const std::string PROBABILITY;
        static const std::string RAW_ANOMALY_SCORE;
        static const std::string ANOMALY_SCORE;
        static const std::string RECORD_SCORE;
        static const std::string INITIAL_RECORD_SCORE;
        static const std::string INFLUENCER_SCORE;
        static const std::string INITIAL_INFLUENCER_SCORE;
        static const std::string FIELD_NAME;
        static const std::string BY_FIELD_NAME;
        static const std::string BY_FIELD_VALUE;
        static const std::string CORRELATED_BY_FIELD_VALUE;
        static const std::string TYPICAL;
        static const std::string ACTUAL;
        static const std::string CAUSES;
        static const std::string FUNCTION;
        static const std::string FUNCTION_DESCRIPTION;
        static const std::string OVER_FIELD_NAME;
        static const std::string OVER_FIELD_VALUE;
        static const std::string PARTITION_FIELD_NAME;
        static const std::string PARTITION_FIELD_VALUE;
        static const std::string INITIAL_SCORE;
        static const std::string BUCKET_INFLUENCERS;
        static const std::string INFLUENCERS;
        static const std::string INFLUENCER_FIELD_NAME;
        static const std::string INFLUENCER_FIELD_VALUE;
        static const std::string INFLUENCER_FIELD_VALUES;
        static const std::string FLUSH;
        static const std::string ID;
        static const std::string LAST_FINALIZED_BUCKET_END;
        static const std::string QUANTILE_STATE;
        static const std::string QUANTILES;
        static const std::string MODEL_SIZE_STATS;
        static const std::string MODEL_BYTES;
        static const std::string TOTAL_BY_FIELD_COUNT;
        static const std::string TOTAL_OVER_FIELD_COUNT;
        static const std::string TOTAL_PARTITION_FIELD_COUNT;
        static const std::string BUCKET_ALLOCATION_FAILURES_COUNT;
        static const std::string MEMORY_STATUS;
        static const std::string CATEGORY_DEFINITION;
        static const std::string CATEGORY_ID;
        static const std::string TERMS;
        static const std::string REGEX;
        static const std::string MAX_MATCHING_LENGTH;
        static const std::string EXAMPLES;
        static const std::string MODEL_SNAPSHOT;
        static const std::string SNAPSHOT_ID;
        static const std::string SNAPSHOT_DOC_COUNT;
        static const std::string DESCRIPTION;
        static const std::string LATEST_RECORD_TIME;
        static const std::string BUCKET_SPAN;
        static const std::string LATEST_RESULT_TIME;
        static const std::string PROCESSING_TIME;
        static const std::string TIME_INFLUENCER;
        static const std::string PARTITION_SCORES;
        static const std::string SCHEDULED_EVENTS;

    private:
        typedef CCategoryExamplesCollector::TStrSet TStrSet;
        typedef TStrSet::const_iterator             TStrSetCItr;

        struct SModelSnapshotReport {
            SModelSnapshotReport(core_t::TTime snapshotTimestamp,
                                 const std::string &description,
                                 const std::string &snapshotId,
                                 size_t numDocs,
                                 const model::CResourceMonitor::SResults &modelSizeStats,
                                 const std::string &normalizerState,
                                 core_t::TTime latestRecordTime,
                                 core_t::TTime latestFinalResultTime);

            core_t::TTime s_SnapshotTimestamp;
            std::string s_Description;
            std::string s_SnapshotId;
            size_t s_NumDocs;
            model::CResourceMonitor::SResults s_ModelSizeStats;
            std::string s_NormalizerState;
            core_t::TTime s_LatestRecordTime;
            core_t::TTime s_LatestFinalResultTime;
        };

        typedef std::queue<SModelSnapshotReport> TModelSnapshotReportQueue;

    public:
        //! Constructor that causes output to be written to the specified wrapped stream
        CJsonOutputWriter(const std::string &jobId,
                          core::CJsonOutputStreamWrapper &strmOut);

        //! Destructor flushes the stream
        virtual ~CJsonOutputWriter(void);

        //! Set field names.  In this class this function has no effect and it
        //! always returns true
        virtual bool fieldNames(const TStrVec &fieldNames,
                                const TStrVec &extraFieldNames);

        // Bring the other overload of fieldNames() into scope
        using COutputHandler::fieldNames;

        //! Returns an empty vector
        virtual const TStrVec &fieldNames(void) const;

        //! Write the data row fields as a JSON object
        virtual bool writeRow(const TStrStrUMap &dataRowFields,
                              const TStrStrUMap &overrideDataRowFields);

        //! Limit the output to the top count anomalous records and influencers.
        //! Each detector will write no more than count records and influencers
        //! per bucket (i.e. a max of N records, N influencers and N bucket
        //! influencers).
        //! The bucket time influencer does not add to this count but only
        //! if it is added after all the other bucket influencers
        void limitNumberRecords(size_t count);

        //! A value of 0 indicates no limit has been set
        size_t limitNumberRecords(void) const;

        //! Close the JSON structures and flush output.
        //! This method should only be called once and will have no affect
        //! on subsequent invocations
        virtual void finalise(void);

        //! Receive a count of possible results
        void possibleResultCount(core_t::TTime time, size_t count);

        //! Accept a result from the anomaly detector
        //! Virtual for testing mocks
        virtual bool acceptResult(const CHierarchicalResultsWriter::TResults &results);

        //! Accept the influencer
        bool acceptInfluencer(core_t::TTime time,
                              const model::CHierarchicalResults::TNode &node,
                              bool isBucketInfluencer);

        //! Creates a time bucket influencer.
        //! If limitNumberRecords is set add this influencer after all other influencers
        //! have been added otherwise it may be filtered out if its anomaly score is lower
        //! than the others.
        //! Only one per bucket is expected, this does not add to the influencer
        //! count if limitNumberRecords is used
        virtual void acceptBucketTimeInfluencer(core_t::TTime time,
                                                double probability,
                                                double rawAnomalyScore,
                                                double normalizedAnomalyScore);

        //! This method must be called after all the results for a given bucket
        //! are available.  It triggers the writing of the results.
        bool endOutputBatch(bool isInterim, uint64_t bucketProcessingTime);

        //! Report the current levels of resource usage, as given to us
        //! from the CResourceMonitor via a callback
        void reportMemoryUsage(const model::CResourceMonitor::SResults &results);

        //! Report information about completion of model persistence.
        //! This method can be called in a thread other than the one
        //! receiving the majority of results, so reporting is done
        //! asynchronously.
        void reportPersistComplete(core_t::TTime snapshotTimestamp,
                                   const std::string &description,
                                   const std::string &snapshotId,
                                   size_t numDocs,
                                   const model::CResourceMonitor::SResults &modelSizeStats,
                                   const std::string &normalizerState,
                                   core_t::TTime latestRecordTime,
                                   core_t::TTime latestFinalResultTime);

        //! Acknowledge a flush request by echoing back the flush ID
        void acknowledgeFlush(const std::string &flushId, core_t::TTime lastFinalizedBucketEnd);

        //! Write a category definition
        void writeCategoryDefinition(int categoryId,
                                     const std::string &terms,
                                     const std::string &regex,
                                     std::size_t maxMatchingFieldLength,
                                     const TStrSet &examples);

        //! Persist a normalizer by writing its state to the output
        void persistNormalizer(const model::CHierarchicalResultsNormalizer &normalizer,
                               core_t::TTime &persistTime);

    private:
        template <typename > friend class core::CScopedRapidJsonPoolAllocator;
        // hooks for the CScopedRapidJsonPoolAllocator interface

        //! use a new allocator for JSON output processing
        //! \p allocatorName A unique identifier for the allocator
        void pushAllocator(const std::string &allocatorName);

        //! revert to using the previous allocator for JSON output processing
        void popAllocator();

    private:
        //! Write out all the JSON documents that have been built up for
        //! a particular bucket
        void writeBucket(bool isInterim,
                         core_t::TTime bucketTime,
                         SBucketData &bucketData,
                         uint64_t bucketProcessingTime);

        //! Add the fields for a metric detector
        void addMetricFields(const CHierarchicalResultsWriter::TResults &results,
                             TDocumentWeakPtr weakDoc);

        //! Write the fields for a population detector
        void addPopulationFields(const CHierarchicalResultsWriter::TResults &results,
                                 TDocumentWeakPtr weakDoc);

        //! Write the fields for a population detector cause
        void addPopulationCauseFields(const CHierarchicalResultsWriter::TResults &results,
                                      TDocumentWeakPtr weakDoc);

        //! Write the fields for an event rate detector
        void addEventRateFields(const CHierarchicalResultsWriter::TResults &results,
                                TDocumentWeakPtr weakDoc);

        //! Add the influencer fields to the doc
        void addInfluencerFields(bool isBucketInfluencer,
                                 const model::CHierarchicalResults::TNode &node,
                                 TDocumentWeakPtr weakDoc);

        //! Write the influence results.
        void addInfluences(const CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPrDoublePrVec &influenceResults,
                           TDocumentWeakPtr weakDoc);

        //! Write partition score & probability
        void addPartitionScores(const CHierarchicalResultsWriter::TResults &results,
                                TDocumentWeakPtr weakDoc);

        //! Write any model snapshot reports that are queuing up.
        void writeModelSnapshotReports(void);

        //! Write the JSON object showing current levels of resource usage, as
        //! given to us from the CResourceMonitor via a callback
        void writeMemoryUsageObject(const model::CResourceMonitor::SResults &results);

        //! Write the quantile's state
        void writeQuantileState(const std::string &state, core_t::TTime timestamp);

    private:
        //! The job ID
        std::string                          m_JobId;

        //! JSON line writer
        core::CRapidJsonConcurrentLineWriter m_Writer;

        //! Time of last non-interim bucket written to output
        core_t::TTime                        m_LastNonInterimBucketTime;

        //! Has the output been finalised?
        bool                                 m_Finalised;

        //! Max number of records to write for each bucket/detector
        size_t                               m_RecordOutputLimit;

        //! Vector for building up documents representing nested sub-results.
        //! The documents in this vector will reference memory owned by
        //! m_JsonPoolAllocator.  (Hence this is declared after the memory pool
        //! so that it's destroyed first when the destructor runs.)
        TDocumentWeakPtrVec                  m_NestedDocs;

        //! Bucket data waiting to be written.  The map is keyed on bucket time.
        //! The documents in this map will reference memory owned by
        //! m_JsonPoolAllocator.  (Hence this is declared after the memory pool
        //! so that it's destroyed first when the destructor runs.)
        TTimeBucketDataMap                   m_BucketDataByTime;

        //! Protects the m_ModelSnapshotReports from concurrent access.
        core::CMutex                         m_ModelSnapshotReportsQueueMutex;

        //! Queue of model snapshot reports waiting to be output.
        TModelSnapshotReportQueue            m_ModelSnapshotReports;
};


}
}

#endif // INCLUDED_ml_api_CJsonOutputWriter_h

