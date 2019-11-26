/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CJsonOutputWriter_h
#define INCLUDED_ml_api_CJsonOutputWriter_h

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CRapidJsonConcurrentLineWriter.h>
#include <core/CSmallVector.h>
#include <core/CoreTypes.h>

#include <model/CCategoryExamplesCollector.h>
#include <model/CHierarchicalResults.h>
#include <model/CResourceMonitor.h>

#include <api/CHierarchicalResultsWriter.h>
#include <api/COutputHandler.h>
#include <api/ImportExport.h>

#include <rapidjson/document.h>

#include <boost/optional.hpp>

#include <iosfwd>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace ml {
namespace model {
class CHierarchicalResultsNormalizer;
}
namespace core {
template<typename>
class CScopedRapidJsonPoolAllocator;
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
    using TDocumentPtr = std::shared_ptr<rapidjson::Document>;
    using TDocumentWeakPtr = std::weak_ptr<rapidjson::Document>;
    using TDocumentWeakPtrVec = std::vector<TDocumentWeakPtr>;
    using TDocumentWeakPtrVecItr = TDocumentWeakPtrVec::iterator;
    using TDocumentWeakPtrVecCItr = TDocumentWeakPtrVec::const_iterator;

    using TDocumentWeakPtrIntPr = std::pair<TDocumentWeakPtr, int>;
    using TDocumentWeakPtrIntPrVec = std::vector<TDocumentWeakPtrIntPr>;
    using TDocumentWeakPtrIntPrVecItr = TDocumentWeakPtrIntPrVec::iterator;
    using TStrDocumentPtrVecMap = std::map<std::string, TDocumentWeakPtrVec>;

    using TStrVec = std::vector<std::string>;
    using TStr1Vec = core::CSmallVector<std::string, 1>;
    using TTimeVec = std::vector<core_t::TTime>;
    using TDoubleVec = std::vector<double>;
    using TDoubleDoublePr = std::pair<double, double>;
    using TDoubleDoublePrVec = std::vector<TDoubleDoublePr>;
    using TDoubleDoubleDoublePrPr = std::pair<double, TDoubleDoublePr>;
    using TDoubleDoubleDoublePrPrVec = std::vector<TDoubleDoubleDoublePrPr>;
    using TStringDoublePr = std::pair<std::string, double>;
    using TStringDoublePrVec = std::vector<TStringDoublePr>;

    using TValuePtr = std::shared_ptr<rapidjson::Value>;

    //! Structure to buffer up information about each bucket that we have
    //! unwritten results for
    struct SBucketData {
        SBucketData();

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

        //! scheduled event descriptions
        TStr1Vec s_ScheduledEventDescriptions;
    };

    using TTimeBucketDataMap = std::map<core_t::TTime, SBucketData>;
    using TTimeBucketDataMapItr = TTimeBucketDataMap::iterator;
    using TTimeBucketDataMapCItr = TTimeBucketDataMap::const_iterator;
    using TStrFSet = model::CCategoryExamplesCollector::TStrFSet;
    using TStrFSetCItr = TStrFSet::const_iterator;

public:
    //! Constructor that causes output to be written to the specified wrapped stream
    CJsonOutputWriter(const std::string& jobId, core::CJsonOutputStreamWrapper& strmOut);

    //! Destructor flushes the stream
    virtual ~CJsonOutputWriter();

    // Bring the other overload of fieldNames() into scope
    using COutputHandler::fieldNames;

    //! Set field names.  In this class this function has no effect and it
    //! always returns true
    bool fieldNames(const TStrVec& fieldNames, const TStrVec& extraFieldNames) override;

    //! Write the data row fields as a JSON object
    bool writeRow(const TStrStrUMap& dataRowFields,
                  const TStrStrUMap& overrideDataRowFields) override;

    //! Limit the output to the top count anomalous records and influencers.
    //! Each detector will write no more than count records and influencers
    //! per bucket (i.e. a max of N records, N influencers and N bucket
    //! influencers).
    //! The bucket time influencer does not add to this count but only
    //! if it is added after all the other bucket influencers
    void limitNumberRecords(size_t count);

    //! A value of 0 indicates no limit has been set
    size_t limitNumberRecords() const;

    //! Close the JSON structures and flush output.
    //! This method should only be called once and will have no affect
    //! on subsequent invocations
    void finalise() override;

    //! Accept a result from the anomaly detector
    //! Virtual for testing mocks
    virtual bool acceptResult(const CHierarchicalResultsWriter::TResults& results);

    //! Accept the influencer
    bool acceptInfluencer(core_t::TTime time,
                          const model::CHierarchicalResults::TNode& node,
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
    void reportMemoryUsage(const model::CResourceMonitor::SResults& results);

    //! Acknowledge a flush request by echoing back the flush ID
    void acknowledgeFlush(const std::string& flushId, core_t::TTime lastFinalizedBucketEnd);

    //! Write a category definition
    void writeCategoryDefinition(int categoryId,
                                 const std::string& terms,
                                 const std::string& regex,
                                 std::size_t maxMatchingFieldLength,
                                 const TStrFSet& examples);

    //! Persist a normalizer by writing its state to the output
    void persistNormalizer(const model::CHierarchicalResultsNormalizer& normalizer,
                           core_t::TTime& persistTime);

private:
    template<typename>
    friend class core::CScopedRapidJsonPoolAllocator;
    // hooks for the CScopedRapidJsonPoolAllocator interface

    //! use a new allocator for JSON output processing
    //! \p allocatorName A unique identifier for the allocator
    void pushAllocator(const std::string& allocatorName);

    //! revert to using the previous allocator for JSON output processing
    void popAllocator();

private:
    //! Write out all the JSON documents that have been built up for
    //! a particular bucket
    void writeBucket(bool isInterim,
                     core_t::TTime bucketTime,
                     SBucketData& bucketData,
                     uint64_t bucketProcessingTime);

    //! Add the fields for a metric detector
    void addMetricFields(const CHierarchicalResultsWriter::TResults& results,
                         TDocumentWeakPtr weakDoc);

    //! Write the fields for a population detector
    void addPopulationFields(const CHierarchicalResultsWriter::TResults& results,
                             TDocumentWeakPtr weakDoc);

    //! Write the fields for a population detector cause
    void addPopulationCauseFields(const CHierarchicalResultsWriter::TResults& results,
                                  TDocumentWeakPtr weakDoc);

    //! Write the fields for an event rate detector
    void addEventRateFields(const CHierarchicalResultsWriter::TResults& results,
                            TDocumentWeakPtr weakDoc);

    //! Add the influencer fields to the doc
    void addInfluencerFields(bool isBucketInfluencer,
                             const model::CHierarchicalResults::TNode& node,
                             TDocumentWeakPtr weakDoc);

    //! Write the influence results.
    void addInfluences(const CHierarchicalResultsWriter::TStoredStringPtrStoredStringPtrPrDoublePrVec& influenceResults,
                       TDocumentWeakPtr weakDoc);

private:
    //! The job ID
    std::string m_JobId;

    //! JSON line writer
    core::CRapidJsonConcurrentLineWriter m_Writer;

    //! Time of last non-interim bucket written to output
    core_t::TTime m_LastNonInterimBucketTime;

    //! Has the output been finalised?
    bool m_Finalised;

    //! Max number of records to write for each bucket/detector
    size_t m_RecordOutputLimit;

    //! Vector for building up documents representing nested sub-results.
    //! The documents in this vector will reference memory owned by
    //! m_JsonPoolAllocator.  (Hence this is declared after the memory pool
    //! so that it's destroyed first when the destructor runs.)
    TDocumentWeakPtrVec m_NestedDocs;

    //! Bucket data waiting to be written.  The map is keyed on bucket time.
    //! The documents in this map will reference memory owned by
    //! m_JsonPoolAllocator.  (Hence this is declared after the memory pool
    //! so that it's destroyed first when the destructor runs.)
    TTimeBucketDataMap m_BucketDataByTime;
};
}
}

#endif // INCLUDED_ml_api_CJsonOutputWriter_h
