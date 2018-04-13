/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_api_CAnomalyJob_h
#define INCLUDED_ml_api_CAnomalyJob_h

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CStopWatch.h>
#include <core/CoreTypes.h>

#include <model/CAnomalyDetector.h>
#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CBucketQueue.h>
#include <model/CHierarchicalResults.h>
#include <model/CHierarchicalResultsAggregator.h>
#include <model/CHierarchicalResultsNormalizer.h>
#include <model/CResourceMonitor.h>
#include <model/CResultsQueue.h>
#include <model/CSearchKey.h>

#include <api/CDataProcessor.h>
#include <api/CForecastRunner.h>
#include <api/CJsonOutputWriter.h>
#include <api/CModelSnapshotJsonWriter.h>
#include <api/ImportExport.h>

#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>

#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <stdint.h>

class CBackgroundPersisterTest;
class CAnomalyJobTest;

namespace ml {
namespace core {
class CDataAdder;
class CDataSearcher;
class CStateRestoreTraverser;
}
namespace model {
class CHierarchicalResults;
class CLimits;
}
namespace api {
class CBackgroundPersister;
class CModelPlotDataJsonWriter;
class CFieldConfig;

//! \brief
//! The Ml anomaly detector.
//!
//! DESCRIPTION:\n
//! Take a stream of input records and read those records
//! according to the given field config.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Input must be in ascending time order.
//!
//! The output format is so complex that this class requires its output
//! handler to be a CJsonOutputWriter rather than a writer for an
//! arbitrary format
//!
class API_EXPORT CAnomalyJob : public CDataProcessor {
public:
    //! Elasticsearch index for state
    static const std::string ML_STATE_INDEX;

    //! Discriminant for Elasticsearch IDs
    static const std::string STATE_TYPE;

    //! Input field names
    static const std::string EMPTY_STRING;
    static const std::string DEFAULT_TIME_FIELD_NAME;

public:
    //! Enum represents the result of persisted Model restoration
    //! Possible states are:
    //!   -# IncorrectVersion: The version of the stored model state
    //!      does not match the anomaly detector version.
    //!   -# UnexpectedTag: State is malformed or could not be parsed
    //!      correctly
    //!   -# MemoryLimitReached: The detector could not be allocated
    //!      becasuse it would violate the memory usage restrictions
    //!   -# NotRestoredToTime: The detector was not restored to the
    //!      requested time
    //!   -# Success:
    //!   -# Failure:
    enum ERestoreStateStatus {
        E_IncorrectVersion,
        E_UnexpectedTag,
        E_MemoryLimitReached,
        E_NotRestoredToTime,
        E_NoDetectorsRecovered,
        E_Success,
        E_Failure
    };

public:
    using TPersistCompleteFunc =
        std::function<void(const CModelSnapshotJsonWriter::SModelSnapshotReport&)>;
    using TAnomalyDetectorPtr = model::CAnomalyDetector::TAnomalyDetectorPtr;
    using TAnomalyDetectorPtrVec = std::vector<TAnomalyDetectorPtr>;
    using TAnomalyDetectorPtrVecItr = std::vector<TAnomalyDetectorPtr>::iterator;
    using TAnomalyDetectorPtrVecCItr = std::vector<TAnomalyDetectorPtr>::const_iterator;
    using TKeyVec = std::vector<model::CSearchKey>;
    using TKeyAnomalyDetectorPtrUMap =
        boost::unordered_map<model::CSearchKey::TStrKeyPr, TAnomalyDetectorPtr, model::CStrKeyPrHash, model::CStrKeyPrEqual>;
    using TKeyCRefAnomalyDetectorPtrPr =
        std::pair<model::CSearchKey::TStrCRefKeyCRefPr, TAnomalyDetectorPtr>;
    using TKeyCRefAnomalyDetectorPtrPrVec = std::vector<TKeyCRefAnomalyDetectorPtrPr>;
    using TModelPlotDataVec = model::CAnomalyDetector::TModelPlotDataVec;
    using TModelPlotDataVecCItr = TModelPlotDataVec::const_iterator;
    using TModelPlotDataVecQueue = model::CBucketQueue<TModelPlotDataVec>;

    struct API_EXPORT SRestoredStateDetail {
        ERestoreStateStatus s_RestoredStateStatus;
        boost::optional<std::string> s_Extra;
    };

    struct SBackgroundPersistArgs {
        SBackgroundPersistArgs(const model::CResultsQueue& resultsQueue,
                               const TModelPlotDataVecQueue& modelPlotQueue,
                               core_t::TTime time,
                               const model::CResourceMonitor::SResults& modelSizeStats,
                               const model::CHierarchicalResultsAggregator& aggregator,
                               core_t::TTime latestRecordTime,
                               core_t::TTime lastResultsTime);

        model::CResultsQueue s_ResultsQueue;
        TModelPlotDataVecQueue s_ModelPlotQueue;
        core_t::TTime s_Time;
        model::CResourceMonitor::SResults s_ModelSizeStats;
        model::CHierarchicalResultsAggregator s_Aggregator;
        std::string s_NormalizerState;
        core_t::TTime s_LatestRecordTime;
        core_t::TTime s_LastResultsTime;
        TKeyCRefAnomalyDetectorPtrPrVec s_Detectors;
    };

    using TBackgroundPersistArgsPtr = boost::shared_ptr<SBackgroundPersistArgs>;

public:
    CAnomalyJob(const std::string& jobId,
                model::CLimits& limits,
                CFieldConfig& fieldConfig,
                model::CAnomalyDetectorModelConfig& modelConfig,
                core::CJsonOutputStreamWrapper& outputBuffer,
                const TPersistCompleteFunc& persistCompleteFunc = TPersistCompleteFunc(),
                CBackgroundPersister* periodicPersister = nullptr,
                core_t::TTime maxQuantileInterval = -1,
                const std::string& timeFieldName = DEFAULT_TIME_FIELD_NAME,
                const std::string& timeFieldFormat = EMPTY_STRING,
                size_t maxAnomalyRecords = 0u);

    virtual ~CAnomalyJob();

    //! We're going to be writing to a new output stream
    virtual void newOutputStream();

    //! Access the output handler
    virtual COutputHandler& outputHandler();

    //! Receive a single record to be processed, and produce output
    //! with any required modifications
    virtual bool handleRecord(const TStrStrUMap& dataRowFields);

    //! Perform any final processing once all input data has been seen.
    virtual void finalise();

    //! Restore previously saved state
    virtual bool restoreState(core::CDataSearcher& restoreSearcher,
                              core_t::TTime& completeToTime);

    //! Persist current state
    virtual bool persistState(core::CDataAdder& persister);

    //! Initialise normalizer from quantiles state
    virtual bool initNormalizer(const std::string& quantilesStateFile);

    //! How many records did we handle?
    virtual uint64_t numRecordsHandled() const;

    //! Log a list of the detectors and keys
    void description() const;

    //! Log a list of the detectors, keys and their memory usage
    void descriptionAndDebugMemoryUsage() const;

    //! Extra information on the success/failure of restoring the model state.
    //! In certain situations such as no data being loaded from the restorer
    //! or the stored state version is wrong the restoreState function will
    //! still return true. If interested in these kinds of errors check them
    //! here.
    const SRestoredStateDetail& restoreStateStatus() const;

private:
    //! NULL pointer that we can take a long-lived const reference to
    static const TAnomalyDetectorPtr NULL_DETECTOR;

private:
    //! Handle a control message.  The first character of the control
    //! message indicates its type.  Currently defined types are:
    //! ' ' => Dummy message to force all previously uploaded data through
    //!        buffers
    //! 'f' => Echo a flush ID so that the attached process knows that data
    //!        sent previously has all been processed
    //! 'i' => Generate interim results
    bool handleControlMessage(const std::string& controlMessage);

    //! Write out the results for the bucket starting at \p bucketStartTime.
    void outputResults(core_t::TTime bucketStartTime);

    //! Write out interim results for the bucket starting at \p bucketStartTime.
    void outputInterimResults(core_t::TTime bucketStartTime);

    //! Helper function for outputResults.
    //! \p processingTimer is the processing time can be written to the bucket
    //! \p sumPastProcessingTime is the total time previously spent processing
    //! but resulted in no bucket being outputted.
    void writeOutResults(bool interim,
                         model::CHierarchicalResults& results,
                         core_t::TTime bucketTime,
                         uint64_t processingTime,
                         uint64_t sumPastProcessingTime);

    //! Reset buckets in the range specified by the control message.
    void resetBuckets(const std::string& controlMessage);

    //! Attempt to restore the detectors
    bool restoreState(core::CStateRestoreTraverser& traverser,
                      core_t::TTime& completeToTime,
                      std::size_t& numDetectors);

    //! Attempt to restore one detector from an already-created traverser.
    bool restoreSingleDetector(core::CStateRestoreTraverser& traverser);

    //! Restore the detector identified by \p key and \p partitionFieldValue
    //! from \p traverser.
    bool restoreDetectorState(const model::CSearchKey& key,
                              const std::string& partitionFieldValue,
                              core::CStateRestoreTraverser& traverser);

    //! Persist current state in the background
    bool backgroundPersistState(CBackgroundPersister& backgroundPersister);

    //! This is the function that is called in a different thread to the
    //! main processing when background persistence is triggered.
    bool runBackgroundPersist(TBackgroundPersistArgsPtr args, core::CDataAdder& persister);

    //! Persist the detectors to a stream.
    bool persistState(const std::string& descriptionPrefix,
                      const model::CResultsQueue& resultsQueue,
                      const TModelPlotDataVecQueue& modelPlotQueue,
                      core_t::TTime time,
                      const TKeyCRefAnomalyDetectorPtrPrVec& detectors,
                      const model::CResourceMonitor::SResults& modelSizeStats,
                      const model::CHierarchicalResultsAggregator& aggregator,
                      const std::string& normalizerState,
                      core_t::TTime latestRecordTime,
                      core_t::TTime lastResultsTime,
                      core::CDataAdder& persister);

    //! Persist current state due to the periodic persistence being triggered.
    virtual bool periodicPersistState(CBackgroundPersister& persister);

    //! Acknowledge a flush request
    void acknowledgeFlush(const std::string& flushId);

    //! Advance time until \p time, if it can be parsed.
    //!
    //! This also calls outputBucketResultsUntil, so may generate results if
    //! a bucket boundary is crossed and updates time in *all* the detector
    //! models.
    void advanceTime(const std::string& time);

    //! Output any results new results which are available at \p time.
    void outputBucketResultsUntil(core_t::TTime time);

    //! Skip time to the bucket end of \p time, if it can be parsed.
    void skipTime(const std::string& time);

    //! Rolls time to \p endTime while skipping sampling the models for buckets
    //! within the gap.
    //!
    //! \param[in] endTime The end of the time interval to skip sampling.
    void skipSampling(core_t::TTime endTime);

    //! Outputs queued results and resets the queue to the given \p startTime
    void flushAndResetResultsQueue(core_t::TTime startTime);

    //! Roll time forward to \p time
    void timeNow(core_t::TTime time);

    //! Get the bucketLength, or half the bucketLength if
    //! out-of-phase buckets are active
    core_t::TTime effectiveBucketLength() const;

    //! Update configuration
    void updateConfig(const std::string& config);

    //! Generate interim results.
    void generateInterimResults(const std::string& controlMessage);

    //! Parses the time range in a control message assuming the time range follows after a
    //! single character code (e.g. starts with 'i10 20').
    bool parseTimeRangeInControlMessage(const std::string& controlMessage,
                                        core_t::TTime& start,
                                        core_t::TTime& end);

    //! Update equalizers if not interim and aggregate.
    void updateAggregatorAndAggregate(bool isInterim, model::CHierarchicalResults& results);

    //! Update quantiles if not interim and normalize.
    void updateQuantilesAndNormalize(bool isInterim, model::CHierarchicalResults& results);

    //! Outputs results for the buckets that are within the specified range.
    //! The range includes the start but does not include the end.
    void outputResultsWithinRange(bool isInterim, core_t::TTime start, core_t::TTime end);

    //! Generate the model plot for the models of the specified detector in the
    //! specified time range.
    void generateModelPlot(core_t::TTime startTime,
                           core_t::TTime endTime,
                           const model::CAnomalyDetector& detector);

    //! Write the pre-generated model plot to the output stream of the user's
    //! choosing: either file or streamed to the API
    void writeOutModelPlot(core_t::TTime resultsTime);

    //! Write the pre-generated model plot to the output stream of the user's
    //! choosing: either file or streamed to the API
    void writeOutModelPlot(core_t::TTime, CModelPlotDataJsonWriter& writer);

    //! Persist one detector to a stream.
    //! This method is static so that there is no danger of it accessing
    //! the member variables of an object.  This makes it safer to call
    //! from within a persistence thread that's working off a cloned
    //! anomaly detector.
    static void persistIndividualDetector(const model::CAnomalyDetector& detector,
                                          core::CStatePersistInserter& inserter);

    //! Iterate over the models, refresh their memory status, and send a report
    //! to the API
    void refreshMemoryAndReport();

    //! Update configuration
    void doForecast(const std::string& controlMessage);

    model::CAnomalyDetector::TAnomalyDetectorPtr
    makeDetector(int identifier,
                 const model::CAnomalyDetectorModelConfig& modelConfig,
                 model::CLimits& limits,
                 const std::string& partitionFieldValue,
                 core_t::TTime firstTime,
                 const model::CAnomalyDetector::TModelFactoryCPtr& modelFactory);

    //! Populate detector keys from the field config.
    void populateDetectorKeys(const CFieldConfig& fieldConfig, TKeyVec& keys);

    //! Extract the field called \p fieldName from \p dataRowFields.
    const std::string* fieldValue(const std::string& fieldName, const TStrStrUMap& dataRowFields);

    //! Extract the required fields from \p dataRowFields
    //! and add the new record to \p detector
    void addRecord(const TAnomalyDetectorPtr detector,
                   core_t::TTime time,
                   const TStrStrUMap& dataRowFields);

protected:
    //! Get all the detectors.
    void detectors(TAnomalyDetectorPtrVec& detectors) const;

    //! Get the detectors by parition
    const TKeyAnomalyDetectorPtrUMap& detectorPartitionMap() const;

    //! Get all sorted references to the detectors.
    void sortedDetectors(TKeyCRefAnomalyDetectorPtrPrVec& detectors) const;

    //! Get a reference to the detector for a given key
    const TAnomalyDetectorPtr& detectorForKey(bool isRestoring,
                                              core_t::TTime time,
                                              const model::CSearchKey& key,
                                              const std::string& partitionFieldValue,
                                              model::CResourceMonitor& resourceMonitor);

    //! Prune all the models
    void pruneAllModels();

private:
    //! The job ID
    std::string m_JobId;

    //! Configurable limits
    model::CLimits& m_Limits;

    //! Stream used by the output writer
    core::CJsonOutputStreamWrapper& m_OutputStream;

    //! Responsible for performing forecasts
    CForecastRunner m_ForecastRunner;

    //! Object to which the output is passed
    CJsonOutputWriter m_JsonOutputWriter;

    //! Field names to use for the analysis
    CFieldConfig& m_FieldConfig;

    //! The model configuration
    model::CAnomalyDetectorModelConfig& m_ModelConfig;

    //! Keep count of how many records we've handled
    uint64_t m_NumRecordsHandled;

    //! Detector keys.
    TKeyVec m_DetectorKeys;

    //! Map of objects to provide the inner workings
    TKeyAnomalyDetectorPtrUMap m_Detectors;

    //! The end time of the last bucket out of latency window we've seen
    core_t::TTime m_LastFinalisedBucketEndTime;

    //! Optional function to be called when persistence is complete
    TPersistCompleteFunc m_PersistCompleteFunc;

    //! Name of field holding the time
    std::string m_TimeFieldName;

    //! Time field format.  Blank means seconds since the epoch, i.e. the
    //! time field can be converted to a time_t by simply converting the
    //! string to a number.
    std::string m_TimeFieldFormat;

    //! License restriction on the number of detectors allowed
    size_t m_MaxDetectors;

    //! Pointer to periodic persister that works in the background.  May be
    //! nullptr if this object is not responsible for starting periodic
    //! persistence.
    CBackgroundPersister* m_PeriodicPersister;

    //! If we haven't output quantiles for this long due to a big anomaly
    //! we'll output them to reflect decay.  Non-positive values mean never.
    core_t::TTime m_MaxQuantileInterval;

    //! What was the wall clock time when we last persisted the
    //! normalizer? The normalizer is persisted for two reasons:
    //! either there was a significant change or more than a
    //! certain period of time has passed since last time it was persisted.
    core_t::TTime m_LastNormalizerPersistTime;

    //! Latest record time seen.
    core_t::TTime m_LatestRecordTime;

    //! Last time we sent a finalised result to the API.
    core_t::TTime m_LastResultsTime;

    //! When the model state was restored was it entirely successful.
    //! Extra information about any errors that may have occurred
    SRestoredStateDetail m_RestoredStateDetail;

    //! The hierarchical results aggregator.
    model::CHierarchicalResultsAggregator m_Aggregator;

    //! The hierarchical results normalizer.
    model::CHierarchicalResultsNormalizer m_Normalizer;

    //! Store the last N half-buckets' results in order
    //! to choose the best result
    model::CResultsQueue m_ResultsQueue;

    //! Also store the model plot for the buckets for each
    //! result time - these will be output when the corresponding
    //! result is output
    TModelPlotDataVecQueue m_ModelPlotQueue;

    friend class ::CBackgroundPersisterTest;
    friend class ::CAnomalyJobTest;
};
}
}

#endif // INCLUDED_ml_api_CAnomalyJob_h
