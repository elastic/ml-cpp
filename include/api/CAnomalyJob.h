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
#include <model/CHierarchicalResults.h>
#include <model/CHierarchicalResultsAggregator.h>
#include <model/CHierarchicalResultsNormalizer.h>
#include <model/CInterimBucketCorrector.h>
#include <model/CResourceMonitor.h>
#include <model/CSearchKey.h>

#include <api/CDataProcessor.h>
#include <api/CForecastRunner.h>
#include <api/CJsonOutputWriter.h>
#include <api/CModelSnapshotJsonWriter.h>
#include <api/ImportExport.h>

#include <boost/unordered_map.hpp>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace CAnomalyJobTest {
struct testParsePersistControlMessageArgs;
}

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
class CAnomalyJobConfig;
class CPersistenceManager;
class CModelPlotDataJsonWriter;

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
    using TAnomalyDetectorPtr = std::shared_ptr<model::CAnomalyDetector>;
    using TAnomalyDetectorPtrVec = std::vector<TAnomalyDetectorPtr>;
    using TKeyVec = std::vector<model::CSearchKey>;
    using TKeyAnomalyDetectorPtrUMap =
        boost::unordered_map<model::CSearchKey::TStrKeyPr, TAnomalyDetectorPtr, model::CStrKeyPrHash, model::CStrKeyPrEqual>;
    using TKeyCRefAnomalyDetectorPtrPr =
        std::pair<model::CSearchKey::TStrCRefKeyCRefPr, TAnomalyDetectorPtr>;
    using TKeyCRefAnomalyDetectorPtrPrVec = std::vector<TKeyCRefAnomalyDetectorPtrPr>;
    using TModelPlotDataVec = model::CAnomalyDetector::TModelPlotDataVec;
    using TAnnotationVec = model::CAnomalyDetector::TAnnotationVec;

    struct API_EXPORT SRestoredStateDetail {
        ERestoreStateStatus s_RestoredStateStatus;
        boost::optional<std::string> s_Extra;
    };

    struct SBackgroundPersistArgs {
        SBackgroundPersistArgs(core_t::TTime time,
                               const model::CResourceMonitor::SModelSizeStats& modelSizeStats,
                               const model::CInterimBucketCorrector& interimBucketCorrector,
                               const model::CHierarchicalResultsAggregator& aggregator,
                               core_t::TTime latestRecordTime,
                               core_t::TTime lastResultsTime);

        core_t::TTime s_Time;
        model::CResourceMonitor::SModelSizeStats s_ModelSizeStats;
        model::CInterimBucketCorrector s_InterimBucketCorrector;
        model::CHierarchicalResultsAggregator s_Aggregator;
        std::string s_NormalizerState;
        core_t::TTime s_LatestRecordTime;
        core_t::TTime s_LastResultsTime;
        TKeyCRefAnomalyDetectorPtrPrVec s_Detectors;
    };

    using TBackgroundPersistArgsPtr = std::shared_ptr<SBackgroundPersistArgs>;

public:
    CAnomalyJob(const std::string& jobId,
                model::CLimits& limits,
                CAnomalyJobConfig& jobConfig,
                model::CAnomalyDetectorModelConfig& modelConfig,
                core::CJsonOutputStreamWrapper& outputBuffer,
                const TPersistCompleteFunc& persistCompleteFunc,
                CPersistenceManager* persistenceManager,
                core_t::TTime maxQuantileInterval,
                const std::string& timeFieldName,
                const std::string& timeFieldFormat,
                std::size_t maxAnomalyRecords);

    ~CAnomalyJob() override;

    //! Receive a single record to be processed, and produce output
    //! with any required modifications
    bool handleRecord(const TStrStrUMap& dataRowFields, TOptionalTime time) override;

    //! Perform any final processing once all input data has been seen.
    void finalise() override;

    //! Restore previously saved state
    bool restoreState(core::CDataSearcher& restoreSearcher,
                      core_t::TTime& completeToTime) override;

    //! Persist state in the foreground. As this blocks the current thread of execution
    //! it should only be called in special circumstances, e.g. at job close, where it won't impact job analysis.
    bool persistStateInForeground(core::CDataAdder& persister,
                                  const std::string& descriptionPrefix) override;

    //! Persist the current model state in the foreground regardless of whether
    //! any results have been output.
    bool doPersistStateInForeground(core::CDataAdder& persister,
                                    const std::string& description,
                                    const std::string& snapshotId,
                                    core_t::TTime snapshotTimestamp);

    //! Persist state of the residual models only.
    //! This method is not intended to be called in production code
    //! as it only persists a very small subset of model state with longer,
    //! human readable tags.
    bool persistModelsState(core::CDataAdder& persister,
                            core_t::TTime timestamp,
                            const std::string& outputFormat);

    //! Initialise normalizer from quantiles state
    virtual bool initNormalizer(const std::string& quantilesStateFile);

    //! How many records did we handle?
    std::uint64_t numRecordsHandled() const override;

    //! Is persistence needed?
    bool isPersistenceNeeded(const std::string& description) const override;

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
    //! \p processingTime is the processing time of the bucket
    void writeOutResults(bool interim,
                         model::CHierarchicalResults& results,
                         core_t::TTime bucketTime,
                         std::uint64_t processingTime);

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
    bool backgroundPersistState();

    //! This is the function that is called in a different thread to the
    //! main processing when background persistence is triggered.
    bool runBackgroundPersist(TBackgroundPersistArgsPtr args, core::CDataAdder& persister);

    //! This function is called from the persistence manager when foreground persistence is triggered
    bool runForegroundPersist(core::CDataAdder& persister);

    //! Persist the detectors to a stream.
    bool persistCopiedState(const std::string& description,
                            const std::string& snapshotId,
                            core_t::TTime snapshotTimestamp,
                            core_t::TTime time,
                            const TKeyCRefAnomalyDetectorPtrPrVec& detectors,
                            const model::CResourceMonitor::SModelSizeStats& modelSizeStats,
                            const model::CInterimBucketCorrector& interimBucketCorrector,
                            const model::CHierarchicalResultsAggregator& aggregator,
                            const std::string& normalizerState,
                            core_t::TTime latestRecordTime,
                            core_t::TTime lastResultsTime,
                            core::CDataAdder& persister);

    //! Persist current state due to the periodic persistence being triggered.
    bool periodicPersistStateInBackground() override;
    bool periodicPersistStateInForeground() override;

    //! Persist state of the residual models only.
    //! This method is not intended to be called in production code.
    //! \p outputFormat specifies the format of the output document and may
    //! either be JSON or XML.
    bool persistModelsState(const TKeyCRefAnomalyDetectorPtrPrVec& detectors,
                            core::CDataAdder& persister,
                            core_t::TTime timestamp,
                            const std::string& outputFormat);

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

    //! Roll time forward to \p time
    void timeNow(core_t::TTime time);

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
    void updateNormalizerAndNormalizeResults(bool isInterim,
                                             model::CHierarchicalResults& results);

    //! Outputs results for the buckets that are within the specified range.
    //! The range includes the start but does not include the end.
    void outputResultsWithinRange(bool isInterim, core_t::TTime start, core_t::TTime end);

    //! Generate the model plot for the models of the specified detector in the
    //! specified time range.
    void generateModelPlot(core_t::TTime startTime,
                           core_t::TTime endTime,
                           const model::CAnomalyDetector& detector,
                           TModelPlotDataVec& modelPlotData);

    //! Write the pre-generated model plot to the output stream of the user's
    //! choosing: either file or streamed to the API
    void writeOutModelPlot(const TModelPlotDataVec& modelPlotData);

    //! Write the annotations to the output stream.
    void writeOutAnnotations(const TAnnotationVec& annotations);

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

    TAnomalyDetectorPtr
    makeDetector(const model::CAnomalyDetectorModelConfig& modelConfig,
                 model::CLimits& limits,
                 const std::string& partitionFieldValue,
                 core_t::TTime firstTime,
                 const model::CAnomalyDetector::TModelFactoryCPtr& modelFactory);

    //! Populate detector keys from the anomaly job config.
    void populateDetectorKeys(const CAnomalyJobConfig& jobConfig, TKeyVec& keys);

    //! Extract the field called \p fieldName from \p dataRowFields.
    const std::string* fieldValue(const std::string& fieldName, const TStrStrUMap& dataRowFields);

    //! Extract the required fields from \p dataRowFields
    //! and add the new record to \p detector
    void addRecord(const TAnomalyDetectorPtr detector,
                   core_t::TTime time,
                   const TStrStrUMap& dataRowFields);

    //! Parses a control message requesting that model state be persisted.
    //! Extracts optional arguments to be used for persistence.
    static bool parsePersistControlMessageArgs(const std::string& controlMessageArgs,
                                               core_t::TTime& snapshotTimestamp,
                                               std::string& snapshotId,
                                               std::string& snapshotDescription);

    //! Perform foreground persistence if control message contains valid optional
    //! arguments else request a background persist
    void processPersistControlMessage(const std::string& controlMessageArgs);

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

    //! Configuration settings for the analysis parsed from
    //! JSON configuration file.
    //! Note that this is a non-const reference as it needs to be capable of
    //! being modified by job updates (and those changes reflected wherever a
    //! reference is held).
    CAnomalyJobConfig& m_JobConfig;

    //! The model configuration
    model::CAnomalyDetectorModelConfig& m_ModelConfig;

    //! Keep count of how many records we've handled
    std::uint64_t m_NumRecordsHandled;

    //! Detector keys.
    TKeyVec m_DetectorKeys;

    //! Map of objects to provide the inner workings
    TKeyAnomalyDetectorPtrUMap m_Detectors;

    //! The end time of the last bucket out of latency window we've seen
    core_t::TTime m_LastFinalisedBucketEndTime;

    //! Optional function to be called when persistence is complete
    TPersistCompleteFunc m_PersistCompleteFunc;

    //! License restriction on the number of detectors allowed
    std::size_t m_MaxDetectors;

    //! Pointer to the persistence manager. May be nullptr if state persistence
    //! is not required, for example in unit tests.
    CPersistenceManager* m_PersistenceManager;

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

    //! Flag indicating whether or not time has been advanced.
    bool m_TimeAdvanced{false};

    // Test case access
    friend struct CAnomalyJobTest::testParsePersistControlMessageArgs;
};
}
}

#endif // INCLUDED_ml_api_CAnomalyJob_h
