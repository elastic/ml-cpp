/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
//! \brief
//! Analyse event rates and metric time series for anomalies
//!
//! DESCRIPTION:\n
//! Expects to be streamed CSV or length encoded data on STDIN or a named pipe,
//! and sends its JSON results to STDOUT or another named pipe.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Standalone program.
//!
#include <core/CBlockingCallCancellingTimer.h>
#include <core/CDataAdder.h>
#include <core/CDataSearcher.h>
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CLogger.h>
#include <core/CProcessPriority.h>
#include <core/CProgramCounters.h>
#include <core/CStringUtils.h>
#include <core/CoreTypes.h>

#include <ver/CBuildInfo.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CLimits.h>
#include <model/ModelTypes.h>

#include <api/CAnomalyJob.h>
#include <api/CAnomalyJobConfig.h>
#include <api/CCmdSkeleton.h>
#include <api/CCsvInputParser.h>
#include <api/CFieldDataCategorizer.h>
#include <api/CIoManager.h>
#include <api/CJsonOutputWriter.h>
#include <api/CLengthEncodedInputParser.h>
#include <api/CModelSnapshotJsonWriter.h>
#include <api/CPersistenceManager.h>
#include <api/CSingleStreamDataAdder.h>
#include <api/CSingleStreamSearcher.h>
#include <api/CStateRestoreStreamFilter.h>

#include <seccomp/CSystemCallFilter.h>

#include "CCmdLineParser.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>

int main(int argc, char** argv) {

    // Register the set of counters in which this program is interested
    const ml::counter_t::TCounterTypeSet counters{
        ml::counter_t::E_TSADNumberNewPeopleNotAllowed,
        ml::counter_t::E_TSADNumberNewPeople,
        ml::counter_t::E_TSADNumberNewPeopleRecycled,
        ml::counter_t::E_TSADNumberApiRecordsHandled,
        ml::counter_t::E_TSADMemoryUsage,
        ml::counter_t::E_TSADPeakMemoryUsage,
        ml::counter_t::E_TSADNumberMemoryUsageChecks,
        ml::counter_t::E_TSADNumberMemoryUsageEstimates,
        ml::counter_t::E_TSADNumberRecordsNoTimeField,
        ml::counter_t::E_TSADNumberTimeFieldConversionErrors,
        ml::counter_t::E_TSADNumberTimeOrderErrors,
        ml::counter_t::E_TSADNumberNewAttributesNotAllowed,
        ml::counter_t::E_TSADNumberNewAttributes,
        ml::counter_t::E_TSADNumberNewAttributesRecycled,
        ml::counter_t::E_TSADNumberByFields,
        ml::counter_t::E_TSADNumberOverFields,
        ml::counter_t::E_TSADNumberExcludedFrequentInvocations,
        ml::counter_t::E_TSADNumberSamplesOutsideLatencyWindow,
        ml::counter_t::E_TSADNumberMemoryLimitModelCreationFailures,
        ml::counter_t::E_TSADNumberPrunedItems,
        ml::counter_t::E_TSADAssignmentMemoryBasis};

    ml::core::CProgramCounters::registerProgramCounterTypes(counters);

    using TStrVec = ml::autodetect::CCmdLineParser::TStrVec;

    // Read command line options
    std::string configFile;
    std::string filtersConfigFile;
    std::string eventsConfigFile;
    std::string modelConfigFile;
    std::string logProperties;
    std::string logPipe;
    char delimiter{'\t'};
    bool lengthEncodedInput{false};
    std::string timeFormat;
    std::string quantilesStateFile;
    bool deleteStateFiles{false};
    std::size_t bucketPersistInterval{0};
    ml::core_t::TTime namedPipeConnectTimeout{
        ml::core::CBlockingCallCancellingTimer::DEFAULT_TIMEOUT_SECONDS};
    std::string inputFileName;
    bool isInputFileNamedPipe{false};
    std::string outputFileName;
    bool isOutputFileNamedPipe{false};
    std::string restoreFileName;
    bool isRestoreFileNamedPipe{false};
    std::string persistFileName;
    bool isPersistFileNamedPipe{false};
    bool isPersistInForeground{false};
    std::size_t maxAnomalyRecords{100};
    bool memoryUsage{false};
    if (ml::autodetect::CCmdLineParser::parse(
            argc, argv, configFile, filtersConfigFile, eventsConfigFile,
            modelConfigFile, logProperties, logPipe, delimiter, lengthEncodedInput,
            timeFormat, quantilesStateFile, deleteStateFiles, bucketPersistInterval,
            namedPipeConnectTimeout, inputFileName, isInputFileNamedPipe,
            outputFileName, isOutputFileNamedPipe, restoreFileName,
            isRestoreFileNamedPipe, persistFileName, isPersistFileNamedPipe,
            isPersistInForeground, maxAnomalyRecords, memoryUsage) == false) {
        return EXIT_FAILURE;
    }

    ml::core::CBlockingCallCancellingTimer cancellerThread{
        ml::core::CThread::currentThreadId(), std::chrono::seconds{namedPipeConnectTimeout}};

    // Construct the IO manager before reconfiguring the logger, as it performs
    // std::ios actions that only work before first use
    ml::api::CIoManager ioMgr{
        cancellerThread,        inputFileName,         isInputFileNamedPipe,
        outputFileName,         isOutputFileNamedPipe, restoreFileName,
        isRestoreFileNamedPipe, persistFileName,       isPersistFileNamedPipe};

    if (cancellerThread.start() == false) {
        // This log message will probably never been seen as it will go to the
        // real stderr of this process rather than the log pipe...
        LOG_FATAL(<< "Could not start blocking call canceller thread");
        return EXIT_FAILURE;
    }
    if (ml::core::CLogger::instance().reconfigure(
            logPipe, logProperties, cancellerThread.hasCancelledBlockingCall()) == false) {
        LOG_FATAL(<< "Could not reconfigure logging");
        cancellerThread.stop();
        return EXIT_FAILURE;
    }
    cancellerThread.stop();

    // Log the program version immediately after reconfiguring the logger.  This
    // must be done from the program, and NOT a shared library, as each program
    // statically links its own version library.
    LOG_DEBUG(<< ml::ver::CBuildInfo::fullInfo());

    // Reduce memory priority before installing system call filters.
    ml::core::CProcessPriority::reduceMemoryPriority();

    ml::seccomp::CSystemCallFilter::installSystemCallFilter();

    if (ioMgr.initIo() == false) {
        LOG_FATAL(<< "Failed to initialise IO");
        return EXIT_FAILURE;
    }

    // Reduce CPU priority after connecting named pipes so the JVM gets more
    // time when CPU is constrained.  Named pipe connection is time-sensitive,
    // hence is done before reducing CPU priority.
    ml::core::CProcessPriority::reduceCpuPriority();

    ml::api::CAnomalyJobConfig jobConfig;
    if (jobConfig.initFromFiles(configFile, filtersConfigFile, eventsConfigFile) == false) {
        LOG_FATAL(<< "JSON config could not be interpreted");
        return EXIT_FAILURE;
    }

    const ml::api::CAnomalyJobConfig::CAnalysisLimits& analysisLimits =
        jobConfig.analysisLimits();
    ml::model::CLimits limits{isPersistInForeground};
    limits.init(analysisLimits.categorizationExamplesLimit(),
                analysisLimits.modelMemoryLimitMb());

    const ml::api::CAnomalyJobConfig::CAnalysisConfig& analysisConfig =
        jobConfig.analysisConfig();

    bool doingCategorization{analysisConfig.categorizationFieldName().empty() == false};
    TStrVec mutableFields;
    if (doingCategorization) {
        mutableFields.push_back(ml::api::CFieldDataCategorizer::MLCATEGORY_NAME);
    }

    ml::model::CAnomalyDetectorModelConfig modelConfig = analysisConfig.makeModelConfig();

    if (!modelConfigFile.empty() && modelConfig.init(modelConfigFile) == false) {
        LOG_FATAL(<< "ML model config file '" << modelConfigFile << "' could not be loaded");
        return EXIT_FAILURE;
    }

    const ml::api::CAnomalyJobConfig::CModelPlotConfig& modelPlotConfig =
        jobConfig.modelPlotConfig();
    modelConfig.configureModelPlot(modelPlotConfig.enabled(),
                                   modelPlotConfig.annotationsEnabled(),
                                   modelPlotConfig.terms());

    using TDataSearcherUPtr = std::unique_ptr<ml::core::CDataSearcher>;
    const TDataSearcherUPtr restoreSearcher{[isRestoreFileNamedPipe, &ioMgr]() -> TDataSearcherUPtr {
        if (ioMgr.restoreStream()) {
            // Check whether state is restored from a file, if so we assume that this is a debugging case
            // and therefore does not originate from the ML Java code.
            if (!isRestoreFileNamedPipe) {
                // apply a filter to overcome differences in the way persistence vs. restore works
                auto strm = std::make_shared<boost::iostreams::filtering_istream>();
                strm->push(ml::api::CStateRestoreStreamFilter());
                strm->push(*ioMgr.restoreStream());
                return std::make_unique<ml::api::CSingleStreamSearcher>(strm);
            }
            return std::make_unique<ml::api::CSingleStreamSearcher>(ioMgr.restoreStream());
        }
        return nullptr;
    }()};

    using TDataAdderUPtr = std::unique_ptr<ml::core::CDataAdder>;
    const TDataAdderUPtr persister{[&ioMgr]() -> TDataAdderUPtr {
        if (ioMgr.persistStream()) {
            return std::make_unique<ml::api::CSingleStreamDataAdder>(ioMgr.persistStream());
        }
        return nullptr;
    }()};

    ml::core_t::TTime persistInterval{jobConfig.persistInterval()};
    if ((bucketPersistInterval > 0 || persistInterval >= 0) && persister == nullptr) {
        LOG_FATAL(<< "Periodic persistence cannot be enabled using the '"
                  << ((persistInterval >= 0) ? "persistInterval" : "bucketPersistInterval")
                  << "' argument "
                     "unless a place to persist to has been specified using the 'persist' argument");
        return EXIT_FAILURE;
    }

    using TPersistenceManagerUPtr = std::unique_ptr<ml::api::CPersistenceManager>;
    const TPersistenceManagerUPtr persistenceManager{
        [persistInterval, isPersistInForeground, &persister,
         &bucketPersistInterval]() -> TPersistenceManagerUPtr {
            if (persistInterval >= 0 || bucketPersistInterval > 0) {
                return std::make_unique<ml::api::CPersistenceManager>(
                    persistInterval, isPersistInForeground, *persister, bucketPersistInterval);
            }
            return nullptr;
        }()};

    using InputParserCUPtr = std::unique_ptr<ml::api::CInputParser>;
    const InputParserCUPtr inputParser{[lengthEncodedInput, &mutableFields,
                                        &ioMgr, delimiter]() -> InputParserCUPtr {
        if (lengthEncodedInput) {
            return std::make_unique<ml::api::CLengthEncodedInputParser>(
                mutableFields, ioMgr.inputStream());
        }
        return std::make_unique<ml::api::CCsvInputParser>(
            mutableFields, ioMgr.inputStream(), delimiter);
    }()};

    const std::string jobId{jobConfig.jobId()};
    ml::core::CJsonOutputStreamWrapper wrappedOutputStream{ioMgr.outputStream()};
    ml::api::CModelSnapshotJsonWriter modelSnapshotWriter{jobId, wrappedOutputStream};

    // The anomaly job knows how to detect anomalies
    ml::api::CAnomalyJob job{jobId,
                             limits,
                             jobConfig,
                             modelConfig,
                             wrappedOutputStream,
                             std::bind(&ml::api::CModelSnapshotJsonWriter::write,
                                       &modelSnapshotWriter, std::placeholders::_1),
                             persistenceManager.get(),
                             jobConfig.quantilePersistInterval(),
                             jobConfig.dataDescription().timeField(),
                             timeFormat,
                             maxAnomalyRecords};

    if (!quantilesStateFile.empty()) {
        if (job.initNormalizer(quantilesStateFile) == false) {
            LOG_FATAL(<< "Failed to restore quantiles and initialize normalizer");
            return EXIT_FAILURE;
        }
        if (deleteStateFiles) {
            std::remove(quantilesStateFile.c_str());
        }
    }

    // The categorizer knows how to assign categories to records
    ml::api::CFieldDataCategorizer categorizer{
        jobId,
        analysisConfig,
        limits,
        jobConfig.dataDescription().timeField(),
        timeFormat,
        &job,
        wrappedOutputStream,
        persistenceManager.get(),
        analysisConfig.perPartitionCategorizationStopOnWarn()};

    ml::api::CDataProcessor* firstProcessor{nullptr};
    if (doingCategorization) {
        LOG_DEBUG(<< "Applying the categorizer for anomaly detection");
        firstProcessor = &categorizer;
    } else {
        firstProcessor = &job;
    }

    if (persistenceManager != nullptr) {
        persistenceManager->firstProcessorBackgroundPeriodicPersistFunc(std::bind(
            &ml::api::CDataProcessor::periodicPersistStateInBackground, firstProcessor));

        persistenceManager->firstProcessorForegroundPeriodicPersistFunc(std::bind(
            &ml::api::CDataProcessor::periodicPersistStateInForeground, firstProcessor));
    }

    // The skeleton avoids the need to duplicate a lot of boilerplate code
    ml::api::CCmdSkeleton skeleton{restoreSearcher.get(), persister.get(),
                                   *inputParser, *firstProcessor};
    if (skeleton.ioLoop() == false) {
        LOG_FATAL(<< "ML anomaly detector job failed");
        return EXIT_FAILURE;
    }

    if (memoryUsage) {
        job.descriptionAndDebugMemoryUsage();
    }

    // Print out the runtime counters generated during this execution context
    LOG_DEBUG(<< ml::core::CProgramCounters::instance());

    // This message makes it easier to spot process crashes in a log file - if
    // this isn't present in the log for a given PID and there's no other log
    // message indicating early exit then the process has probably core dumped
    LOG_DEBUG(<< "ML anomaly detector job exiting");

    return EXIT_SUCCESS;
}
