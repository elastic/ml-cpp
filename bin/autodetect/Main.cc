/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
//! \brief
//! Analyse event rates and metrics
//!
//! DESCRIPTION:\n
//! Expects to be streamed CSV or length encoded data on STDIN or a named pipe,
//! and sends its JSON results to STDOUT or another named pipe.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Standalone program.
//!
#include <core/CDataAdder.h>
#include <core/CDataSearcher.h>
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CLogger.h>
#include <core/CProcessPriority.h>
#include <core/CStatistics.h>
#include <core/CoreTypes.h>

#include <ver/CBuildInfo.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CLimits.h>
#include <model/ModelTypes.h>

#include <api/CAnomalyJob.h>
#include <api/CBackgroundPersister.h>
#include <api/CCmdSkeleton.h>
#include <api/CCsvInputParser.h>
#include <api/CFieldConfig.h>
#include <api/CFieldDataTyper.h>
#include <api/CIoManager.h>
#include <api/CJsonOutputWriter.h>
#include <api/CLengthEncodedInputParser.h>
#include <api/CModelSnapshotJsonWriter.h>
#include <api/COutputChainer.h>
#include <api/CSingleStreamDataAdder.h>
#include <api/CSingleStreamSearcher.h>
#include <api/CStateRestoreStreamFilter.h>

#include <seccomp/CSystemCallFilter.h>

#include "CCmdLineParser.h"

#include <boost/bind.hpp>

#include <memory>
#include <string>

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    using TStrVec = ml::autodetect::CCmdLineParser::TStrVec;

    // Read command line options
    std::string limitConfigFile;
    std::string modelConfigFile;
    std::string fieldConfigFile;
    std::string modelPlotConfigFile;
    std::string jobId;
    std::string logProperties;
    std::string logPipe;
    ml::core_t::TTime bucketSpan(0);
    ml::core_t::TTime latency(0);
    std::string summaryCountFieldName;
    char delimiter('\t');
    bool lengthEncodedInput(false);
    std::string timeField(ml::api::CAnomalyJob::DEFAULT_TIME_FIELD_NAME);
    std::string timeFormat;
    std::string quantilesStateFile;
    bool deleteStateFiles(false);
    ml::core_t::TTime persistInterval(-1);
    ml::core_t::TTime maxQuantileInterval(-1);
    std::string inputFileName;
    bool isInputFileNamedPipe(false);
    std::string outputFileName;
    bool isOutputFileNamedPipe(false);
    std::string restoreFileName;
    bool isRestoreFileNamedPipe(false);
    std::string persistFileName;
    bool isPersistFileNamedPipe(false);
    size_t maxAnomalyRecords(100u);
    bool memoryUsage(false);
    std::size_t bucketResultsDelay(0);
    bool multivariateByFields(false);
    TStrVec clauseTokens;
    if (ml::autodetect::CCmdLineParser::parse(
            argc, argv, limitConfigFile, modelConfigFile, fieldConfigFile,
            modelPlotConfigFile, jobId, logProperties, logPipe, bucketSpan, latency,
            summaryCountFieldName, delimiter, lengthEncodedInput, timeField,
            timeFormat, quantilesStateFile, deleteStateFiles, persistInterval,
            maxQuantileInterval, inputFileName, isInputFileNamedPipe, outputFileName,
            isOutputFileNamedPipe, restoreFileName, isRestoreFileNamedPipe, persistFileName,
            isPersistFileNamedPipe, maxAnomalyRecords, memoryUsage, bucketResultsDelay,
            multivariateByFields, clauseTokens) == false) {
        return EXIT_FAILURE;
    }

    // Construct the IO manager before reconfiguring the logger, as it performs
    // std::ios actions that only work before first use
    ml::api::CIoManager ioMgr(inputFileName, isInputFileNamedPipe, outputFileName,
                              isOutputFileNamedPipe, restoreFileName, isRestoreFileNamedPipe,
                              persistFileName, isPersistFileNamedPipe);

    if (ml::core::CLogger::instance().reconfigure(logPipe, logProperties) == false) {
        LOG_FATAL(<< "Could not reconfigure logging");
        return EXIT_FAILURE;
    }

    // Log the program version immediately after reconfiguring the logger.  This
    // must be done from the program, and NOT a shared library, as each program
    // statically links its own version library.
    LOG_DEBUG(<< ml::ver::CBuildInfo::fullInfo());

    ml::core::CProcessPriority::reducePriority();

    ml::seccomp::CSystemCallFilter::installSystemCallFilter();

    if (ioMgr.initIo() == false) {
        LOG_FATAL(<< "Failed to initialise IO");
        return EXIT_FAILURE;
    }

    if (jobId.empty()) {
        LOG_FATAL(<< "No job ID specified");
        return EXIT_FAILURE;
    }

    ml::model::CLimits limits;
    if (!limitConfigFile.empty() && limits.init(limitConfigFile) == false) {
        LOG_FATAL(<< "Ml limit config file '" << limitConfigFile << "' could not be loaded");
        return EXIT_FAILURE;
    }

    ml::api::CFieldConfig fieldConfig;

    ml::model_t::ESummaryMode summaryMode(
        summaryCountFieldName.empty() ? ml::model_t::E_None : ml::model_t::E_Manual);
    ml::model::CAnomalyDetectorModelConfig modelConfig =
        ml::model::CAnomalyDetectorModelConfig::defaultConfig(
            bucketSpan, summaryMode, summaryCountFieldName, latency,
            bucketResultsDelay, multivariateByFields);
    modelConfig.detectionRules(ml::model::CAnomalyDetectorModelConfig::TIntDetectionRuleVecUMapCRef(
        fieldConfig.detectionRules()));
    modelConfig.scheduledEvents(ml::model::CAnomalyDetectorModelConfig::TStrDetectionRulePrVecCRef(
        fieldConfig.scheduledEvents()));

    if (!modelConfigFile.empty() && modelConfig.init(modelConfigFile) == false) {
        LOG_FATAL(<< "Ml model config file '" << modelConfigFile << "' could not be loaded");
        return EXIT_FAILURE;
    }

    if (!modelPlotConfigFile.empty() &&
        modelConfig.configureModelPlot(modelPlotConfigFile) == false) {
        LOG_FATAL(<< "Ml model plot config file '" << modelPlotConfigFile
                  << "' could not be loaded");
        return EXIT_FAILURE;
    }

    using TDataSearcherCUPtr = const std::unique_ptr<ml::core::CDataSearcher>;
    TDataSearcherCUPtr restoreSearcher{[isRestoreFileNamedPipe,
                                        &ioMgr]() -> ml::core::CDataSearcher* {
        if (ioMgr.restoreStream()) {
            // Check whether state is restored from a file, if so we assume that this is a debugging case
            // and therefore does not originate from X-Pack.
            if (!isRestoreFileNamedPipe) {
                // apply a filter to overcome differences in the way persistence vs. restore works
                auto strm = std::make_shared<boost::iostreams::filtering_istream>();
                strm->push(ml::api::CStateRestoreStreamFilter());
                strm->push(*ioMgr.restoreStream());
                return new ml::api::CSingleStreamSearcher(strm);
            }
            return new ml::api::CSingleStreamSearcher(ioMgr.restoreStream());
        }
        return nullptr;
    }()};

    using TDataAdderCUPtr = const std::unique_ptr<ml::core::CDataAdder>;
    TDataAdderCUPtr persister{[&ioMgr]() -> ml::core::CDataAdder* {
        if (ioMgr.persistStream()) {
            return new ml::api::CSingleStreamDataAdder(ioMgr.persistStream());
        }
        return nullptr;
    }()};

    if (persistInterval >= 0 && persister == nullptr) {
        LOG_FATAL(<< "Periodic persistence cannot be enabled using the 'persistInterval' argument "
                     "unless a place to persist to has been specified using the 'persist' argument");
        return EXIT_FAILURE;
    }

    using TBackgroundPersisterCUPtr = const std::unique_ptr<ml::api::CBackgroundPersister>;
    TBackgroundPersisterCUPtr periodicPersister{
        [persistInterval, &persister]() -> ml::api::CBackgroundPersister* {
            if (persistInterval >= 0) {
                return new ml::api::CBackgroundPersister(persistInterval, *persister);
            }
            return nullptr;
        }()};

    using InputParserCUPtr = const std::unique_ptr<ml::api::CInputParser>;
    InputParserCUPtr inputParser{[lengthEncodedInput, &ioMgr, delimiter]() -> ml::api::CInputParser* {
        if (lengthEncodedInput) {
            return new ml::api::CLengthEncodedInputParser(ioMgr.inputStream());
        }
        return new ml::api::CCsvInputParser(ioMgr.inputStream(), delimiter);
    }()};

    ml::core::CJsonOutputStreamWrapper wrappedOutputStream(ioMgr.outputStream());

    ml::api::CModelSnapshotJsonWriter modelSnapshotWriter(jobId, wrappedOutputStream);
    if (fieldConfig.initFromCmdLine(fieldConfigFile, clauseTokens) == false) {
        LOG_FATAL(<< "Field config could not be interpreted");
        return EXIT_FAILURE;
    }

    // The anomaly job knows how to detect anomalies
    ml::api::CAnomalyJob job(jobId, limits, fieldConfig, modelConfig, wrappedOutputStream,
                             boost::bind(&ml::api::CModelSnapshotJsonWriter::write,
                                         &modelSnapshotWriter, _1),
                             periodicPersister.get(), maxQuantileInterval,
                             timeField, timeFormat, maxAnomalyRecords);

    if (!quantilesStateFile.empty()) {
        if (job.initNormalizer(quantilesStateFile) == false) {
            LOG_FATAL(<< "Failed to restore quantiles and initialize normalizer");
            return EXIT_FAILURE;
        }
        if (deleteStateFiles) {
            ::remove(quantilesStateFile.c_str());
        }
    }

    ml::api::CDataProcessor* firstProcessor(&job);

    // Chain the categorizer's output to the anomaly detector's input
    ml::api::COutputChainer outputChainer(job);

    ml::api::CJsonOutputWriter fieldDataTyperOutputWriter(jobId, wrappedOutputStream);

    // The typer knows how to assign categories to records
    ml::api::CFieldDataTyper typer(jobId, fieldConfig, limits, outputChainer,
                                   fieldDataTyperOutputWriter);

    if (fieldConfig.fieldNameSuperset().count(ml::api::CFieldDataTyper::MLCATEGORY_NAME) > 0) {
        LOG_DEBUG(<< "Applying the categorization typer for anomaly detection");
        firstProcessor = &typer;
    }

    if (periodicPersister != nullptr) {
        periodicPersister->firstProcessorPeriodicPersistFunc(boost::bind(
            &ml::api::CDataProcessor::periodicPersistState, firstProcessor, _1));
    }

    // The skeleton avoids the need to duplicate a lot of boilerplate code
    ml::api::CCmdSkeleton skeleton(restoreSearcher.get(), persister.get(),
                                   *inputParser, *firstProcessor);
    bool ioLoopSucceeded(skeleton.ioLoop());

    // Unfortunately we cannot rely on destruction to finalise the output writer
    // as it must be finalised before the skeleton is destroyed, and C++
    // destruction order means the skeleton will be destroyed before the output
    // writer as it was constructed last.
    fieldDataTyperOutputWriter.finalise();

    if (!ioLoopSucceeded) {
        LOG_FATAL(<< "Ml anomaly detector job failed");
        return EXIT_FAILURE;
    }

    if (memoryUsage) {
        job.descriptionAndDebugMemoryUsage();
    }

    // Print out the runtime stats generated during this execution context
    LOG_DEBUG(<< ml::core::CStatistics::instance());

    // This message makes it easier to spot process crashes in a log file - if
    // this isn't present in the log for a given PID and there's no other log
    // message indicating early exit then the process has probably core dumped
    LOG_DEBUG(<< "Ml anomaly detector job exiting");

    return EXIT_SUCCESS;
}
