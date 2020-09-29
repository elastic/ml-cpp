/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
//! \brief
//! Group machine generated messages into categories by similarity.
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
#include <core/CoreTypes.h>

#include <ver/CBuildInfo.h>

#include <model/CLimits.h>

#include <api/CCmdSkeleton.h>
#include <api/CCsvInputParser.h>
#include <api/CFieldConfig.h>
#include <api/CFieldDataCategorizer.h>
#include <api/CIoManager.h>
#include <api/CJsonOutputWriter.h>
#include <api/CLengthEncodedInputParser.h>
#include <api/CPersistenceManager.h>
#include <api/CSingleStreamDataAdder.h>
#include <api/CSingleStreamSearcher.h>
#include <api/CStateRestoreStreamFilter.h>

#include <seccomp/CSystemCallFilter.h>

#include "CCmdLineParser.h"

#include <chrono>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>

int main(int argc, char** argv) {
    // Read command line options
    std::string limitConfigFile;
    std::string jobId;
    std::string logProperties;
    std::string logPipe;
    char delimiter{'\t'};
    bool lengthEncodedInput{false};
    // Currently there aren't command line options for the time field/format
    // and whether stop-on-warn is enabled.
    // TODO: add options to set these if this program is used in the future
    std::string timeField;
    std::string timeFormat;
    bool stopCategorizationOnWarnStatus{false};
    ml::core_t::TTime persistInterval{-1};
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
    std::string categorizationFieldName;
    if (ml::categorize::CCmdLineParser::parse(
            argc, argv, limitConfigFile, jobId, logProperties, logPipe, delimiter,
            lengthEncodedInput, persistInterval, namedPipeConnectTimeout, inputFileName,
            isInputFileNamedPipe, outputFileName, isOutputFileNamedPipe, restoreFileName,
            isRestoreFileNamedPipe, persistFileName, isPersistFileNamedPipe,
            isPersistInForeground, categorizationFieldName) == false) {
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

    if (jobId.empty()) {
        LOG_FATAL(<< "No job ID specified");
        return EXIT_FAILURE;
    }

    ml::model::CLimits limits{isPersistInForeground};
    if (!limitConfigFile.empty() && limits.init(limitConfigFile) == false) {
        LOG_FATAL(<< "ML limit config file '" << limitConfigFile << "' could not be loaded");
        return EXIT_FAILURE;
    }

    if (categorizationFieldName.empty()) {
        LOG_FATAL(<< "No categorization field name specified");
        return EXIT_FAILURE;
    }
    ml::api::CFieldConfig fieldConfig{categorizationFieldName};

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

    if (persistInterval >= 0 && persister == nullptr) {
        LOG_FATAL(<< "Periodic persistence cannot be enabled using the 'persistInterval' argument "
                     "unless a place to persist to has been specified using the 'persist' argument");
        return EXIT_FAILURE;
    }
    using TPersistenceManagerUPtr = std::unique_ptr<ml::api::CPersistenceManager>;
    const TPersistenceManagerUPtr persistenceManager{
        [persistInterval, isPersistInForeground, &persister]() -> TPersistenceManagerUPtr {
            if (persistInterval >= 0) {
                return std::make_unique<ml::api::CPersistenceManager>(
                    persistInterval, isPersistInForeground, *persister);
            }
            return nullptr;
        }()};

    using TInputParserUPtr = std::unique_ptr<ml::api::CInputParser>;
    const TInputParserUPtr inputParser{[lengthEncodedInput, &ioMgr, delimiter]() -> TInputParserUPtr {
        if (lengthEncodedInput) {
            return std::make_unique<ml::api::CLengthEncodedInputParser>(ioMgr.inputStream());
        }
        return std::make_unique<ml::api::CCsvInputParser>(ioMgr.inputStream(), delimiter);
    }()};

    ml::core::CJsonOutputStreamWrapper wrappedOutputStream{ioMgr.outputStream()};

    // The categorizer knows how to assign categories to records
    ml::api::CFieldDataCategorizer categorizer{jobId,
                                               fieldConfig,
                                               limits,
                                               timeField,
                                               timeFormat,
                                               nullptr,
                                               wrappedOutputStream,
                                               persistenceManager.get(),
                                               stopCategorizationOnWarnStatus};

    if (persistenceManager != nullptr) {
        persistenceManager->firstProcessorBackgroundPeriodicPersistFunc(std::bind(
            &ml::api::CFieldDataCategorizer::periodicPersistStateInBackground, &categorizer));

        persistenceManager->firstProcessorForegroundPeriodicPersistFunc(std::bind(
            &ml::api::CFieldDataCategorizer::periodicPersistStateInForeground, &categorizer));
    }

    // The skeleton avoids the need to duplicate a lot of boilerplate code
    ml::api::CCmdSkeleton skeleton{restoreSearcher.get(), persister.get(),
                                   *inputParser, categorizer};
    if (skeleton.ioLoop() == false) {
        LOG_FATAL(<< "ML categorization job failed");
        return EXIT_FAILURE;
    }

    // This message makes it easier to spot process crashes in a log file - if
    // this isn't present in the log for a given PID and there's no other log
    // message indicating early exit then the process has probably core dumped
    LOG_DEBUG(<< "ML categorization job exiting");

    return EXIT_SUCCESS;
}
