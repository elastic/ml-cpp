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
#include <core/CDataAdder.h>
#include <core/CDataSearcher.h>
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CLogger.h>
#include <core/CProcessPriority.h>
#include <core/CoreTypes.h>

#include <ver/CBuildInfo.h>

#include <model/CLimits.h>

#include <api/CBackgroundPersister.h>
#include <api/CCmdSkeleton.h>
#include <api/CCsvInputParser.h>
#include <api/CFieldConfig.h>
#include <api/CFieldDataTyper.h>
#include <api/CIoManager.h>
#include <api/CJsonOutputWriter.h>
#include <api/CLengthEncodedInputParser.h>
#include <api/CNullOutput.h>
#include <api/COutputChainer.h>
#include <api/CSingleStreamDataAdder.h>
#include <api/CSingleStreamSearcher.h>
#include <api/CStateRestoreStreamFilter.h>

#include "CCmdLineParser.h"

#include <boost/bind.hpp>

#include <memory>
#include <string>

#include <stdlib.h>

int main(int argc, char** argv) {
    // Read command line options
    std::string limitConfigFile;
    std::string jobId;
    std::string logProperties;
    std::string logPipe;
    char delimiter('\t');
    bool lengthEncodedInput(false);
    ml::core_t::TTime persistInterval(-1);
    std::string inputFileName;
    bool isInputFileNamedPipe(false);
    std::string outputFileName;
    bool isOutputFileNamedPipe(false);
    std::string restoreFileName;
    bool isRestoreFileNamedPipe(false);
    std::string persistFileName;
    bool isPersistFileNamedPipe(false);
    std::string categorizationFieldName;
    if (ml::categorize::CCmdLineParser::parse(
            argc, argv, limitConfigFile, jobId, logProperties, logPipe, delimiter,
            lengthEncodedInput, persistInterval, inputFileName, isInputFileNamedPipe,
            outputFileName, isOutputFileNamedPipe, restoreFileName, isRestoreFileNamedPipe,
            persistFileName, isPersistFileNamedPipe, categorizationFieldName) == false) {
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

    if (categorizationFieldName.empty()) {
        LOG_FATAL(<< "No categorization field name specified");
        return EXIT_FAILURE;
    }
    ml::api::CFieldConfig fieldConfig(categorizationFieldName);

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
        LOG_FATAL(<< "Periodic persistence cannot be enabled using the "
                     "'persistInterval' argument "
                     "unless a place to persist to has been specified "
                     "using the 'persist' argument");
        return EXIT_FAILURE;
    }
    using TBackgroundPersisterCUPtr = const std::unique_ptr<ml::api::CBackgroundPersister>;
    TBackgroundPersisterCUPtr periodicPersister{[persistInterval, &persister]() {
        return new ml::api::CBackgroundPersister(persistInterval, *persister);
    }()};

    using TInputParserCUPtr = const std::unique_ptr<ml::api::CInputParser>;
    TInputParserCUPtr inputParser{[lengthEncodedInput, &ioMgr,
                                   delimiter]() -> ml::api::CInputParser* {
        if (lengthEncodedInput) {
            return new ml::api::CLengthEncodedInputParser(ioMgr.inputStream());
        }
        return new ml::api::CCsvInputParser(ioMgr.inputStream(), delimiter);
    }()};

    ml::core::CJsonOutputStreamWrapper wrappedOutputStream(ioMgr.outputStream());

    // All output we're interested in goes via the JSON output writer, so output
    // of the categorised input data can be dropped
    ml::api::CNullOutput nullOutput;

    // output writer for CFieldDataTyper and persistence callback
    ml::api::CJsonOutputWriter outputWriter(jobId, wrappedOutputStream);

    // The typer knows how to assign categories to records
    ml::api::CFieldDataTyper typer(jobId, fieldConfig, limits, nullOutput,
                                   outputWriter, periodicPersister.get());

    if (periodicPersister != nullptr) {
        periodicPersister->firstProcessorPeriodicPersistFunc(boost::bind(
            &ml::api::CFieldDataTyper::periodicPersistState, &typer, _1));
    }

    // The skeleton avoids the need to duplicate a lot of boilerplate code
    ml::api::CCmdSkeleton skeleton(restoreSearcher.get(), persister.get(),
                                   *inputParser, typer);
    bool ioLoopSucceeded(skeleton.ioLoop());

    // Unfortunately we cannot rely on destruction to finalise the output writer
    // as it must be finalised before the skeleton is destroyed, and C++
    // destruction order means the skeleton will be destroyed before the output
    // writer as it was constructed last.
    outputWriter.finalise();

    if (!ioLoopSucceeded) {
        LOG_FATAL(<< "Ml categorization job failed");
        return EXIT_FAILURE;
    }

    // This message makes it easier to spot process crashes in a log file - if
    // this isn't present in the log for a given PID and there's no other log
    // message indicating early exit then the process has probably core dumped
    LOG_DEBUG(<< "Ml categorization job exiting");

    return EXIT_SUCCESS;
}
