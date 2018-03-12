/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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
//! Creates model state files for Anomaly Detectors, Categorizers and the
//! Normalizer quantiles. The resulting files can then be used in backwards
//! compatibility tests.
//!
//! DESCRIPTION:\n
//! The state files produced by this program are written to the
//! ../unittest/testfiles/state/$VERSION directory and can be used
//! by the CRestorePreviousStateTest. Some detectors are configured
//! with non-zero latency buckets the same latency buckets value
//! should be used in CRestorePreviousStateTest.
//!
//!
//! IMPLEMENTATION DECISIONS:\n
//!
//!
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CLogger.h>
#include <core/CoreTypes.h>
#include <core/COsFileFuncs.h>
#include <core/CRegex.h>

#include <ver/CBuildInfo.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CLimits.h>

#include <api/CAnomalyJob.h>
#include <api/CCsvInputParser.h>
#include <api/CFieldConfig.h>
#include <api/CFieldDataTyper.h>
#include <api/CLineifiedJsonInputParser.h>
#include <api/CSingleStreamDataAdder.h>
#include <api/CSingleStreamSearcher.h>

#include <boost/bind.hpp>
#include <boost/scoped_ptr.hpp>

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>


static std::string persistedNormalizerState;
static std::vector<std::string> persistedStateFiles;

std::string versionNumber() {
    ml::core::CRegex regex;
    regex.init("\\d\\.\\d\\.\\d");
    std::string longVersion = ml::ver::CBuildInfo::versionNumber();
    std::size_t pos;
    std::string version;
    if (regex.search(longVersion, pos)) {
        version = longVersion.substr(pos, 5);
    }
    return version;
}

void reportPersistComplete(ml::core_t::TTime /*snapshotTimestamp*/,
                           const std::string &description,
                           const std::string &/*snapshotIdIn*/,
                           size_t /*numDocsIn*/,
                           const ml::model::CResourceMonitor::SResults &/*results*/,
                           const std::string &normalizerState) {
    LOG_INFO("Persist complete with description: " << description);
    persistedNormalizerState = normalizerState;
}

bool writeNormalizerState(const std::string &outputFileName) {
    std::ofstream out(outputFileName);
    if (!out.is_open()) {
        LOG_ERROR("Failed to open normalizer state output file " << outputFileName);
        return false;
    }

    out << persistedNormalizerState;
    out.close();

    persistedStateFiles.push_back(outputFileName);
    return true;
}

bool persistCategorizerStateToFile(const std::string &outputFileName) {
    ml::model::CLimits limits;
    ml::api::CFieldConfig config("count", "mlcategory");

    std::ofstream outStream(ml::core::COsFileFuncs::NULL_FILENAME);
    ml::core::CJsonOutputStreamWrapper wrappendOutStream(outStream);
    ml::api::CJsonOutputWriter writer("job", wrappendOutStream);

    ml::api::CFieldDataTyper typer("job", config, limits, writer, writer);

    ml::api::CFieldDataTyper::TStrStrUMap dataRowFields;
    dataRowFields["_raw"] = "thing";
    dataRowFields["two"] = "other";

    typer.handleRecord(dataRowFields);

    // Persist the categorizer state to file
    {
        std::ofstream *out = nullptr;
        ml::api::CSingleStreamDataAdder::TOStreamP ptr(out = new std::ofstream(outputFileName));
        if (!out->is_open()) {
            LOG_ERROR("Failed to open categorizer state output file " << outputFileName);
            return false;
        }

        ml::api::CSingleStreamDataAdder persister(ptr);
        if (!typer.persistState(persister)) {
            LOG_ERROR("Error persisting state to " << outputFileName);
            return false;
        }
    }

    persistedStateFiles.push_back(outputFileName);
    return true;
}

bool persistAnomalyDetectorStateToFile(const std::string &configFileName,
                                       const std::string &inputFilename,
                                       const std::string &outputFileName,
                                       int latencyBuckets,
                                       const std::string &timeFormat = std::string()) {
    // Open the input and output files
    std::ifstream inputStrm(inputFilename);
    if (!inputStrm.is_open()) {
        LOG_ERROR("Cannot open input file " << inputFilename);
        return false;
    }
    std::ofstream outputStrm(ml::core::COsFileFuncs::NULL_FILENAME);

    ml::core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

    ml::model::CLimits limits;
    ml::api::CFieldConfig fieldConfig;
    if (!fieldConfig.initFromFile(configFileName)) {
        LOG_ERROR("Failed to init field config from " << configFileName);
        return false;
    }

    ml::core_t::TTime bucketSize(3600);
    std::string jobId("foo");
    ml::model::CAnomalyDetectorModelConfig modelConfig =
        ml::model::CAnomalyDetectorModelConfig::defaultConfig(bucketSize,
                                                              ml::model_t::E_None,
                                                              "",
                                                              bucketSize * latencyBuckets,
                                                              0,
                                                              false,
                                                              "");

    ml::api::CAnomalyJob origJob(jobId,
                                 limits,
                                 fieldConfig,
                                 modelConfig,
                                 wrappedOutputStream,
                                 boost::bind(&reportPersistComplete,
                                             _1,
                                             _2,
                                             _3,
                                             _4,
                                             _5,
                                             _6),
                                 nullptr,
                                 -1,
                                 "time",
                                 timeFormat);

    using TScopedInputParserP = boost::scoped_ptr<ml::api::CInputParser>;
    TScopedInputParserP parser;
    if (inputFilename.rfind(".csv") == inputFilename.length() - 4) {
        parser.reset(new ml::api::CCsvInputParser(inputStrm));
    } else {
        parser.reset(new ml::api::CLineifiedJsonInputParser(inputStrm));
    }

    if (!parser->readStream(boost::bind(&ml::api::CAnomalyJob::handleRecord,
                                        &origJob,
                                        _1))) {
        LOG_ERROR("Failed to processs input");
        return false;
    }

    // Persist the job state to file
    {
        std::ofstream *out = nullptr;
        ml::api::CSingleStreamDataAdder::TOStreamP ptr(out = new std::ofstream(outputFileName));
        if (!out->is_open()) {
            LOG_ERROR("Failed to open state output file " << outputFileName);
            return false;
        }

        ml::api::CSingleStreamDataAdder persister(ptr);
        if (!origJob.persistState(persister)) {
            LOG_ERROR("Error persisting state to " << outputFileName);
            return false;
        }
    }

    persistedStateFiles.push_back(outputFileName);

    return true;
}

bool persistByDetector(const std::string &version) {
    return persistAnomalyDetectorStateToFile("../unittest/testfiles/new_mlfields.conf",
                                             "../unittest/testfiles/big_ascending.txt",
                                             "../unittest/testfiles/state/" + version + "/by_detector_state.json",
                                             0,
                                             "%d/%b/%Y:%T %z");
}

bool persistOverDetector(const std::string &version) {
    return persistAnomalyDetectorStateToFile("../unittest/testfiles/new_mlfields_over.conf",
                                             "../unittest/testfiles/big_ascending.txt",
                                             "../unittest/testfiles/state/" + version + "/over_detector_state.json",
                                             0,
                                             "%d/%b/%Y:%T %z");
}

bool persistPartitionDetector(const std::string &version) {
    return persistAnomalyDetectorStateToFile("../unittest/testfiles/new_mlfields_partition.conf",
                                             "../unittest/testfiles/big_ascending.txt",
                                             "../unittest/testfiles/state/" + version + "/partition_detector_state.json",
                                             0,
                                             "%d/%b/%Y:%T %z");
}

bool persistDcDetector(const std::string &version) {
    return persistAnomalyDetectorStateToFile("../unittest/testfiles/new_persist_dc.conf",
                                             "../unittest/testfiles/files_users_programs.csv",
                                             "../unittest/testfiles/state/" + version + "/dc_detector_state.json",
                                             5);
}

bool persistCountDetector(const std::string &version) {
    return persistAnomalyDetectorStateToFile("../unittest/testfiles/new_persist_count.conf",
                                             "../unittest/testfiles/files_users_programs.csv",
                                             "../unittest/testfiles/state/" + version + "/count_detector_state.json",
                                             5);
}

int main(int /*argc*/, char **/*argv*/) {
    ml::core::CLogger::instance().setLoggingLevel(ml::core::CLogger::E_Info);

    std::string version = versionNumber();
    if (version.empty()) {
        LOG_ERROR("Cannot get version number");
        return EXIT_FAILURE;
    }
    LOG_INFO("Saving model state for version: " << version);

    bool persisted = persistByDetector(version);
    if (!persisted) {
        LOG_ERROR("Failed to persist state for by detector");
        return EXIT_FAILURE;
    }

    if (persistedNormalizerState.empty()) {
        LOG_ERROR("Normalizer state not persisted");
        return EXIT_FAILURE;
    }
    if (!writeNormalizerState("../unittest/testfiles/state/" + version + "/normalizer_state.json")) {
        LOG_ERROR("Error writing normalizer state file");
        return EXIT_FAILURE;
    }

    persisted = persistOverDetector(version);
    if (!persisted) {
        LOG_ERROR("Failed to persist state for over detector");
        return EXIT_FAILURE;
    }

    persisted = persistPartitionDetector(version);
    if (!persisted) {
        LOG_ERROR("Failed to persist state for partition detector");
        return EXIT_FAILURE;
    }

    persisted = persistDcDetector(version);
    if (!persisted) {
        LOG_ERROR("Failed to persist state for DC detector");
        return EXIT_FAILURE;
    }

    persisted = persistCountDetector(version);
    if (!persisted) {
        LOG_ERROR("Failed to persist state for count detector");
        return EXIT_FAILURE;
    }

    persisted = persistCategorizerStateToFile("../unittest/testfiles/state/" + version + "/categorizer_state.json");
    if (!persisted) {
        LOG_ERROR("Failed to persist categorizer state");
        return EXIT_FAILURE;
    }

    LOG_INFO("Written state files:");
    for (const auto &stateFile : persistedStateFiles) {
        LOG_INFO("\t" << stateFile)
    }

    return EXIT_SUCCESS;
}

