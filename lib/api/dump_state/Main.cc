/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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
#include <core/COsFileFuncs.h>
#include <core/CRegex.h>
#include <core/CoreTypes.h>

#include <ver/CBuildInfo.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CLimits.h>

#include <api/CAnomalyJob.h>
#include <api/CCsvInputParser.h>
#include <api/CFieldConfig.h>
#include <api/CFieldDataCategorizer.h>
#include <api/CJsonOutputWriter.h>
#include <api/CModelSnapshotJsonWriter.h>
#include <api/CNdJsonInputParser.h>
#include <api/CSingleStreamDataAdder.h>
#include <api/CSingleStreamSearcher.h>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

static std::string persistedNormalizerState;
static std::vector<std::string> persistedStateFiles;

static const std::string TEST_FILES_PATH{"../unittest/testfiles/"};

bool parseOptions(int argc, const char* const* argv, std::string& outputDir) {
    try {
        boost::program_options::options_description desc(
            "Utility for creating ML model state files for use in BWC tests");
        desc.add_options()("help", "Display this information and exit")(
            "outputDir", boost::program_options::value<std::string>(),
            "Optional directory to write state files to");

        boost::program_options::variables_map vm;
        boost::program_options::parsed_options parsed =
            boost::program_options::command_line_parser(argc, argv)
                .options(desc)
                .run();
        boost::program_options::store(parsed, vm);
        if (vm.count("help") > 0) {
            std::cerr << desc << std::endl;
            return false;
        }
        if (vm.count("outputDir") > 0) {
            outputDir = vm["outputDir"].as<std::string>();
            if (outputDir.empty()) {
                std::cerr << "Error processing command line: outputDir is an empty string"
                          << std::endl;
                return false;
            }
            if (outputDir.back() != '/') {
                outputDir.push_back('/');
            }
        }

        return true;
    } catch (std::exception& e) {
        std::cerr << "Error processing command line: " << e.what() << std::endl;
        return false;
    }
}

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

void reportPersistComplete(ml::api::CModelSnapshotJsonWriter::SModelSnapshotReport modelSnapshotReport) {
    LOG_INFO(<< "Persist complete with description: " << modelSnapshotReport.s_Description);
    persistedNormalizerState = modelSnapshotReport.s_NormalizerState;
}

bool writeNormalizerState(const std::string& outputFileName) {
    std::ofstream out(outputFileName);
    if (!out.is_open()) {
        LOG_ERROR(<< "Failed to open normalizer state output file " << outputFileName);
        return false;
    }

    out << persistedNormalizerState;
    out.close();

    persistedStateFiles.push_back(outputFileName);
    return true;
}

bool persistCategorizerStateToFile(const std::string& outputFileName) {
    ml::model::CLimits limits(true);
    ml::api::CFieldConfig config("count", "mlcategory");

    std::ofstream outStream(ml::core::COsFileFuncs::NULL_FILENAME);
    ml::core::CJsonOutputStreamWrapper wrappendOutStream(outStream);
    ml::api::CJsonOutputWriter writer("job", wrappendOutStream);

    ml::api::CFieldDataCategorizer categorizer("job", config, limits, writer, writer, nullptr);

    ml::api::CFieldDataCategorizer::TStrStrUMap dataRowFields;
    dataRowFields["_raw"] = "thing";
    dataRowFields["two"] = "other";

    categorizer.handleRecord(dataRowFields);

    // Persist the categorizer state to file
    {
        std::ofstream* out = nullptr;
        ml::api::CSingleStreamDataAdder::TOStreamP ptr(out = new std::ofstream(outputFileName));
        if (!out->is_open()) {
            LOG_ERROR(<< "Failed to open categorizer state output file " << outputFileName);
            return false;
        }

        ml::api::CSingleStreamDataAdder persister(ptr);
        if (!categorizer.persistState(persister, "State persisted due to job close at ")) {
            LOG_ERROR(<< "Error persisting state to " << outputFileName);
            return false;
        }
    }

    persistedStateFiles.push_back(outputFileName);
    return true;
}

bool persistAnomalyDetectorStateToFile(const std::string& configFileName,
                                       const std::string& inputFilename,
                                       const std::string& outputFileName,
                                       int latencyBuckets,
                                       const std::string& timeFormat = std::string()) {
    // Open the input and output files
    std::ifstream inputStrm(inputFilename);
    if (!inputStrm.is_open()) {
        LOG_ERROR(<< "Cannot open input file " << inputFilename);
        return false;
    }
    std::ofstream outputStrm(ml::core::COsFileFuncs::NULL_FILENAME);

    ml::core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

    ml::model::CLimits limits(true);
    ml::api::CFieldConfig fieldConfig;
    if (!fieldConfig.initFromFile(configFileName)) {
        LOG_ERROR(<< "Failed to init field config from " << configFileName);
        return false;
    }

    ml::core_t::TTime bucketSize(3600);
    std::string jobId("foo");
    ml::model::CAnomalyDetectorModelConfig modelConfig =
        ml::model::CAnomalyDetectorModelConfig::defaultConfig(
            bucketSize, ml::model_t::E_None, "", bucketSize * latencyBuckets, false);

    ml::api::CAnomalyJob origJob(
        jobId, limits, fieldConfig, modelConfig, wrappedOutputStream, nullptr,
        std::bind(&reportPersistComplete, std::placeholders::_1), -1, "time", timeFormat);

    using TInputParserUPtr = std::unique_ptr<ml::api::CInputParser>;
    const TInputParserUPtr parser{[&inputFilename, &inputStrm]() -> TInputParserUPtr {
        if (inputFilename.rfind(".csv") == inputFilename.length() - 4) {
            return std::make_unique<ml::api::CCsvInputParser>(inputStrm);
        }
        return std::make_unique<ml::api::CNdJsonInputParser>(inputStrm);
    }()};

    if (!parser->readStreamIntoMaps(std::bind(&ml::api::CAnomalyJob::handleRecord,
                                              &origJob, std::placeholders::_1))) {
        LOG_ERROR(<< "Failed to processs input");
        return false;
    }

    // Persist the job state to file
    {
        std::ofstream* out = nullptr;
        ml::api::CSingleStreamDataAdder::TOStreamP ptr(out = new std::ofstream(outputFileName));
        if (!out->is_open()) {
            LOG_ERROR(<< "Failed to open state output file " << outputFileName);
            return false;
        }

        ml::api::CSingleStreamDataAdder persister(ptr);
        if (!origJob.persistState(persister, "State persisted due to job close at ")) {
            LOG_ERROR(<< "Error persisting state to " << outputFileName);
            return false;
        }
    }

    persistedStateFiles.push_back(outputFileName);

    return true;
}

bool persistByDetector(const std::string& stateFilesPath) {
    return persistAnomalyDetectorStateToFile(
        TEST_FILES_PATH + "new_mlfields.conf", TEST_FILES_PATH + "big_ascending.txt",
        stateFilesPath + "by_detector_state.json", 0, "%d/%b/%Y:%T %z");
}

bool persistOverDetector(const std::string& stateFilesPath) {
    return persistAnomalyDetectorStateToFile(
        TEST_FILES_PATH + "new_mlfields_over.conf", TEST_FILES_PATH + "big_ascending.txt",
        stateFilesPath + "over_detector_state.json", 0, "%d/%b/%Y:%T %z");
}

bool persistPartitionDetector(const std::string& stateFilesPath) {
    return persistAnomalyDetectorStateToFile(
        TEST_FILES_PATH + "new_mlfields_partition.conf", TEST_FILES_PATH + "big_ascending.txt",
        stateFilesPath + "partition_detector_state.json", 0, "%d/%b/%Y:%T %z");
}

bool persistDcDetector(const std::string& stateFilesPath) {
    return persistAnomalyDetectorStateToFile(
        TEST_FILES_PATH + "new_persist_dc.conf", TEST_FILES_PATH + "files_users_programs.csv",
        stateFilesPath + "dc_detector_state.json", 5);
}

bool persistCountDetector(const std::string& stateFilesPath) {
    return persistAnomalyDetectorStateToFile(
        TEST_FILES_PATH + "new_persist_count.conf", TEST_FILES_PATH + "files_users_programs.csv",
        stateFilesPath + "count_detector_state.json", 5);
}

int main(int argc, char** argv) {

    std::string stateFilesPath;
    if (parseOptions(argc, argv, stateFilesPath) == false) {
        return EXIT_FAILURE;
    }

    ml::core::CLogger::instance().setLoggingLevel(ml::core::CLogger::E_Info);

    std::string version = versionNumber();
    if (version.empty()) {
        LOG_ERROR(<< "Cannot get version number");
        return EXIT_FAILURE;
    }

    if (stateFilesPath.empty()) {
        // The outputDir argument wasn't set, use the default path
        stateFilesPath = TEST_FILES_PATH + "state/" + version + "/";
    }

    boost::system::error_code errorCode;
    boost::filesystem::create_directories(stateFilesPath, errorCode);
    if (errorCode) {
        LOG_ERROR(<< "Failed to create directory " << stateFilesPath
                  << ", error: " << errorCode.message());
        return EXIT_FAILURE;
    }

    LOG_INFO(<< "Saving model state for version: " << version
             << " to directory: " << stateFilesPath);

    bool persisted = persistByDetector(stateFilesPath);
    if (!persisted) {
        LOG_ERROR(<< "Failed to persist state for by detector");
        return EXIT_FAILURE;
    }

    if (persistedNormalizerState.empty()) {
        LOG_ERROR(<< "Normalizer state not persisted");
        return EXIT_FAILURE;
    }
    if (!writeNormalizerState(stateFilesPath + "normalizer_state.json")) {
        LOG_ERROR(<< "Error writing normalizer state file");
        return EXIT_FAILURE;
    }

    persisted = persistOverDetector(stateFilesPath);
    if (!persisted) {
        LOG_ERROR(<< "Failed to persist state for over detector");
        return EXIT_FAILURE;
    }

    persisted = persistPartitionDetector(stateFilesPath);
    if (!persisted) {
        LOG_ERROR(<< "Failed to persist state for partition detector");
        return EXIT_FAILURE;
    }

    persisted = persistDcDetector(stateFilesPath);
    if (!persisted) {
        LOG_ERROR(<< "Failed to persist state for DC detector");
        return EXIT_FAILURE;
    }

    persisted = persistCountDetector(stateFilesPath);
    if (!persisted) {
        LOG_ERROR(<< "Failed to persist state for count detector");
        return EXIT_FAILURE;
    }

    persisted = persistCategorizerStateToFile(stateFilesPath + "categorizer_state.json");
    if (!persisted) {
        LOG_ERROR(<< "Failed to persist categorizer state");
        return EXIT_FAILURE;
    }

    LOG_INFO(<< "Written state files:");
    for (const auto& stateFile : persistedStateFiles) {
        LOG_INFO(<< "\t" << stateFile);
    }

    return EXIT_SUCCESS;
}
