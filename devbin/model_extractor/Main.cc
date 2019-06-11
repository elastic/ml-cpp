/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CSingleJsonFileDataAdder.h" // XXX A copy of CMultiFileDataAdder from the test lib but renamed to better match its use here. Include directly from 'test'?

#include <api/CAnomalyJob.h>
#include <api/CFieldConfig.h>
#include <api/CSingleStreamSearcher.h>
#include <api/CStateRestoreStreamFilter.h>

#include <core/CDataSearcher.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>
#include <core/CStateDecompressor.h>
#include <core/CStringUtils.h>
#include <core/CoreTypes.h>
#include <core/COsFileFuncs.h>

#include <fstream>
#include <ios>
#include <iostream>
#include <memory>

#include <stdlib.h>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/filesystem.hpp>

// We use short field names to reduce the state size
namespace {


}

using namespace ml;

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Utility to extract model state from a file created by 'autodetect --persist arg' "
                "to an output file in JSON format." << std::endl;
        std::cerr << "Usage: " << argv[0] << " --input <file> --output <file>" << std::endl;
        exit(EXIT_FAILURE);
    }

    // TODO Everything...
    std::string inputFileName(argv[2]);
    std::string outputFileName(argv[4]);

    std::cout << "Input file = " << inputFileName << std::endl;
    std::cout << "Output file = " << outputFileName << std::endl;


    // XXX For now just restore from file. Support for other data sources such as named pipe at a later date?
    std::ifstream restoreStream(inputFileName.c_str());
    if (!restoreStream.is_open()) {
        std::cerr << "Failed to open " << inputFileName << " for reading." << std::endl;
        exit(EXIT_FAILURE);
    }

    auto filterStream = std::make_shared<boost::iostreams::filtering_istream>();
    filterStream->push(ml::api::CStateRestoreStreamFilter());
    filterStream->push(restoreStream);

    // create a job with basic config sufficient enough to restore state.
    static const ml::core_t::TTime BUCKET_SIZE(600);
    static const std::string JOB_ID("job");

    ml::model::CLimits limits;
    ml::api::CFieldConfig fieldConfig;

    if (!fieldConfig.initFromFile(ml::core::COsFileFuncs::NULL_FILENAME)) {
        std::cerr << "Failed to initialize field configuration." << std::endl;
        exit(EXIT_FAILURE);
    }
    const int latencyBuckets{0};

    ml::model::CAnomalyDetectorModelConfig modelConfig =
        ml::model::CAnomalyDetectorModelConfig::defaultConfig(
            BUCKET_SIZE, ml::model_t::E_None, "", BUCKET_SIZE * latencyBuckets, false);

    // dummy output stream to satisfy CAnomalyJob ctor requirements
    std::ofstream outputStrm(ml::core::COsFileFuncs::NULL_FILENAME);
    if(!outputStrm.is_open()) {
        LOG_ERROR(<< "Failed to open output stream.");
    }
    ml::core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

    ml::api::CAnomalyJob restoredJob(JOB_ID, limits, fieldConfig, modelConfig, wrappedOutputStream, [](ml::api::CModelSnapshotJsonWriter::SModelSnapshotReport){});

    ml::core_t::TTime completeToTime(0);
    ml::api::CSingleStreamSearcher fileSearcher(filterStream);
    if (!restoredJob.restoreState(fileSearcher, completeToTime)) {
        LOG_ERROR(<< "Failed to restore state from file \"" << inputFileName << "\"");
        exit(EXIT_FAILURE);
    }
    assert(completeToTime > 0);
    std::size_t numDocsInStateFile{filterStream->component<ml::api::CStateRestoreStreamFilter>(0)->getDocCount()};
    LOG_DEBUG( << "Restored " << numDocsInStateFile << " documents from persisted state. Complete to time " << completeToTime << std::endl);


    boost::filesystem::path outputPath(outputFileName);

    try {
        boost::filesystem::remove_all(outputPath);
    } catch (const boost::filesystem::filesystem_error &fsx) {
        LOG_ERROR(<< fsx.what());
        exit(EXIT_FAILURE);
    }

    ml::modelextractor::CSingleJsonFileDataAdder persister(outputFileName);

    // Attempt to persist state to a plain JSON formatted file
    if (restoredJob.persistResidualModelsState(persister) == false) {
        LOG_FATAL(<< "Failed to persist state as JSON");
        exit(EXIT_FAILURE);
    }

    exit(EXIT_SUCCESS);
}
