/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CDataAdder.h>
#include <core/CJsonOutputStreamWrapper.h>
#include <core/COsFileFuncs.h>
#include <core/CoreTypes.h>

#include <maths/CModelWeight.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CLimits.h>

#include <api/CAnomalyJob.h>
#include <api/CCsvInputParser.h>
#include <api/CFieldConfig.h>
#include <api/CJsonOutputWriter.h>
#include <api/CNdJsonInputParser.h>

#include <test/CMultiFileDataAdder.h>
#include <test/CMultiFileSearcher.h>
#include <test/CTestTmpDir.h>

#include <rapidjson/document.h>

#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>

#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

BOOST_AUTO_TEST_SUITE(CMultiFileDataAdderTest)

namespace {

using TStrVec = std::vector<std::string>;

void reportPersistComplete(ml::api::CModelSnapshotJsonWriter::SModelSnapshotReport modelSnapshotReport,
                           std::string& snapshotIdOut,
                           size_t& numDocsOut) {
    LOG_INFO(<< "Persist complete with description: " << modelSnapshotReport.s_Description);
    snapshotIdOut = modelSnapshotReport.s_SnapshotId;
    numDocsOut = modelSnapshotReport.s_NumDocs;
}

void detectorPersistHelper(const std::string& configFileName,
                           const std::string& inputFilename,
                           int latencyBuckets,
                           const std::string& timeFormat = std::string()) {
    // Start by creating a detector with non-trivial state
    static const ml::core_t::TTime BUCKET_SIZE(3600);
    static const std::string JOB_ID("job");

    // Open the input and output files
    std::ifstream inputStrm(inputFilename.c_str());
    BOOST_TEST_REQUIRE(inputStrm.is_open());

    std::ofstream outputStrm(ml::core::COsFileFuncs::NULL_FILENAME);
    BOOST_TEST_REQUIRE(outputStrm.is_open());
    ml::core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);

    ml::model::CLimits limits;
    ml::api::CFieldConfig fieldConfig;
    BOOST_TEST_REQUIRE(fieldConfig.initFromFile(configFileName));

    ml::model::CAnomalyDetectorModelConfig modelConfig =
        ml::model::CAnomalyDetectorModelConfig::defaultConfig(
            BUCKET_SIZE, ml::model_t::E_None, "", BUCKET_SIZE * latencyBuckets, false);

    std::string origSnapshotId;
    std::size_t numOrigDocs(0);
    ml::api::CAnomalyJob origJob(
        JOB_ID, limits, fieldConfig, modelConfig, wrappedOutputStream, nullptr,
        std::bind(&reportPersistComplete, std::placeholders::_1,
                  std::ref(origSnapshotId), std::ref(numOrigDocs)),
        -1, "time", timeFormat);

    using TInputParserUPtr = std::unique_ptr<ml::api::CInputParser>;
    const TInputParserUPtr parser{[&inputFilename, &inputStrm]() -> TInputParserUPtr {
        if (inputFilename.rfind(".csv") == inputFilename.length() - 4) {
            return std::make_unique<ml::api::CCsvInputParser>(inputStrm);
        }
        return std::make_unique<ml::api::CNdJsonInputParser>(inputStrm);
    }()};

    BOOST_TEST_REQUIRE(parser->readStreamIntoMaps(std::bind(
        &ml::api::CAnomalyJob::handleRecord, &origJob, std::placeholders::_1)));

    // Persist the detector state to file(s)

    std::string baseOrigOutputFilename(ml::test::CTestTmpDir::tmpDir() + "/orig");
    {
        // Clean up any leftovers of previous failures
        boost::filesystem::path origDir(baseOrigOutputFilename);
        BOOST_REQUIRE_NO_THROW(boost::filesystem::remove_all(origDir));

        ml::test::CMultiFileDataAdder persister(baseOrigOutputFilename);
        BOOST_TEST_REQUIRE(origJob.persistState(persister, ""));
    }

    std::string origBaseDocId(JOB_ID + '_' + ml::api::CAnomalyJob::STATE_TYPE +
                              '_' + origSnapshotId);

    std::string temp;
    TStrVec origFileContents(numOrigDocs);
    for (size_t index = 0; index < numOrigDocs; ++index) {
        std::string expectedOrigFilename(baseOrigOutputFilename);
        expectedOrigFilename += "/_";
        expectedOrigFilename += ml::api::CAnomalyJob::ML_STATE_INDEX;
        expectedOrigFilename += '/';
        expectedOrigFilename +=
            ml::core::CDataAdder::makeCurrentDocId(origBaseDocId, 1 + index);
        expectedOrigFilename += ml::test::CMultiFileDataAdder::JSON_FILE_EXT;
        LOG_DEBUG(<< "Trying to open file: " << expectedOrigFilename);
        std::ifstream origFile(expectedOrigFilename.c_str());
        BOOST_TEST_REQUIRE(origFile.is_open());
        std::string json((std::istreambuf_iterator<char>(origFile)),
                         std::istreambuf_iterator<char>());
        origFileContents[index] = json;

        // Ensure that the JSON is valid, by parsing string using Rapidjson
        rapidjson::Document document;
        BOOST_TEST_REQUIRE(!document.Parse<0>(origFileContents[index].c_str()).HasParseError());
        BOOST_TEST_REQUIRE(document.IsObject());
    }

    // Now restore the state into a different detector

    std::string restoredSnapshotId;
    std::size_t numRestoredDocs(0);
    ml::api::CAnomalyJob restoredJob(
        JOB_ID, limits, fieldConfig, modelConfig, wrappedOutputStream, nullptr,
        std::bind(&reportPersistComplete, std::placeholders::_1,
                  std::ref(restoredSnapshotId), std::ref(numRestoredDocs)));

    {
        ml::core_t::TTime completeToTime(0);

        ml::test::CMultiFileSearcher retriever(baseOrigOutputFilename, origBaseDocId);
        BOOST_TEST_REQUIRE(restoredJob.restoreState(retriever, completeToTime));
        BOOST_TEST_REQUIRE(completeToTime > 0);
    }

    // Finally, persist the new detector state to a file

    std::string baseRestoredOutputFilename(ml::test::CTestTmpDir::tmpDir() + "/restored");
    {
        // Clean up any leftovers of previous failures
        boost::filesystem::path restoredDir(baseRestoredOutputFilename);
        BOOST_REQUIRE_NO_THROW(boost::filesystem::remove_all(restoredDir));

        ml::test::CMultiFileDataAdder persister(baseRestoredOutputFilename);
        BOOST_TEST_REQUIRE(restoredJob.persistState(persister, ""));
    }

    std::string restoredBaseDocId(JOB_ID + '_' + ml::api::CAnomalyJob::STATE_TYPE +
                                  '_' + restoredSnapshotId);

    for (size_t index = 0; index < numRestoredDocs; ++index) {
        std::string expectedRestoredFilename(baseRestoredOutputFilename);
        expectedRestoredFilename += "/_";
        expectedRestoredFilename += ml::api::CAnomalyJob::ML_STATE_INDEX;
        expectedRestoredFilename += '/';
        expectedRestoredFilename +=
            ml::core::CDataAdder::makeCurrentDocId(restoredBaseDocId, 1 + index);
        expectedRestoredFilename += ml::test::CMultiFileDataAdder::JSON_FILE_EXT;
        std::ifstream restoredFile(expectedRestoredFilename.c_str());
        BOOST_TEST_REQUIRE(restoredFile.is_open());
        std::string json((std::istreambuf_iterator<char>(restoredFile)),
                         std::istreambuf_iterator<char>());

        BOOST_REQUIRE_EQUAL(origFileContents[index], json);
    }

    // Clean up
    boost::filesystem::path origDir(baseOrigOutputFilename);
    BOOST_REQUIRE_NO_THROW(boost::filesystem::remove_all(origDir));
    boost::filesystem::path restoredDir(baseRestoredOutputFilename);
    BOOST_REQUIRE_NO_THROW(boost::filesystem::remove_all(restoredDir));
}
}

BOOST_AUTO_TEST_CASE(testSimpleWrite) {
    static const std::string EVENT("Hello Event");
    static const std::string SUMMARY_EVENT("Hello Summary Event");

    static const std::string EXTENSION(".txt");
    std::string baseOutputFilename(ml::test::CTestTmpDir::tmpDir() + "/filepersister");

    std::string expectedFilename(baseOutputFilename);
    expectedFilename += "/_hello/1";
    expectedFilename += EXTENSION;

    {
        // Clean up any leftovers of previous failures
        boost::filesystem::path workDir(baseOutputFilename);
        BOOST_REQUIRE_NO_THROW(boost::filesystem::remove_all(workDir));

        ml::test::CMultiFileDataAdder persister(baseOutputFilename, EXTENSION);
        ml::core::CDataAdder::TOStreamP strm = persister.addStreamed("hello", "1");
        BOOST_TEST_REQUIRE(strm);
        (*strm) << EVENT;
        BOOST_TEST_REQUIRE(persister.streamComplete(strm, true));
    }

    {
        std::ifstream persistedFile(expectedFilename.c_str());

        BOOST_TEST_REQUIRE(persistedFile.is_open());
        std::string line;
        std::getline(persistedFile, line);
        BOOST_REQUIRE_EQUAL(EVENT, line);
    }

    BOOST_REQUIRE_EQUAL(0, ::remove(expectedFilename.c_str()));

    expectedFilename = baseOutputFilename;
    expectedFilename += "/_stash/1";
    expectedFilename += EXTENSION;

    {
        ml::test::CMultiFileDataAdder persister(baseOutputFilename, EXTENSION);
        ml::core::CDataAdder::TOStreamP strm = persister.addStreamed("stash", "1");
        BOOST_TEST_REQUIRE(strm);
        (*strm) << SUMMARY_EVENT;
        BOOST_TEST_REQUIRE(persister.streamComplete(strm, true));
    }

    {
        std::ifstream persistedFile(expectedFilename.c_str());

        BOOST_TEST_REQUIRE(persistedFile.is_open());
        std::string line;
        std::getline(persistedFile, line);
        BOOST_REQUIRE_EQUAL(SUMMARY_EVENT, line);
    }

    // Clean up
    boost::filesystem::path workDir(baseOutputFilename);
    BOOST_REQUIRE_NO_THROW(boost::filesystem::remove_all(workDir));
}

BOOST_AUTO_TEST_CASE(testDetectorPersistBy) {
    detectorPersistHelper("testfiles/new_mlfields.conf",
                          "testfiles/big_ascending.txt", 0, "%d/%b/%Y:%T %z");
}

BOOST_AUTO_TEST_CASE(testDetectorPersistOver) {
    detectorPersistHelper("testfiles/new_mlfields_over.conf",
                          "testfiles/big_ascending.txt", 0, "%d/%b/%Y:%T %z");
}

BOOST_AUTO_TEST_CASE(testDetectorPersistPartition) {
    detectorPersistHelper("testfiles/new_mlfields_partition.conf",
                          "testfiles/big_ascending.txt", 0, "%d/%b/%Y:%T %z");
}

BOOST_AUTO_TEST_CASE(testDetectorPersistDc) {
    detectorPersistHelper("testfiles/new_persist_dc.conf",
                          "testfiles/files_users_programs.csv", 5);
}

BOOST_AUTO_TEST_CASE(testDetectorPersistCount) {
    detectorPersistHelper("testfiles/new_persist_count.conf",
                          "testfiles/files_users_programs.csv", 5);
}

BOOST_AUTO_TEST_SUITE_END()
