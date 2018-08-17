/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CSingleStreamDataAdderTest.h"

#include <core/CJsonOutputStreamWrapper.h>
#include <core/COsFileFuncs.h>
#include <core/CStringUtils.h>
#include <core/CoreTypes.h>

#include <maths/CModelWeight.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CLimits.h>

#include <api/CAnomalyJob.h>
#include <api/CCsvInputParser.h>
#include <api/CFieldConfig.h>
#include <api/CFieldDataTyper.h>
#include <api/CJsonOutputWriter.h>
#include <api/CLineifiedJsonInputParser.h>
#include <api/COutputChainer.h>
#include <api/CSingleStreamDataAdder.h>
#include <api/CSingleStreamSearcher.h>
#include <api/CStateRestoreStreamFilter.h>

#include <boost/bind.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/ref.hpp>

#include <fstream>
#include <memory>
#include <sstream>
#include <string>

namespace {

void reportPersistComplete(ml::api::CModelSnapshotJsonWriter::SModelSnapshotReport modelSnapshotReport,
                           std::string& snapshotIdOut,
                           size_t& numDocsOut) {
    LOG_INFO(<< "Persist complete with description: " << modelSnapshotReport.s_Description);
    snapshotIdOut = modelSnapshotReport.s_SnapshotId;
    numDocsOut = modelSnapshotReport.s_NumDocs;
}
}

CppUnit::Test* CSingleStreamDataAdderTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CSingleStreamDataAdderTest");
    suiteOfTests->addTest(new CppUnit::TestCaller<CSingleStreamDataAdderTest>(
        "CSingleStreamDataAdderTest::testDetectorPersistBy",
        &CSingleStreamDataAdderTest::testDetectorPersistBy));
    suiteOfTests->addTest(new CppUnit::TestCaller<CSingleStreamDataAdderTest>(
        "CSingleStreamDataAdderTest::testDetectorPersistOver",
        &CSingleStreamDataAdderTest::testDetectorPersistOver));
    suiteOfTests->addTest(new CppUnit::TestCaller<CSingleStreamDataAdderTest>(
        "CSingleStreamDataAdderTest::testDetectorPersistPartition",
        &CSingleStreamDataAdderTest::testDetectorPersistPartition));
    suiteOfTests->addTest(new CppUnit::TestCaller<CSingleStreamDataAdderTest>(
        "CSingleStreamDataAdderTest::testDetectorPersistDc",
        &CSingleStreamDataAdderTest::testDetectorPersistDc));
    suiteOfTests->addTest(new CppUnit::TestCaller<CSingleStreamDataAdderTest>(
        "CSingleStreamDataAdderTest::testDetectorPersistCount",
        &CSingleStreamDataAdderTest::testDetectorPersistCount));
    suiteOfTests->addTest(new CppUnit::TestCaller<CSingleStreamDataAdderTest>(
        "CSingleStreamDataAdderTest::testDetectorPersistCategorization",
        &CSingleStreamDataAdderTest::testDetectorPersistCategorization));
    return suiteOfTests;
}

void CSingleStreamDataAdderTest::testDetectorPersistBy() {
    this->detectorPersistHelper("testfiles/new_mlfields.conf",
                                "testfiles/big_ascending.txt", 0, "%d/%b/%Y:%T %z");
}

void CSingleStreamDataAdderTest::testDetectorPersistOver() {
    this->detectorPersistHelper("testfiles/new_mlfields_over.conf",
                                "testfiles/big_ascending.txt", 0, "%d/%b/%Y:%T %z");
}

void CSingleStreamDataAdderTest::testDetectorPersistPartition() {
    this->detectorPersistHelper("testfiles/new_mlfields_partition.conf",
                                "testfiles/big_ascending.txt", 0, "%d/%b/%Y:%T %z");
}

void CSingleStreamDataAdderTest::testDetectorPersistDc() {
    this->detectorPersistHelper("testfiles/new_persist_dc.conf",
                                "testfiles/files_users_programs.csv", 5);
}

void CSingleStreamDataAdderTest::testDetectorPersistCount() {
    this->detectorPersistHelper("testfiles/new_persist_count.conf",
                                "testfiles/files_users_programs.csv", 5);
}

void CSingleStreamDataAdderTest::testDetectorPersistCategorization() {
    this->detectorPersistHelper("testfiles/new_persist_categorization.conf",
                                "testfiles/time_messages.csv", 0);
}

void CSingleStreamDataAdderTest::detectorPersistHelper(const std::string& configFileName,
                                                       const std::string& inputFilename,
                                                       int latencyBuckets,
                                                       const std::string& timeFormat) {
    // Start by creating a detector with non-trivial state
    static const ml::core_t::TTime BUCKET_SIZE(3600);
    static const std::string JOB_ID("job");

    // Open the input and output files
    std::ifstream inputStrm(inputFilename.c_str());
    CPPUNIT_ASSERT(inputStrm.is_open());

    std::ofstream outputStrm(ml::core::COsFileFuncs::NULL_FILENAME);
    CPPUNIT_ASSERT(outputStrm.is_open());

    ml::model::CLimits limits;
    ml::api::CFieldConfig fieldConfig;
    CPPUNIT_ASSERT(fieldConfig.initFromFile(configFileName));

    ml::model::CAnomalyDetectorModelConfig modelConfig =
        ml::model::CAnomalyDetectorModelConfig::defaultConfig(
            BUCKET_SIZE, ml::model_t::E_None, "", BUCKET_SIZE * latencyBuckets, 0, false);

    ml::core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
    ml::api::CJsonOutputWriter outputWriter(JOB_ID, wrappedOutputStream);

    std::string origSnapshotId;
    std::size_t numOrigDocs(0);
    ml::api::CAnomalyJob origJob(
        JOB_ID, limits, fieldConfig, modelConfig, wrappedOutputStream,
        boost::bind(&reportPersistComplete, _1, boost::ref(origSnapshotId),
                    boost::ref(numOrigDocs)),
        nullptr, -1, "time", timeFormat);

    ml::api::CDataProcessor* firstProcessor(&origJob);

    // Chain the detector's input
    ml::api::COutputChainer outputChainer(origJob);

    // The typer knows how to assign categories to records
    ml::api::CFieldDataTyper typer(JOB_ID, fieldConfig, limits, outputChainer, outputWriter);

    if (fieldConfig.fieldNameSuperset().count(ml::api::CFieldDataTyper::MLCATEGORY_NAME) > 0) {
        LOG_DEBUG(<< "Applying the categorization typer for anomaly detection");
        firstProcessor = &typer;
    }

    using TInputParserUPtr = std::unique_ptr<ml::api::CInputParser>;
    const TInputParserUPtr parser{[&inputFilename, &inputStrm]() -> TInputParserUPtr {
        if (inputFilename.rfind(".csv") == inputFilename.length() - 4) {
            return std::make_unique<ml::api::CCsvInputParser>(inputStrm);
        }
        return std::make_unique<ml::api::CLineifiedJsonInputParser>(inputStrm);
    }()};

    CPPUNIT_ASSERT(parser->readStream(
        boost::bind(&ml::api::CDataProcessor::handleRecord, firstProcessor, _1)));

    // Persist the detector state to a stringstream

    std::string origPersistedState;
    {
        std::ostringstream* strm(nullptr);
        ml::api::CSingleStreamDataAdder::TOStreamP ptr(strm = new std::ostringstream());
        ml::api::CSingleStreamDataAdder persister(ptr);
        CPPUNIT_ASSERT(firstProcessor->persistState(persister));
        origPersistedState = strm->str();
    }

    // Now restore the state into a different detector

    std::string restoredSnapshotId;
    std::size_t numRestoredDocs(0);
    ml::api::CAnomalyJob restoredJob(
        JOB_ID, limits, fieldConfig, modelConfig, wrappedOutputStream,
        boost::bind(&reportPersistComplete, _1, boost::ref(restoredSnapshotId),
                    boost::ref(numRestoredDocs)));

    ml::api::CDataProcessor* restoredFirstProcessor(&restoredJob);

    // Chain the detector's input
    ml::api::COutputChainer restoredOutputChainer(restoredJob);

    // The typer knows how to assign categories to records
    ml::api::CFieldDataTyper restoredTyper(JOB_ID, fieldConfig, limits,
                                           restoredOutputChainer, outputWriter);

    size_t numCategorizerDocs(0);

    if (fieldConfig.fieldNameSuperset().count(ml::api::CFieldDataTyper::MLCATEGORY_NAME) > 0) {
        LOG_DEBUG(<< "Applying the categorization typer for anomaly detection");
        numCategorizerDocs = 1;
        restoredFirstProcessor = &restoredTyper;
    }

    {
        ml::core_t::TTime completeToTime(0);

        auto strm = std::make_shared<boost::iostreams::filtering_istream>();
        strm->push(ml::api::CStateRestoreStreamFilter());
        std::istringstream inputStream(origPersistedState);
        strm->push(inputStream);

        ml::api::CSingleStreamSearcher retriever(strm);

        CPPUNIT_ASSERT(restoredFirstProcessor->restoreState(retriever, completeToTime));
        CPPUNIT_ASSERT(completeToTime > 0);
        CPPUNIT_ASSERT_EQUAL(
            numOrigDocs + numCategorizerDocs,
            strm->component<ml::api::CStateRestoreStreamFilter>(0)->getDocCount());
    }

    // Finally, persist the new detector state and compare the result
    std::string newPersistedState;
    {
        std::ostringstream* strm(nullptr);
        ml::api::CSingleStreamDataAdder::TOStreamP ptr(strm = new std::ostringstream());
        ml::api::CSingleStreamDataAdder persister(ptr);
        CPPUNIT_ASSERT(restoredFirstProcessor->persistState(persister));
        newPersistedState = strm->str();
    }

    CPPUNIT_ASSERT_EQUAL(numOrigDocs, numRestoredDocs);

    // The snapshot ID can be different between the two persists, so replace the
    // first occurrence of it (which is in the bulk metadata)
    CPPUNIT_ASSERT_EQUAL(size_t(1), ml::core::CStringUtils::replaceFirst(
                                        origSnapshotId, "snap", origPersistedState));
    CPPUNIT_ASSERT_EQUAL(size_t(1), ml::core::CStringUtils::replaceFirst(
                                        restoredSnapshotId, "snap", newPersistedState));

    CPPUNIT_ASSERT_EQUAL(origPersistedState, newPersistedState);
}
