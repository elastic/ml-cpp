/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CBackgroundPersisterTest.h"

#include <core/CJsonOutputStreamWrapper.h>
#include <core/COsFileFuncs.h>
#include <core/CStringUtils.h>
#include <core/CoreTypes.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CLimits.h>

#include <api/CAnomalyJob.h>
#include <api/CBackgroundPersister.h>
#include <api/CDataProcessor.h>
#include <api/CFieldConfig.h>
#include <api/CFieldDataTyper.h>
#include <api/CJsonOutputWriter.h>
#include <api/CModelSnapshotJsonWriter.h>
#include <api/CNdJsonInputParser.h>
#include <api/CNullOutput.h>
#include <api/COutputChainer.h>
#include <api/CSingleStreamDataAdder.h>

#include <boost/bind.hpp>
#include <boost/ref.hpp>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

namespace {

void reportPersistComplete(ml::api::CModelSnapshotJsonWriter::SModelSnapshotReport modelSnapshotReport,
                           std::string& snapshotIdOut,
                           size_t& numDocsOut) {
    LOG_DEBUG(<< "Persist complete with description: " << modelSnapshotReport.s_Description);
    snapshotIdOut = modelSnapshotReport.s_SnapshotId;
    numDocsOut = modelSnapshotReport.s_NumDocs;
}
}

CppUnit::Test* CBackgroundPersisterTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CBackgroundPersisterTest");
    suiteOfTests->addTest(new CppUnit::TestCaller<CBackgroundPersisterTest>(
        "CBackgroundPersisterTest::testDetectorPersistBy",
        &CBackgroundPersisterTest::testDetectorPersistBy));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBackgroundPersisterTest>(
        "CBackgroundPersisterTest::testDetectorPersistOver",
        &CBackgroundPersisterTest::testDetectorPersistOver));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBackgroundPersisterTest>(
        "CBackgroundPersisterTest::testDetectorPersistPartition",
        &CBackgroundPersisterTest::testDetectorPersistPartition));
    suiteOfTests->addTest(new CppUnit::TestCaller<CBackgroundPersisterTest>(
        "CBackgroundPersisterTest::testCategorizationOnlyPersist",
        &CBackgroundPersisterTest::testCategorizationOnlyPersist));

    return suiteOfTests;
}

void CBackgroundPersisterTest::testDetectorPersistBy() {
    this->foregroundBackgroundCompCategorizationAndAnomalyDetection("testfiles/new_mlfields.conf");
}

void CBackgroundPersisterTest::testDetectorPersistOver() {
    this->foregroundBackgroundCompCategorizationAndAnomalyDetection("testfiles/new_mlfields_over.conf");
}

void CBackgroundPersisterTest::testDetectorPersistPartition() {
    this->foregroundBackgroundCompCategorizationAndAnomalyDetection(
        "testfiles/new_mlfields_partition.conf");
}

void CBackgroundPersisterTest::testCategorizationOnlyPersist() {
    // Start by creating a categorizer with non-trivial state

    static const std::string JOB_ID("job");

    std::string inputFilename("testfiles/big_ascending.txt");

    // Open the input and output files
    std::ifstream inputStrm(inputFilename);
    CPPUNIT_ASSERT(inputStrm.is_open());

    std::ofstream outputStrm(ml::core::COsFileFuncs::NULL_FILENAME);
    CPPUNIT_ASSERT(outputStrm.is_open());

    ml::model::CLimits limits;
    ml::api::CFieldConfig fieldConfig("agent");

    std::ostringstream* backgroundStream(nullptr);
    ml::api::CSingleStreamDataAdder::TOStreamP backgroundStreamPtr(
        backgroundStream = new std::ostringstream());
    ml::api::CSingleStreamDataAdder backgroundDataAdder(backgroundStreamPtr);
    // The 300 second persist interval is irrelevant here - we bypass the timer
    // in this test and kick off the background persistence chain explicitly
    ml::api::CBackgroundPersister backgroundPersister(300, backgroundDataAdder);

    std::ostringstream* foregroundStream(nullptr);
    ml::api::CSingleStreamDataAdder::TOStreamP foregroundStreamPtr(
        foregroundStream = new std::ostringstream());
    {
        ml::core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
        ml::api::CJsonOutputWriter outputWriter(JOB_ID, wrappedOutputStream);

        // All output we're interested in goes via the JSON output writer, so
        // output of the categorised input data can be dropped
        ml::api::CNullOutput nullOutput;

        // The typer knows how to assign categories to records
        ml::api::CFieldDataTyper typer(JOB_ID, fieldConfig, limits, nullOutput,
                                       outputWriter, &backgroundPersister);

        ml::api::CNdJsonInputParser parser(inputStrm);

        CPPUNIT_ASSERT(parser.readStreamAsMaps(
            boost::bind(&ml::api::CDataProcessor::handleRecord, &typer, _1)));

        // Persist the processors' state in the background
        CPPUNIT_ASSERT(typer.periodicPersistState(backgroundPersister));
        CPPUNIT_ASSERT(backgroundPersister.startPersist());

        LOG_DEBUG(<< "Before waiting for the background persister to be idle");
        CPPUNIT_ASSERT(backgroundPersister.waitForIdle());
        LOG_DEBUG(<< "After waiting for the background persister to be idle");

        // Now persist the processors' state in the foreground
        ml::api::CSingleStreamDataAdder foregroundDataAdder(foregroundStreamPtr);
        CPPUNIT_ASSERT(typer.persistState(foregroundDataAdder));
    }

    std::string backgroundState = backgroundStream->str();
    std::string foregroundState = foregroundStream->str();

    // Replace the zero byte separators so the expected/actual strings don't get
    // truncated by CppUnit if the test fails
    std::replace(backgroundState.begin(), backgroundState.end(), '\0', ',');
    std::replace(foregroundState.begin(), foregroundState.end(), '\0', ',');

    CPPUNIT_ASSERT_EQUAL(backgroundState, foregroundState);
}

void CBackgroundPersisterTest::foregroundBackgroundCompCategorizationAndAnomalyDetection(
    const std::string& configFileName) {
    // Start by creating processors with non-trivial state

    static const ml::core_t::TTime BUCKET_SIZE(3600);
    static const std::string JOB_ID("job");

    std::string inputFilename("testfiles/big_ascending.txt");

    // Open the input and output files
    std::ifstream inputStrm(inputFilename.c_str());
    CPPUNIT_ASSERT(inputStrm.is_open());

    std::ofstream outputStrm(ml::core::COsFileFuncs::NULL_FILENAME);
    CPPUNIT_ASSERT(outputStrm.is_open());

    ml::model::CLimits limits;
    ml::api::CFieldConfig fieldConfig;
    CPPUNIT_ASSERT(fieldConfig.initFromFile(configFileName));

    ml::model::CAnomalyDetectorModelConfig modelConfig =
        ml::model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);

    std::ostringstream* backgroundStream(nullptr);
    ml::api::CSingleStreamDataAdder::TOStreamP backgroundStreamPtr(
        backgroundStream = new std::ostringstream());
    ml::api::CSingleStreamDataAdder backgroundDataAdder(backgroundStreamPtr);
    // The 300 second persist interval is irrelevant here - we bypass the timer
    // in this test and kick off the background persistence chain explicitly
    ml::api::CBackgroundPersister backgroundPersister(300, backgroundDataAdder);

    std::string snapshotId;
    std::size_t numDocs(0);

    std::string backgroundSnapshotId;
    std::string foregroundSnapshotId;

    std::ostringstream* foregroundStream(nullptr);
    ml::api::CSingleStreamDataAdder::TOStreamP foregroundStreamPtr(
        foregroundStream = new std::ostringstream());
    {
        ml::core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
        ml::api::CJsonOutputWriter outputWriter(JOB_ID, wrappedOutputStream);

        ml::api::CAnomalyJob job(
            JOB_ID, limits, fieldConfig, modelConfig, wrappedOutputStream,
            boost::bind(&reportPersistComplete, _1, boost::ref(snapshotId),
                        boost::ref(numDocs)),
            &backgroundPersister, -1, "time", "%d/%b/%Y:%T %z");

        ml::api::CDataProcessor* firstProcessor(&job);

        // Chain the detector's input
        ml::api::COutputChainer outputChainer(job);

        // The typer knows how to assign categories to records
        ml::api::CFieldDataTyper typer(JOB_ID, fieldConfig, limits, outputChainer, outputWriter);

        if (fieldConfig.fieldNameSuperset().count(ml::api::CFieldDataTyper::MLCATEGORY_NAME) > 0) {
            LOG_DEBUG(<< "Applying the categorization typer for anomaly detection");
            firstProcessor = &typer;
        }

        ml::api::CNdJsonInputParser parser(inputStrm);

        CPPUNIT_ASSERT(parser.readStreamAsMaps(boost::bind(
            &ml::api::CDataProcessor::handleRecord, firstProcessor, _1)));

        // Persist the processors' state in the background
        CPPUNIT_ASSERT(firstProcessor->periodicPersistState(backgroundPersister));
        CPPUNIT_ASSERT(backgroundPersister.startPersist());

        LOG_DEBUG(<< "Before waiting for the background persister to be idle");
        CPPUNIT_ASSERT(backgroundPersister.waitForIdle());
        LOG_DEBUG(<< "After waiting for the background persister to be idle");
        backgroundSnapshotId = snapshotId;

        // Now persist the processors' state in the foreground
        ml::api::CSingleStreamDataAdder foregroundDataAdder(foregroundStreamPtr);
        CPPUNIT_ASSERT(firstProcessor->persistState(foregroundDataAdder));
        foregroundSnapshotId = snapshotId;
    }

    std::string backgroundState = backgroundStream->str();
    std::string foregroundState = foregroundStream->str();

    // The snapshot ID can be different between the two persists, so replace the
    // first occurrence of it (which is in the bulk metadata)
    CPPUNIT_ASSERT_EQUAL(size_t(1), ml::core::CStringUtils::replaceFirst(
                                        backgroundSnapshotId, "snap", backgroundState));
    CPPUNIT_ASSERT_EQUAL(size_t(1), ml::core::CStringUtils::replaceFirst(
                                        foregroundSnapshotId, "snap", foregroundState));

    // Replace the zero byte separators so the expected/actual strings don't get
    // truncated by CppUnit if the test fails
    std::replace(backgroundState.begin(), backgroundState.end(), '\0', ',');
    std::replace(foregroundState.begin(), foregroundState.end(), '\0', ',');

    CPPUNIT_ASSERT_EQUAL(backgroundState, foregroundState);
}
