/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "COutputChainerTest.h"

#include <core/CJsonOutputStreamWrapper.h>

#include <model/CLimits.h>

#include <api/CAnomalyJob.h>
#include <api/CFieldConfig.h>
#include <api/CJsonOutputWriter.h>
#include <api/CNdJsonInputParser.h>
#include <api/COutputChainer.h>

#include <test/CTestTmpDir.h>

#include "CMockDataProcessor.h"

#include <fstream>

CppUnit::Test* COutputChainerTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("COutputChainerTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<COutputChainerTest>(
        "COutputChainerTest::testChaining", &COutputChainerTest::testChaining));

    return suiteOfTests;
}

void COutputChainerTest::testChaining() {
    static const ml::core_t::TTime BUCKET_SIZE(3600);

    std::string inputFileName("testfiles/big_ascending.txt");
    std::string outputFileName(ml::test::CTestTmpDir::tmpDir() + "/chainerOutput.txt");

    {
        // Open the input and output files
        std::ifstream inputStrm(inputFileName.c_str());
        CPPUNIT_ASSERT(inputStrm.is_open());

        std::ofstream outputStrm(outputFileName.c_str());
        CPPUNIT_ASSERT(outputStrm.is_open());
        ml::core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
        // Set up the processing chain as:
        // big.txt -> typer -> chainer -> detector -> chainerOutput.txt

        ml::model::CLimits limits;
        ml::api::CFieldConfig fieldConfig;
        CPPUNIT_ASSERT(fieldConfig.initFromFile("testfiles/new_mlfields.conf"));

        ml::model::CAnomalyDetectorModelConfig modelConfig =
            ml::model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_SIZE);

        ml::api::CAnomalyJob job("job", limits, fieldConfig, modelConfig, wrappedOutputStream,
                                 ml::api::CAnomalyJob::TPersistCompleteFunc(),
                                 nullptr, -1, "time", "%d/%b/%Y:%T %z");

        ml::api::COutputChainer outputChainer(job);

        CMockDataProcessor mockProcessor(outputChainer);

        ml::api::CNdJsonInputParser parser(inputStrm);

        CPPUNIT_ASSERT(parser.readStreamIntoMaps(std::bind(
            &CMockDataProcessor::handleRecord, &mockProcessor, std::placeholders::_1)));
    }

    // Check the results by re-reading the output file
    std::ifstream reReadStrm(outputFileName.c_str());
    std::string line;
    std::string modelSizeString("\"model_bytes\":");

    std::string expectedLineStart("{\"bucket\":{\"job_id\":\"job\",\"timestamp\":1431853200000,");

    while (line.length() == 0 || line.find(modelSizeString) != std::string::npos) {
        std::getline(reReadStrm, line);
        LOG_DEBUG(<< "Read line: " << line);
    }

    // The first character of "line" will either be "[" or ","
    line = line.substr(1);
    // We don't care what the exact output is for this test
    // only that it is present and looks valid
    CPPUNIT_ASSERT_EQUAL(expectedLineStart, line.substr(0, expectedLineStart.length()));

    // TODO add more checks

    reReadStrm.close();
    CPPUNIT_ASSERT_EQUAL(0, ::remove(outputFileName.c_str()));
}
