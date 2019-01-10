/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CDataFrameAnalyzerTest.h"

#include <core/CContainerPrinter.h>
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CStringUtils.h>

#include <maths/CLocalOutlierFactors.h>

#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CDataFrameAnalyzer.h>

#include <test/CRandomNumbers.h>

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace ml;

namespace {
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TStrVec = std::vector<std::string>;
using TPoint = maths::CDenseVector<maths::CFloatStorage>;
using TPointVec = std::vector<TPoint>;

std::unique_ptr<api::CDataFrameAnalysisSpecification> outlierSpec() {
    std::string spec{"{\n"
                     "  \"rows\": 100,\n"
                     "  \"cols\": 5,\n"
                     "  \"memory_limit\": 100000,\n"
                     "  \"threads\": 1,\n"
                     "  \"analysis\": {\n"
                     "    \"name\": \"outliers\""
                     "  }"
                     "}"};
    return std::make_unique<api::CDataFrameAnalysisSpecification>(spec);
}

void addTestData(TStrVec fieldNames,
                 TStrVec fieldValues,
                 api::CDataFrameAnalyzer& analyzer,
                 TDoubleVec& expectedScores) {

    std::size_t numberInliers{100};
    std::size_t numberOutliers{10};

    test::CRandomNumbers rng;

    TDoubleVec mean{1.0, 10.0, 4.0, 8.0, 3.0};
    TDoubleVecVec covariance{{1.0, 0.1, -0.1, 0.3, 0.2},
                             {0.1, 1.3, -0.3, 0.1, 0.1},
                             {-0.1, -0.3, 2.1, 0.1, 0.2},
                             {0.3, 0.1, 0.1, 0.8, 0.2},
                             {0.2, 0.1, 0.2, 0.2, 2.2}};

    TDoubleVecVec inliers;
    rng.generateMultivariateNormalSamples(mean, covariance, numberInliers, inliers);

    TDoubleVec outliers;
    rng.generateUniformSamples(0.0, 10.0, numberOutliers * 5, outliers);

    TPointVec points(numberInliers + numberOutliers, TPoint(5));
    for (std::size_t i = 0; i < inliers.size(); ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            fieldValues[j] = core::CStringUtils::typeToStringPrecise(
                inliers[i][j], core::CIEEE754::E_DoublePrecision);
            points[i](j) = inliers[i][j];
        }
        analyzer.handleRecord(fieldNames, fieldValues);
    }
    for (std::size_t i = 0, j = numberInliers; i < outliers.size(); ++j) {
        for (std::size_t k = 0; k < 5; ++i, ++k) {
            fieldValues[k] = core::CStringUtils::typeToStringPrecise(
                outliers[i], core::CIEEE754::E_DoublePrecision);
            points[j](k) = outliers[i];
        }
        analyzer.handleRecord(fieldNames, fieldValues);
    }

    maths::CLocalOutlierFactors lofs;
    lofs.ensemble(points, expectedScores);
}
}

void CDataFrameAnalyzerTest::testWithoutControlMessages() {

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    api::CDataFrameAnalyzer analyzer{outlierSpec(), outputWriterFactory};

    TDoubleVec expectedScores;

    TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5"};
    TStrVec fieldValues(fieldNames.size());
    addTestData(fieldNames, fieldValues, analyzer, expectedScores);

    analyzer.receivedAllRows();
    analyzer.run();

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str().c_str()));
    CPPUNIT_ASSERT(static_cast<bool>(ok) == true);

    auto expectedScore = expectedScores.begin();
    for (const auto& result : results.GetArray()) {
        CPPUNIT_ASSERT(expectedScore != expectedScores.end());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(*expectedScore,
                                     result["results"]["outlier_score"].GetDouble(),
                                     1e-6 * *expectedScore);
        ++expectedScore;
    }
    CPPUNIT_ASSERT(expectedScore == expectedScores.end());
}

void CDataFrameAnalyzerTest::testRunOutlierDetection() {

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    api::CDataFrameAnalyzer analyzer{outlierSpec(), outputWriterFactory};

    TDoubleVec expectedScores;

    TStrVec fieldNames{"c1", "c2", "c3", "c4", "c5", "."};
    TStrVec fieldValues(fieldNames.size());
    addTestData(fieldNames, fieldValues, analyzer, expectedScores);

    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "$"});

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str().c_str()));
    CPPUNIT_ASSERT(static_cast<bool>(ok) == true);

    auto expectedScore = expectedScores.begin();
    for (const auto& result : results.GetArray()) {
        CPPUNIT_ASSERT(expectedScore != expectedScores.end());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(*expectedScore,
                                     result["results"]["outlier_score"].GetDouble(),
                                     1e-6 * *expectedScore);
        ++expectedScore;
    }
    CPPUNIT_ASSERT(expectedScore == expectedScores.end());
}

void CDataFrameAnalyzerTest::testFlushMessage() {

    // Test that white space is just ignored.

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    api::CDataFrameAnalyzer analyzer{outlierSpec(), outputWriterFactory};
    CPPUNIT_ASSERT_EQUAL(
        true, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", "."},
                                    {"", "", "", "", "", "           "}));
}

void CDataFrameAnalyzerTest::testErrors() {

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    // Test with bad analysis specification.
    {
        api::CDataFrameAnalyzer analyzer{
            std::make_unique<api::CDataFrameAnalysisSpecification>(std::string{"junk"}),
            outputWriterFactory};
        CPPUNIT_ASSERT_EQUAL(false,
                             analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5"},
                                                   {"10", "10", "10", "10", "10"}));
    }

    // Test control message in the wrong position
    {
        api::CDataFrameAnalyzer analyzer{outlierSpec(), outputWriterFactory};
        CPPUNIT_ASSERT_EQUAL(
            false, analyzer.handleRecord({"c1", "c2", "c3", ".", "c4", "c5"},
                                         {"10", "10", "10", "", "10", "10"}));
    }

    // Test bad control message
    {
        api::CDataFrameAnalyzer analyzer{outlierSpec(), outputWriterFactory};
        CPPUNIT_ASSERT_EQUAL(
            false, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", "."},
                                         {"10", "10", "10", "10", "10", "foo"}));
    }

    // Test bad input
    {
        api::CDataFrameAnalyzer analyzer{outlierSpec(), outputWriterFactory};
        CPPUNIT_ASSERT_EQUAL(
            false, analyzer.handleRecord({"c1", "c2", "c3", "c4", "c5", "."},
                                         {"10", "10", "10", "10", "10"}));
    }
}

CppUnit::Test* CDataFrameAnalyzerTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CDataFrameAnalyzerTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTest>(
        "CDataFrameAnalyzerTest::testWithoutControlMessages",
        &CDataFrameAnalyzerTest::testWithoutControlMessages));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTest>(
        "CDataFrameAnalyzerTest::testRunOutlierDetection",
        &CDataFrameAnalyzerTest::testRunOutlierDetection));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTest>(
        "CDataFrameAnalyzerTest::testFlushMessage", &CDataFrameAnalyzerTest::testFlushMessage));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalyzerTest>(
        "CDataFrameAnalyzerTest::testErrors", &CDataFrameAnalyzerTest::testErrors));

    return suiteOfTests;
}
