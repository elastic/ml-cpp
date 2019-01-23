/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CDataFrameAnalysisSpecificationTest.h"

#include <core/CContainerPrinter.h>
#include <core/CDataFrame.h>
#include <core/CLogger.h>
#include <core/CStopWatch.h>

#include <api/CDataFrameAnalysisRunner.h>
#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CDataFrameOutliersRunner.h>

#include <test/CRandomNumbers.h>

#include <boost/make_unique.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace ml;

namespace {
using TStrVec = std::vector<std::string>;
using TRunnerFactoryUPtr = std::unique_ptr<api::CDataFrameAnalysisRunnerFactory>;
using TRunnerFactoryUPtrVec = std::vector<TRunnerFactoryUPtr>;

class CDataFrameTestAnalysisRunner : public api::CDataFrameAnalysisRunner {
public:
    CDataFrameTestAnalysisRunner(const api::CDataFrameAnalysisSpecification& spec)
        : api::CDataFrameAnalysisRunner{spec} {}

    virtual std::size_t numberOfPartitions() const { return 1; }
    virtual std::size_t numberExtraColumns() const { return 2; }
    virtual void writeOneRow(TRowRef, core::CRapidJsonConcurrentLineWriter&) const {}

protected:
    void runImpl(core::CDataFrame&) {
        TProgressRecorder recordProgress{this->progressRecorder()};
        for (std::size_t i = 0; i < 31; ++i) {
            std::vector<std::size_t> wait;
            ms_Rng.generateUniformSamples(1, 20, 1, wait);

            std::this_thread::sleep_for(std::chrono::milliseconds(wait[0]));

            recordProgress(1.0 / 30.0);
        }
        this->setToFinished();
    }

private:
    virtual std::size_t
    estimateBookkeepingMemoryUsage(std::size_t, std::size_t, std::size_t) const {
        return 0;
    }

private:
    static test::CRandomNumbers ms_Rng;
};

test::CRandomNumbers CDataFrameTestAnalysisRunner::ms_Rng;

class CDataFrameTestAnalysisRunnerFactory : public api::CDataFrameAnalysisRunnerFactory {
public:
    virtual const char* name() const { return "test"; }

private:
    virtual TRunnerUPtr makeImpl(const api::CDataFrameAnalysisSpecification& spec) const {
        return std::make_unique<CDataFrameTestAnalysisRunner>(spec);
    }

    virtual TRunnerUPtr makeImpl(const api::CDataFrameAnalysisSpecification& spec,
                                 const rapidjson::Value&) const {
        return std::make_unique<CDataFrameTestAnalysisRunner>(spec);
    }
};
}

void CDataFrameAnalysisSpecificationTest::testCreate() {
    // This test focuses on checking the validation code we apply to the object
    // rather than the JSON parsing so we don't bother with random fuzzing of the
    // input string and simply check validation for each field.

    TStrVec errors;
    auto errorHandler = [&errors](std::string error) { errors.push_back(error); };
    core::CLogger::CScopeSetFatalErrorHandler scope{errorHandler};

    auto outliersFactory = []() {
        TRunnerFactoryUPtr factory{
            boost::make_unique<api::CDataFrameOutliersRunnerFactory>()};
        TRunnerFactoryUPtrVec factories;
        factories.push_back(std::move(factory));
        return factories;
    };
    auto jsonSpec = [](const std::string& rows, const std::string& cols,
                       const std::string& memory, const std::string& threads,
                       const std::string& name, const std::string& parameters = "",
                       const std::string& junk = "") {
        std::ostringstream result;
        result << "{\n";
        if (rows.size() > 0) {
            result << "   \"rows\": " << rows << ",\n";
        }
        if (cols.size() > 0) {
            result << "   \"cols\": " << cols << ",\n";
        }
        if (memory.size() > 0) {
            result << "   \"memory_limit\": " << memory << ",\n";
        }
        if (threads.size() > 0) {
            result << "   \"threads\": " << threads << ",\n";
        }
        if (junk.size() > 0) {
            result << "   \"" << junk << "\": 4,\n";
        }
        result << "   \"analysis\": {\n";
        if (name.size() > 0) {
            result << "     \"name\": \"" << name << "\"";
        }
        if (parameters.size() > 0) {
            result << ",\n     \"parameters\": " << parameters << "\n";
        } else {
            result << "\n";
        }
        result << "   }\n}";
        return result.str();
    };

    LOG_DEBUG(<< "Valid input");
    {
        LOG_TRACE(<< jsonSpec("1000", "20", "100000", "2", "outliers"));
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(), jsonSpec("1000", "20", "100000", "2", "outliers")};
        CPPUNIT_ASSERT_EQUAL(std::size_t{1000}, spec.numberRows());
        CPPUNIT_ASSERT_EQUAL(std::size_t{20}, spec.numberColumns());
        CPPUNIT_ASSERT_EQUAL(std::size_t{100000}, spec.memoryLimit());
        CPPUNIT_ASSERT_EQUAL(std::size_t{2}, spec.numberThreads());
    }
    LOG_DEBUG(<< "Bad input");
    {
        LOG_TRACE(<< jsonSpec("", "20", "100000", "2", "outliers"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(), jsonSpec("", "20", "100000", "2", "outliers")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("1000", "", "100000", "2", "outliers"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(), jsonSpec("1000", "", "100000", "2", "outliers")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("1000", "20", "", "2", "outliers"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(), jsonSpec("1000", "20", "", "2", "outliers")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("1000", "20", "100000", "", "outliers"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(), jsonSpec("1000", "20", "100000", "", "outliers")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("1000", "20", "100000", "2", ""));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(), jsonSpec("1000", "20", "100000", "2", "")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("-3", "20", "100000", "2", "outliers"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(), jsonSpec("-3", "20", "100000", "2", "outliers")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("1000", "0", "100000", "2", "outliers"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(), jsonSpec("1000", "0", "100000", "2", "outliers")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("1000", "20", "ZZ", "2", "outliers"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(), jsonSpec("1000", "20", "\"ZZ\"", "2", "outliers")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("1000", "20", "100000", "-1", "outliers"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(), jsonSpec("1000", "20", "100000", "-1", "outliers")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }
    {
        LOG_TRACE(<< jsonSpec("100", "20", "100000", "2", "outl1ers"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(), jsonSpec("100", "20", "100000", "2", "outl1ers")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }

    LOG_DEBUG(<< "Invalid parameters");
    {
        LOG_TRACE(<< jsonSpec("100", "20", "100000", "2", "outliers",
                              "{\"number_neighbours\": 0}"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(), jsonSpec("100", "20", "100000", "2", "outliers",
                                        "{\"number_neighbours\": 0}")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }

    LOG_DEBUG(<< "Extra junk");
    {
        LOG_TRACE(<< jsonSpec("1000", "2", "100000", "2", "outliers", "", "threeds"));
        errors.clear();
        api::CDataFrameAnalysisSpecification spec{
            outliersFactory(),
            jsonSpec("1000", "2", "100000", "2", "outliers", "", "threeds")};
        LOG_DEBUG(<< core::CContainerPrinter::print(errors));
        CPPUNIT_ASSERT(errors.size() > 0);
    }
}

void CDataFrameAnalysisSpecificationTest::testRunAnalysis() {
    // Test job running basics: start, wait, progress and errors.

    auto testFactory = []() {
        TRunnerFactoryUPtr factory{boost::make_unique<CDataFrameTestAnalysisRunnerFactory>()};
        TRunnerFactoryUPtrVec factories;
        factories.push_back(std::move(factory));
        return factories;
    };
    std::string jsonSpec{"{\n"
                         "  \"rows\": 100,\n"
                         "  \"cols\": 10,\n"
                         "  \"memory_limit\": 1000,\n"
                         "  \"threads\": 1,\n"
                         "  \"analysis\": {\n"
                         "    \"name\": \"test\""
                         "  }"
                         "}"};

    for (std::size_t i = 0; i < 10; ++i) {
        api::CDataFrameAnalysisSpecification spec{testFactory(), jsonSpec};

        auto frameAndDirectory = core::makeMainStorageDataFrame(10);
        auto frame = std::move(frameAndDirectory.first);

        api::CDataFrameAnalysisRunner* runner{spec.run(*frame)};
        CPPUNIT_ASSERT(runner != nullptr);

        double lastProgress{runner->progress()};
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));

            LOG_TRACE(<< "progress = " << lastProgress);
            CPPUNIT_ASSERT(runner->progress() >= lastProgress);
            lastProgress = runner->progress();
            if (runner->finished()) {
                break;
            }
            CPPUNIT_ASSERT(runner->progress() < 1.0);
        }

        LOG_DEBUG(<< "progress = " << lastProgress);

        CPPUNIT_ASSERT_EQUAL(1.0, runner->progress());
    }
}

CppUnit::Test* CDataFrameAnalysisSpecificationTest::suite() {
    CppUnit::TestSuite* suiteOfTests =
        new CppUnit::TestSuite("CDataFrameAnalysisSpecificationTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalysisSpecificationTest>(
        "CDataFrameAnalysisSpecificationTest::testCreate",
        &CDataFrameAnalysisSpecificationTest::testCreate));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameAnalysisSpecificationTest>(
        "CDataFrameAnalysisSpecificationTest::testRunAnalysis",
        &CDataFrameAnalysisSpecificationTest::testRunAnalysis));

    return suiteOfTests;
}
