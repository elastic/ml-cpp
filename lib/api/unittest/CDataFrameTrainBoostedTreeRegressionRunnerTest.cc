/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CDataFrame.h>

#include <maths/CBoostedTreeImpl.h>

#include <api/CDataFrameAnalysisConfigReader.h>
#include <api/CDataFrameTrainBoostedTreeRegressionRunner.h>
#include <api/CSingleStreamSearcher.h>

#include <test/CDataFrameAnalysisSpecificationFactory.h>

#include <boost/test/unit_test.hpp>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

BOOST_AUTO_TEST_SUITE(CDataFrameTrainBoostedTreeRegressionRunnerTest)

using namespace ml;
namespace {
using TStrVec = std::vector<std::string>;
using TDoubleVec = std::vector<double>;
using TDataSearcherUPtr = std::unique_ptr<core::CDataSearcher>;
using TRestoreSearcherSupplier = std::function<TDataSearcherUPtr()>;
using TLossFunctionType = maths::boosted_tree::ELossType;
}

BOOST_AUTO_TEST_CASE(testPredictionFieldNameClash) {
    TStrVec errors;
    auto errorHandler = [&errors](std::string error) { errors.push_back(error); };
    core::CLogger::CScopeSetFatalErrorHandler scope{errorHandler};

    test::CDataFrameAnalysisSpecificationFactory specFactory;
    auto spec = specFactory.rows(5).columns(6).memoryLimit(13000000).predictionSpec(
        test::CDataFrameAnalysisSpecificationFactory::regression(), "dep_var");
    rapidjson::Document jsonParameters;
    jsonParameters.Parse("{"
                         "  \"dependent_variable\": \"dep_var\","
                         "  \"prediction_field_name\": \"is_training\""
                         "}");
    auto parameters =
        api::CDataFrameTrainBoostedTreeRegressionRunner::parameterReader().read(jsonParameters);
    api::CDataFrameTrainBoostedTreeRegressionRunner runner(*spec, parameters);

    BOOST_TEST_REQUIRE(errors.size() == 1);
    BOOST_TEST_REQUIRE(errors[0] == "Input error: prediction_field_name must not be equal to any of [is_training].");
}

BOOST_AUTO_TEST_CASE(testCreationForIncrementalTraining, *utf::tolerance(0.000001)) {
    // This test checks correct initialization of data summarization and the best forest for
    // incremental training.

    std::string filename{"testfiles/restore_incremental_model.json"};
    std::ifstream file{filename};
    if (file) {
        // get restore string stream
        std::stringstream restoreStream;
        restoreStream << file.rdbuf();
        file.close();

        auto restoreStreamPtr = std::make_shared<std::stringstream>(std::move(restoreStream));
        TRestoreSearcherSupplier restorerSupplier{[&restoreStreamPtr]() {
            return std::make_unique<api::CSingleStreamSearcher>(restoreStreamPtr);
        }};

        test::CDataFrameAnalysisSpecificationFactory specFactory;
        auto spec =
            specFactory.rows(100)
                .memoryLimit(15000000)
                .predictionMaximumNumberTrees(10)
                .predictionRestoreSearcherSupplier(&restorerSupplier)
                .regressionLossFunction(TLossFunctionType::E_MseRegression)
                .earlyStoppingEnabled(false)
                .task(test::CDataFrameAnalysisSpecificationFactory::TTask::E_Update)
                .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::regression(),
                                "target");
        auto* runner = static_cast<api::CDataFrameTrainBoostedTreeRegressionRunner*>(
            spec->runner());
        const auto& boostedTreeImpl =
            static_cast<const api::CDataFrameTrainBoostedTreeRegressionRunner&>(*runner)
                .boostedTreeFactory()
                .boostedTreeImpl();
        // // check that all trees restored
        BOOST_REQUIRE_EQUAL(boostedTreeImpl.trainedModel().size(), 8);
        // // check that all encoders restored
        BOOST_REQUIRE_EQUAL(boostedTreeImpl.encoder().numberInputColumns(), 5);
        BOOST_REQUIRE_EQUAL(boostedTreeImpl.encoder().numberEncodedColumns(), 3);
        TDoubleVec actualMics{boostedTreeImpl.encoder().encodedColumnMics()};
        TDoubleVec expectedMics{0.8342732, 0.2650428, 0};
        BOOST_TEST(actualMics == expectedMics, tt::per_element());
    } else {
        BOOST_FAIL("Connot read file " + filename);
    }
}

BOOST_AUTO_TEST_SUITE_END()
