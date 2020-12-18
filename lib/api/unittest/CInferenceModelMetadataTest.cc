/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <boost/test/tools/interface.hpp>
#include <core/CJsonOutputStreamWrapper.h>

#include <maths/CBoostedTreeLoss.h>

#include <api/CDataFrameAnalyzer.h>

#include <test/CDataFrameAnalysisSpecificationFactory.h>
#include <test/CDataFrameAnalyzerTrainingFactory.h>

#include <rapidjson/document.h>
#include <rapidjson/schema.h>

#include <boost/test/unit_test.hpp>

#include <fstream>
#include <string>

BOOST_AUTO_TEST_SUITE(CInferenceModelMetadataTest)

using namespace ml;

namespace {
using TDoubleVec = std::vector<double>;
using TStrVec = std::vector<std::string>;
using TLossFunctionType = maths::boosted_tree::ELossType;
}

BOOST_AUTO_TEST_CASE(testJsonSchema) {
    // Test the results the analyzer produces match running the regression directly.

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TDoubleVec expectedPredictions;

    TStrVec fieldNames{"f1", "f2", "f3", "f4", "target", ".", "."};
    TStrVec fieldValues{"", "", "", "", "", "0", ""};
    api::CDataFrameAnalyzer analyzer{
        test::CDataFrameAnalysisSpecificationFactory{}
        .predictionLambda(0.5)
        .predictionEta(.5)
        .predictionGamma(0.5)
        .predictionSpec(
            test::CDataFrameAnalysisSpecificationFactory::regression(), "target"),
        outputWriterFactory};
    test::CDataFrameAnalyzerTrainingFactory::addPredictionTestData(
        TLossFunctionType::E_MseRegression, fieldNames, fieldValues, analyzer,
        expectedPredictions);

    analyzer.handleRecord(fieldNames, {"", "", "", "", "", "", "$"});

    rapidjson::Document results;
    rapidjson::ParseResult ok(results.Parse(output.str()));
    LOG_DEBUG(<< output.str());
    BOOST_TEST_REQUIRE(static_cast<bool>(ok) == true);

    std::ifstream modelMetaDataSchemaFileStream("testfiles/model_meta_data/model_meta_data.schema.json");
    BOOST_REQUIRE_MESSAGE(modelMetaDataSchemaFileStream.is_open(), "Cannot open test file!");
    std::string modelMetaDataSchemaJson(
        (std::istreambuf_iterator<char>(modelMetaDataSchemaFileStream)),
        std::istreambuf_iterator<char>());
    rapidjson::Document modelMetaDataSchemaDocument;
    BOOST_REQUIRE_MESSAGE(
        modelMetaDataSchemaDocument.Parse(modelMetaDataSchemaJson).HasParseError() == false,
        "Cannot parse JSON schema!");
    rapidjson::SchemaDocument modelMetaDataSchema(modelMetaDataSchemaDocument);
    rapidjson::SchemaValidator modelMetaDataValidator(modelMetaDataSchema);

    bool hasModelMetadata{false};
    for (const auto& result : results.GetArray()) {
        if (result.HasMember("model_metadata")) {
            hasModelMetadata = true;
            BOOST_TEST_REQUIRE(result["model_metadata"].IsObject() = true);
            if(result["model_metadata"].Accept(modelMetaDataValidator) == false) {
               rapidjson::StringBuffer sb;
                modelMetaDataValidator.GetInvalidSchemaPointer().StringifyUriFragment(sb);
                LOG_ERROR(<< "Invalid schema: " << sb.GetString());
                LOG_ERROR(<< "Invalid keyword: "
                          << modelMetaDataValidator.GetInvalidSchemaKeyword());
                sb.Clear();
                modelMetaDataValidator.GetInvalidDocumentPointer().StringifyUriFragment(sb);
                LOG_ERROR(<< "Invalid document: " << sb.GetString());
                BOOST_FAIL("Schema validation failed"); 
            }
        }
    }

    BOOST_TEST_REQUIRE(hasModelMetadata);

}

BOOST_AUTO_TEST_SUITE_END()
