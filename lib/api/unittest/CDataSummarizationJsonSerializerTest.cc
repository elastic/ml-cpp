/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CBase64Filter.h>

#include <maths/CBoostedTreeLoss.h>

#include <api/CDataFrameAnalyzer.h>
#include <api/CDataSummarizationJsonSerializer.h>

#include <test/CDataFrameAnalysisSpecificationFactory.h>
#include <test/CDataFrameAnalyzerTrainingFactory.h>
#include <test/CRandomNumbers.h>

#include <rapidjson/document.h>
#include <rapidjson/schema.h>

#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/test/unit_test.hpp>

#include <fstream>
#include <string>

BOOST_AUTO_TEST_SUITE(CDataSummarizationTest)

using namespace ml;

namespace {
using TDoubleVec = std::vector<double>;
using TStrVec = std::vector<std::string>;
using TStrVecVec = std::vector<TStrVec>;
using TFilteredInput = boost::iostreams::filtering_stream<boost::iostreams::input>;
using TLossFunctionType = maths::boosted_tree::ELossType;

std::stringstream decompressStream(std::stringstream&& compressedStream) {
    std::stringstream decompressedStream;
    {
        TFilteredInput inFilter;
        inFilter.push(boost::iostreams::gzip_decompressor());
        inFilter.push(core::CBase64Decoder());
        inFilter.push(compressedStream);
        boost::iostreams::copy(inFilter, decompressedStream);
    }
    return decompressedStream;
}

void testSchema(TLossFunctionType lossType) {
    std::size_t numberExamples = 100;
    std::size_t cols = 3;

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    test::CDataFrameAnalysisSpecificationFactory specFactory;
    specFactory.rows(numberExamples).columns(cols).memoryLimit(30000000);
    std::string analysisType;
    if (lossType == ml::maths::boosted_tree::E_BinaryClassification) {
        specFactory.predictionCategoricalFieldNames({"categorical_col", "target_col"});
        analysisType = test::CDataFrameAnalysisSpecificationFactory::classification();
    } else {
        specFactory.predictionCategoricalFieldNames({"categorical_col"});
        analysisType = test::CDataFrameAnalysisSpecificationFactory::regression();
    }
    api::CDataFrameAnalyzer analyzer{
        specFactory.predictionSpec(analysisType, "target_col"), outputWriterFactory};

    TStrVec fieldNames{"numeric_col", "categorical_col", "target_col", ".", "."};
    TStrVec fieldValues{"", "", "0", "", ""};
    test::CDataFrameAnalyzerTrainingFactory::addPredictionTestData(
        lossType, fieldNames, fieldValues, analyzer, numberExamples);
    analyzer.handleRecord(fieldNames, {"", "", "", "", "$"});
    auto analysisRunner = analyzer.runner();

    auto dataSummarization = analysisRunner->dataSummarization(analyzer.dataFrame());
    // verify compressed definition
    {
        std::string dataSummarizationStr{dataSummarization->jsonString()};
        std::stringstream decompressedStream{
            decompressStream(dataSummarization->jsonCompressedStream())};
        BOOST_TEST_REQUIRE(decompressedStream.str() == dataSummarizationStr);
    }

    // verify json schema
    {
        std::ifstream schemaFileStream(
            "testfiles/data_summarization_schema/data_summarization.schema.json");
        BOOST_REQUIRE_MESSAGE(schemaFileStream.is_open(), "Cannot open test file!");
        std::string schemaJson((std::istreambuf_iterator<char>(schemaFileStream)),
                               std::istreambuf_iterator<char>());
        rapidjson::Document schemaDocument;
        BOOST_REQUIRE_MESSAGE(schemaDocument.Parse(schemaJson).HasParseError() == false,
                              "Cannot parse JSON schema!");
        rapidjson::SchemaDocument schema(schemaDocument);

        rapidjson::Document doc;
        BOOST_REQUIRE_MESSAGE(doc.Parse(dataSummarization->jsonString()).HasParseError() == false,
                              "Error parsing JSON definition!");

        rapidjson::SchemaValidator validator(schema);
        if (doc.Accept(validator) == false) {
            rapidjson::StringBuffer sb;
            validator.GetInvalidSchemaPointer().StringifyUriFragment(sb);
            LOG_ERROR(<< "Invalid schema: " << sb.GetString());
            LOG_ERROR(<< "Invalid keyword: " << validator.GetInvalidSchemaKeyword());
            sb.Clear();
            validator.GetInvalidDocumentPointer().StringifyUriFragment(sb);
            LOG_ERROR(<< "Invalid document: " << sb.GetString());
            LOG_DEBUG(<< "Document: " << dataSummarization->jsonString());
            BOOST_FAIL("Schema validation failed");
        }
    }
}
}

BOOST_AUTO_TEST_CASE(testRegression) {
    testSchema(TLossFunctionType::E_MseRegression);
}

BOOST_AUTO_TEST_CASE(testClassification) {
    testSchema(TLossFunctionType::E_BinaryClassification);
}

BOOST_AUTO_TEST_SUITE_END()
