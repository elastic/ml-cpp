/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CBase64Filter.h>

#include <api/CDataSummarization.h>
#include <api/CDataFrameAnalyzer.h>

#include <test/CRandomNumbers.h>
#include <test/CDataFrameAnalysisSpecificationFactory.h>

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
using TBoolVec = std::vector<bool>;
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TStrVec = std::vector<std::string>;
using TStrVecVec = std::vector<TStrVec>;
using TDataFrameUPtr = std::unique_ptr<core::CDataFrame>;
using TFilteredInput = boost::iostreams::filtering_stream<boost::iostreams::input>;

auto generateCategoricalData(test::CRandomNumbers& rng,
                             std::size_t rows,
                             const TDoubleVec& expectedFrequencies) {

    TDoubleVecVec frequencies;
    rng.generateDirichletSamples(expectedFrequencies, 1, frequencies);

    TDoubleVec values(1);
    for (std::size_t j = 0; j < frequencies[0].size(); ++j) {
        std::size_t target{static_cast<std::size_t>(
            static_cast<double>(rows) * frequencies[0][j] + 0.5)};
        values.resize(values.size() + target, static_cast<double>(j));
    }
    values.resize(rows, values.back());
    rng.random_shuffle(values.begin(), values.end());
    rng.discard(1000000); // Make sure the categories are not correlated

    return std::make_pair(frequencies[0], values);
}

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
}

BOOST_AUTO_TEST_CASE(testIntegrationRegression) {
    std::size_t numberExamples = 1000;
    std::size_t cols = 3;
    test::CRandomNumbers rng;
    TDoubleVec weights{0.1, 100.0};

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TStrVec fieldNames{"numeric_col", "categorical_col", "target_col", ".", "."};
    TStrVec expectedFieldNames{"numeric_col", "categorical_col"};

    TStrVec fieldValues{"", "", "0", "", ""};

    TDoubleVecVec frequencies;
    TDoubleVecVec values(cols);
    rng.generateUniformSamples(-10.0, 10.0, numberExamples, values[0]);
    values[1] = generateCategoricalData(rng, numberExamples, {100., 5.0, 5.0}).second;

    for (std::size_t i = 0; i < numberExamples; ++i) {
        values[2].push_back(values[0][i] * weights[0] + values[1][i] * weights[1]);
    }

    test::CDataFrameAnalysisSpecificationFactory specFactory;
    api::CDataFrameAnalyzer analyzer{
        specFactory.rows(numberExamples)
            .columns(cols)
            .memoryLimit(30000000)
            .predictionCategoricalFieldNames({"categorical_col"})
            .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::regression(), "target_col"),
        outputWriterFactory};

    for (std::size_t i = 0; i < numberExamples; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            fieldValues[j] = core::CStringUtils::typeToStringPrecise(
                values[j][i], core::CIEEE754::E_DoublePrecision);
        }
        analyzer.handleRecord(fieldNames, fieldValues);
    }
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
         std::ifstream schemaFileStream("testfiles/data_summarization_schema/data_summarization.schema.json");
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

BOOST_AUTO_TEST_CASE(testIntegrationClassification) {
    std::size_t numberExamples = 200;
    std::size_t cols = 3;
    test::CRandomNumbers rng;
    TDoubleVec weights{0.1, 100.0};

    std::stringstream output;
    auto outputWriterFactory = [&output]() {
        return std::make_unique<core::CJsonOutputStreamWrapper>(output);
    };

    TStrVec fieldNames{"numeric_col", "categorical_col", "target_col", ".", "."};
    TStrVec expectedFieldNames{"numeric_col", "categorical_col"};

    TStrVec fieldValues{"", "", "0", "", ""};

    TDoubleVecVec frequencies;
    TDoubleVecVec values(cols);
    rng.generateUniformSamples(-10.0, 10.0, numberExamples, values[0]);
    values[1] = generateCategoricalData(rng, numberExamples, {100., 5.0, 5.0}).second;
    values[2] = generateCategoricalData(rng, numberExamples, {5.0, 5.0}).second;

    test::CDataFrameAnalysisSpecificationFactory specFactory;
    api::CDataFrameAnalyzer analyzer{
        specFactory.rows(numberExamples)
            .columns(cols)
            .memoryLimit(30000000)
            .predictionCategoricalFieldNames({"categorical_col", "target_col"})
            .predictionSpec(test::CDataFrameAnalysisSpecificationFactory::classification(), "target_col"),
        outputWriterFactory};

    TDataFrameUPtr frame{
        core::makeMainStorageDataFrame(cols + 2, numberExamples).first};
    for (std::size_t i = 0; i < numberExamples; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            fieldValues[j] = core::CStringUtils::typeToStringPrecise(
                values[j][i], core::CIEEE754::E_DoublePrecision);
        }
        analyzer.handleRecord(fieldNames, fieldValues);
    }
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
         std::ifstream schemaFileStream("testfiles/data_summarization_schema/data_summarization.schema.json");
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

BOOST_AUTO_TEST_SUITE_END()
