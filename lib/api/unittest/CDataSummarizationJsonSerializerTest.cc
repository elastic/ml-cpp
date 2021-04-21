/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CBase64Filter.h>
#include <core/CJsonStatePersistInserter.h>
#include <core/CPackedBitVector.h>

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
#include <sstream>
#include <string>

BOOST_AUTO_TEST_SUITE(CDataSummarizationJsonSerializerTest)

using namespace ml;

namespace {
using TDoubleVec = std::vector<double>;
using TStrVec = std::vector<std::string>;
using TStrVecVec = std::vector<TStrVec>;
using TFilteredInput = boost::iostreams::filtering_stream<boost::iostreams::input>;
using TLossFunctionType = maths::boosted_tree::ELossType;
using TRowItr = core::CDataFrame::TRowItr;

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
        api::CRetrainableModelJsonDeserializer::dataSummarizationFromJsonStream(
            std::make_shared<std::istringstream>(dataSummarizationStr));
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

BOOST_AUTO_TEST_CASE(testDeserialization) {
    const TStrVec columnNames{"x1", "x2", "x3", "x4", "x5", "y"};
    const TStrVec categoricalColumns{"x1", "x2", "x5"};
    const TStrVecVec rows{{"a", "b", "1.0", "1.0", "cat", "-1.0"},
                          {"a", "b", "1.0", "1.0", "cat", "-0.5"},
                          {"a", "b", "5.0", "0.0", "dog", "-0.1"},
                          {"c", "d", "5.0", "0.0", "dog", "1.0"},
                          {"e", "f", "5.0", "0.0", "dog", "1.5"}};
    auto expectedFrame = core::makeMainStorageDataFrame(columnNames.size()).first;
    expectedFrame->columnNames(columnNames);
    expectedFrame->categoricalColumns(categoricalColumns);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        expectedFrame->parseAndWriteRow(
            core::CVectorRange<const TStrVec>(rows[i], 0, rows[i].size()));
    }
    expectedFrame->finishWritingRows();

    // create encoder and serialize it
    maths::CDataFrameCategoryEncoder expectedEncoder({1, *expectedFrame, 5});
    std::stringstream persistedEncoderStream;
    {
        core::CJsonStatePersistInserter inserter{persistedEncoderStream};
        expectedEncoder.acceptPersistInserter(inserter);
    }

    api::CDataSummarizationJsonSerializer serializer{
        *expectedFrame, core::CPackedBitVector(expectedFrame->numberRows(), true),
        std::move(persistedEncoderStream)};
    auto istream = std::make_shared<std::istringstream>(serializer.jsonString());
    api::CRetrainableModelJsonDeserializer::TDataSummarization dataSummarization{
        api::CRetrainableModelJsonDeserializer::dataSummarizationFromJsonStream(istream)};
    BOOST_REQUIRE(dataSummarization.first && dataSummarization.second);
    BOOST_REQUIRE(expectedFrame->checksum() == dataSummarization.first->checksum());
    BOOST_REQUIRE(dataSummarization.second->numberInputColumns() ==
                  expectedFrame->numberColumns());
    BOOST_REQUIRE(dataSummarization.second->numberEncodedColumns() > 0);
}

BOOST_AUTO_TEST_SUITE_END()
