/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <core/CBase64Filter.h>
#include <core/CContainerPrinter.h>
#include <core/CDataFrame.h>
#include <core/CJsonStatePersistInserter.h>
#include <core/CPackedBitVector.h>
#include <core/CVectorRange.h>

#include <maths/analytics/CBoostedTreeLoss.h>

#include <api/CDataFrameAnalyzer.h>
#include <api/CDataSummarizationJsonWriter.h>
#include <api/CRetrainableModelJsonReader.h>

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
using TRowItr = core::CDataFrame::TRowItr;
using TFilteredInput = boost::iostreams::filtering_stream<boost::iostreams::input>;
using TLossFunctionType = maths::analytics::boosted_tree::ELossType;
using TDataFrameUPtrTemporaryDirectoryPtrPr =
    test::CDataFrameAnalysisSpecificationFactory::TDataFrameUPtrTemporaryDirectoryPtrPr;

std::stringstream decompressStream(std::stringstream compressedStream) {
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
    specFactory.rows(numberExamples).columns(cols).memoryLimit(30000000).dataSummarizationFraction(0.1);
    std::string analysisType;
    if (lossType == ml::maths::analytics::boosted_tree::E_BinaryClassification) {
        specFactory.predictionCategoricalFieldNames({"categorical_col", "target_col"});
        analysisType = test::CDataFrameAnalysisSpecificationFactory::classification();
    } else {
        specFactory.predictionCategoricalFieldNames({"categorical_col"});
        analysisType = test::CDataFrameAnalysisSpecificationFactory::regression();
    }
    TDataFrameUPtrTemporaryDirectoryPtrPr frameAndDirectory;
    auto spec = specFactory.predictionSpec(analysisType, "target_col", &frameAndDirectory);
    api::CDataFrameAnalyzer analyzer{std::move(spec), std::move(frameAndDirectory),
                                     std::move(outputWriterFactory)};

    TStrVec fieldNames{"numeric_col", "categorical_col", "target_col", ".", "."};
    TStrVec fieldValues{"", "", "0", "", ""};
    test::CDataFrameAnalyzerTrainingFactory::addPredictionTestData(
        lossType, fieldNames, fieldValues, analyzer, numberExamples);
    analyzer.handleRecord(fieldNames, {"", "", "", "", "$"});
    auto analysisRunner = analyzer.runner();

    auto dataSummarization = analysisRunner->dataSummarization();
    BOOST_TEST_REQUIRE(dataSummarization.get() != nullptr);

    // Verify compressed definition.
    {
        auto frame = core::makeMainStorageDataFrame(cols).first;
        std::string dataSummarizationStr{dataSummarization->jsonString()};
        std::stringstream decompressedStream{
            decompressStream(dataSummarization->jsonCompressedStream())};
        api::CRetrainableModelJsonReader::dataSummarizationFromJsonStream(
            std::make_shared<std::istringstream>(dataSummarizationStr), *frame);
        BOOST_TEST_REQUIRE(decompressedStream.str() == dataSummarizationStr);
    }

    // Verify json schema.
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
    for (const auto& row : rows) {
        expectedFrame->parseAndWriteRow(core::make_const_range(row, 0, row.size()));
    }
    expectedFrame->finishWritingRows();

    // create encoder and serialize it
    maths::analytics::CDataFrameCategoryEncoder expectedEncoder({1, *expectedFrame, 5});

    api::CDataSummarizationJsonWriter writer{
        *expectedFrame, core::CPackedBitVector(expectedFrame->numberRows(), true),
        columnNames.size(), expectedEncoder};
    auto istream = std::make_shared<std::istringstream>(writer.jsonString());
    auto actualFrame = core::makeMainStorageDataFrame(columnNames.size()).first;
    auto[encoder, encodingIndices] =
        api::CRetrainableModelJsonReader::dataSummarizationFromJsonStream(istream, *actualFrame);
    BOOST_REQUIRE(encoder != nullptr);
    BOOST_REQUIRE(expectedFrame->checksum() == actualFrame->checksum());
    BOOST_REQUIRE(encoder->numberInputColumns() == expectedFrame->numberColumns());
    BOOST_REQUIRE(encoder->numberEncodedColumns() > 0);
    BOOST_REQUIRE_EQUAL("[(y, 0)]", core::CContainerPrinter::print(encodingIndices));
}

BOOST_AUTO_TEST_SUITE_END()
