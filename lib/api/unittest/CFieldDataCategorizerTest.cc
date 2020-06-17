/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CDataAdder.h>
#include <core/CDataSearcher.h>
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CStringUtils.h>

#include <model/CLimits.h>

#include <api/CFieldConfig.h>
#include <api/CJsonOutputWriter.h>
#include <api/CNullOutput.h>
#include <api/COutputChainer.h>
#include <api/COutputHandler.h>

#include "CMockDataProcessor.h"
#include "CTestFieldDataCategorizer.h"

#include <boost/test/unit_test.hpp>

#include <set>
#include <sstream>

BOOST_AUTO_TEST_SUITE(CFieldDataCategorizerTest)

using namespace ml;
using namespace api;

namespace {

//! \brief
//! Mock object for state restore unit tests.
//!
//! DESCRIPTION:\n
//! CDataSearcher that returns an empty stream.
//!
class CEmptySearcher : public ml::core::CDataSearcher {
public:
    //! Do a search that results in an empty input stream.
    virtual TIStreamP search(size_t /*currentDocNum*/, size_t /*limit*/) {
        return TIStreamP(new std::istringstream());
    }
};

class CTestOutputHandler : public COutputHandler {
public:
    using TIntSet = std::set<int>;

public:
    virtual void finalise() { m_Finalised = true; }

    bool hasFinalised() const { return m_Finalised; }

    virtual void newOutputStream() { m_NewStream = true; }

    bool isNewStream() const { return m_NewStream; }

    virtual bool fieldNames(const TStrVec& /*fieldNames*/, const TStrVec& /*extraFieldNames*/) {
        return true;
    }

    virtual bool writeRow(const TStrStrUMap& dataRowFields,
                          const TStrStrUMap& overrideDataRowFields,
                          TOptionalTime /*time*/) {
        ++m_NumRows;
        std::string categoryIdStr;
        auto iter = overrideDataRowFields.find("mlcategory");
        if (iter != overrideDataRowFields.end()) {
            categoryIdStr = iter->second;
        } else {
            iter = dataRowFields.find("mlcategory");
            if (iter != dataRowFields.end()) {
                categoryIdStr = iter->second;
            }
        }
        int categoryId{0};
        if (categoryIdStr.empty() == false &&
            core::CStringUtils::stringToType(categoryIdStr, categoryId) != false &&
            categoryId > 0) {
            m_CategoryIdsHandled.insert(categoryId);
        }
        return true;
    }

    uint64_t numRows() const { return m_NumRows; }

    const TIntSet& categoryIdsHandled() const { return m_CategoryIdsHandled; }

private:
    bool m_NewStream = false;

    bool m_Finalised = false;

    uint64_t m_NumRows = 0;

    TIntSet m_CategoryIdsHandled;
};

class CTestDataSearcher : public core::CDataSearcher {
public:
    CTestDataSearcher(const std::string& data)
        : m_Stream(new std::istringstream(data)) {}

    virtual TIStreamP search(size_t /*currentDocNum*/, size_t /*limit*/) {
        return m_Stream;
    }

private:
    TIStreamP m_Stream;
};

class CTestDataAdder : public core::CDataAdder {
public:
    CTestDataAdder() : m_Stream(new std::ostringstream) {}

    virtual TOStreamP addStreamed(const std::string& /*index*/, const std::string& /*id*/) {
        return m_Stream;
    }

    virtual bool streamComplete(TOStreamP& /*strm*/, bool /*force*/) {
        return true;
    }

    TOStreamP getStream() { return m_Stream; }

private:
    TOStreamP m_Stream;
};
}

BOOST_AUTO_TEST_CASE(testWithoutPerPartitionCategorization) {
    model::CLimits limits;
    CFieldConfig config;
    BOOST_TEST_REQUIRE(config.initFromFile("testfiles/new_persist_categorization.conf"));
    CTestOutputHandler handler;

    std::ostringstream outputStrm;
    core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
    CJsonOutputWriter writer("job", wrappedOutputStream);

    CTestFieldDataCategorizer categorizer("job", config, limits, handler, writer);
    BOOST_REQUIRE_EQUAL(false, handler.isNewStream());
    categorizer.newOutputStream();
    BOOST_REQUIRE_EQUAL(true, handler.isNewStream());

    BOOST_REQUIRE_EQUAL(false, handler.hasFinalised());
    BOOST_REQUIRE_EQUAL(uint64_t(0), categorizer.numRecordsHandled());

    CFieldDataCategorizer::TStrStrUMap dataRowFields;
    dataRowFields["message"] = "thing";
    dataRowFields["two"] = "other";

    BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

    BOOST_REQUIRE_EQUAL(uint64_t(1), categorizer.numRecordsHandled());
    BOOST_REQUIRE_EQUAL(categorizer.numRecordsHandled(), handler.numRows());
    BOOST_REQUIRE_EQUAL("[1]", core::CContainerPrinter::print(handler.categoryIdsHandled()));

    // try a couple of erroneous cases
    dataRowFields.clear();
    BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

    dataRowFields["thing"] = "bling";
    dataRowFields["thang"] = "wing";
    BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

    dataRowFields["message"] = "";
    BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

    BOOST_REQUIRE_EQUAL(uint64_t(4), categorizer.numRecordsHandled());
    BOOST_REQUIRE_EQUAL(categorizer.numRecordsHandled(), handler.numRows());
    // Still only 1, as all the other input was invalid
    BOOST_REQUIRE_EQUAL("[1]", core::CContainerPrinter::print(handler.categoryIdsHandled()));

    dataRowFields["message"] = "and another thing";
    BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

    BOOST_REQUIRE_EQUAL(uint64_t(5), categorizer.numRecordsHandled());
    BOOST_REQUIRE_EQUAL(categorizer.numRecordsHandled(), handler.numRows());
    BOOST_REQUIRE_EQUAL(
        "[1, 2]", core::CContainerPrinter::print(handler.categoryIdsHandled()));

    categorizer.finalise();
    BOOST_TEST_REQUIRE(handler.hasFinalised());

    // do a persist / restore
    std::string origJson;
    {
        CTestDataAdder adder;
        categorizer.persistStateInForeground(adder, "");
        std::ostringstream& ss = dynamic_cast<std::ostringstream&>(*adder.getStream());
        origJson = ss.str();
    }

    std::string newJson;
    LOG_DEBUG(<< "origJson = " << origJson);
    {
        model::CLimits limits2;
        CTestOutputHandler handler2;
        std::ostringstream outputStrm2;
        core::CJsonOutputStreamWrapper wrappedOutputStream2(outputStrm2);
        CJsonOutputWriter writer2("job", wrappedOutputStream2);

        CTestFieldDataCategorizer newCategorizer("job", config, limits2, handler2, writer2);
        CTestDataSearcher restorer(origJson);
        core_t::TTime time = 0;
        newCategorizer.restoreState(restorer, time);

        CTestDataAdder adder;
        newCategorizer.persistStateInForeground(adder, "");
        std::ostringstream& ss = dynamic_cast<std::ostringstream&>(*adder.getStream());
        newJson = ss.str();
    }
    BOOST_REQUIRE_EQUAL(origJson, newJson);
}

BOOST_AUTO_TEST_CASE(testWithPerPartitionCategorization) {
    model::CLimits limits;
    CFieldConfig config;
    BOOST_TEST_REQUIRE(config.initFromFile("testfiles/new_persist_per_partition_categorization.conf"));
    CTestOutputHandler handler;

    std::ostringstream outputStrm;
    core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
    CJsonOutputWriter writer("job", wrappedOutputStream);

    CTestFieldDataCategorizer categorizer("job", config, limits, handler, writer);
    BOOST_REQUIRE_EQUAL(false, handler.isNewStream());
    categorizer.newOutputStream();
    BOOST_REQUIRE_EQUAL(true, handler.isNewStream());

    BOOST_REQUIRE_EQUAL(false, handler.hasFinalised());
    BOOST_REQUIRE_EQUAL(uint64_t(0), categorizer.numRecordsHandled());

    CFieldDataCategorizer::TStrStrUMap dataRowFields;
    dataRowFields["message"] = "thing";
    dataRowFields["event.dataset"] = "elasticsearch";
    dataRowFields["two"] = "other";

    BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

    BOOST_REQUIRE_EQUAL(uint64_t(1), categorizer.numRecordsHandled());
    BOOST_REQUIRE_EQUAL(categorizer.numRecordsHandled(), handler.numRows());
    BOOST_REQUIRE_EQUAL("[1]", core::CContainerPrinter::print(handler.categoryIdsHandled()));

    dataRowFields["event.dataset"] = "kibana";
    BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

    BOOST_REQUIRE_EQUAL(uint64_t(2), categorizer.numRecordsHandled());
    BOOST_REQUIRE_EQUAL(categorizer.numRecordsHandled(), handler.numRows());
    // Now two categories, because even though message was identical, the
    // partition was different
    BOOST_REQUIRE_EQUAL(
        "[1, 2]", core::CContainerPrinter::print(handler.categoryIdsHandled()));

    // try a couple of erroneous cases
    dataRowFields.clear();
    BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

    dataRowFields["thing"] = "bling";
    dataRowFields["thang"] = "wing";
    BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

    dataRowFields["message"] = "";
    BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

    BOOST_REQUIRE_EQUAL(uint64_t(5), categorizer.numRecordsHandled());
    BOOST_REQUIRE_EQUAL(categorizer.numRecordsHandled(), handler.numRows());
    // Still only 2, as all the other input was invalid
    BOOST_REQUIRE_EQUAL(
        "[1, 2]", core::CContainerPrinter::print(handler.categoryIdsHandled()));

    dataRowFields["message"] = "and another thing";
    dataRowFields["event.dataset"] = "elasticsearch";
    BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

    BOOST_REQUIRE_EQUAL(uint64_t(6), categorizer.numRecordsHandled());
    BOOST_REQUIRE_EQUAL(categorizer.numRecordsHandled(), handler.numRows());
    BOOST_REQUIRE_EQUAL(
        "[1, 2, 3]", core::CContainerPrinter::print(handler.categoryIdsHandled()));

    BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

    BOOST_REQUIRE_EQUAL(uint64_t(7), categorizer.numRecordsHandled());
    BOOST_REQUIRE_EQUAL(categorizer.numRecordsHandled(), handler.numRows());
    // Still 3, as the message and partition were identical so won't have
    // created a new category
    BOOST_REQUIRE_EQUAL(
        "[1, 2, 3]", core::CContainerPrinter::print(handler.categoryIdsHandled()));

    dataRowFields["event.dataset"] = "kibana";
    BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

    BOOST_REQUIRE_EQUAL(uint64_t(8), categorizer.numRecordsHandled());
    BOOST_REQUIRE_EQUAL(categorizer.numRecordsHandled(), handler.numRows());
    // Now 4, as the message was the same but the partition changed
    BOOST_REQUIRE_EQUAL("[1, 2, 3, 4]",
                        core::CContainerPrinter::print(handler.categoryIdsHandled()));

    categorizer.finalise();
    BOOST_TEST_REQUIRE(handler.hasFinalised());

    // do a persist / restore
    std::string origJson;
    {
        CTestDataAdder adder;
        categorizer.persistStateInForeground(adder, "");
        std::ostringstream& ss = dynamic_cast<std::ostringstream&>(*adder.getStream());
        origJson = ss.str();
    }

    std::string newJson;
    LOG_DEBUG(<< "origJson = " << origJson);
    {
        model::CLimits limits2;
        CTestOutputHandler handler2;
        std::ostringstream outputStrm2;
        core::CJsonOutputStreamWrapper wrappedOutputStream2(outputStrm2);
        CJsonOutputWriter writer2("job", wrappedOutputStream2);

        CTestFieldDataCategorizer newCategorizer("job", config, limits2, handler2, writer2);
        CTestDataSearcher restorer(origJson);
        core_t::TTime time = 0;
        newCategorizer.restoreState(restorer, time);

        CTestDataAdder adder;
        newCategorizer.persistStateInForeground(adder, "");
        std::ostringstream& ss = dynamic_cast<std::ostringstream&>(*adder.getStream());
        newJson = ss.str();
    }
    BOOST_REQUIRE_EQUAL(origJson, newJson);
}

BOOST_AUTO_TEST_CASE(testNodeReverseSearch) {
    model::CLimits limits;
    CFieldConfig config;
    BOOST_TEST_REQUIRE(config.initFromFile("testfiles/new_persist_categorization.conf"));

    std::ostringstream outputStrm;
    {
        CNullOutput nullOutput;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
        CJsonOutputWriter writer("job", wrappedOutputStream);

        CTestFieldDataCategorizer categorizer("job", config, limits, nullOutput, writer);

        CFieldDataCategorizer::TStrStrUMap dataRowFields;
        dataRowFields["message"] = "Node 1 started";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        dataRowFields["message"] = "Node 2 started";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        categorizer.finalise();
    }

    const std::string& output = outputStrm.str();
    LOG_DEBUG(<< "Output is: " << output);

    // Assert that the reverse search contains all expected tokens when
    // categorization is run end-to-end (obviously computation of categories and
    // reverse search creation are tested more thoroughly in the unit tests for
    // their respective classes, but this test helps to confirm that they work
    // together)
    BOOST_TEST_REQUIRE(output.find("\"terms\":\"Node started\"") != std::string::npos);
    BOOST_TEST_REQUIRE(output.find("\"regex\":\".*?Node.+?started.*\"") != std::string::npos);
    // The input data should NOT be in the output
    BOOST_TEST_REQUIRE(output.find("\"message\"") == std::string::npos);
}

BOOST_AUTO_TEST_CASE(testJobKilledReverseSearch) {
    model::CLimits limits;
    CFieldConfig config;
    BOOST_TEST_REQUIRE(config.initFromFile("testfiles/new_persist_categorization.conf"));

    std::ostringstream outputStrm;
    {
        CNullOutput nullOutput;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
        CJsonOutputWriter writer("job", wrappedOutputStream);

        CTestFieldDataCategorizer categorizer("job", config, limits, nullOutput, writer);

        CFieldDataCategorizer::TStrStrUMap dataRowFields;
        dataRowFields["message"] = "[count_tweets] Killing job";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        dataRowFields["message"] = "Killing job [count_tweets]";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        dataRowFields["message"] = "[tweets_by_location] Killing job";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        dataRowFields["message"] = "Killing job [tweets_by_location]";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        categorizer.finalise();
    }

    const std::string& output = outputStrm.str();
    LOG_DEBUG(<< "Output is: " << output);

    // Assert that the reverse search contains all expected tokens when
    // categorization is run end-to-end (obviously computation of categories and
    // reverse search creation are tested more thoroughly in the unit tests for
    // their respective classes, but this test helps to confirm that they work
    // together)
    BOOST_TEST_REQUIRE(output.find("\"terms\":\"Killing job\"") != std::string::npos);
    BOOST_TEST_REQUIRE(output.find("\"regex\":\".*?Killing.+?job.*\"") != std::string::npos);
    // The input data should NOT be in the output
    BOOST_TEST_REQUIRE(output.find("\"message\"") == std::string::npos);
}

BOOST_AUTO_TEST_CASE(testPassOnControlMessages) {
    model::CLimits limits;
    CFieldConfig config;
    BOOST_TEST_REQUIRE(config.initFromFile("testfiles/new_persist_categorization.conf"));

    std::ostringstream outputStrm;
    {
        CNullOutput nullOutput;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
        CJsonOutputWriter writer("job", wrappedOutputStream);

        CMockDataProcessor mockProcessor(nullOutput);
        COutputChainer outputChainer(mockProcessor);
        CTestFieldDataCategorizer categorizer("job", config, limits, outputChainer, writer);

        CFieldDataCategorizer::TStrStrUMap dataRowFields;
        dataRowFields["."] = "f7";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        categorizer.finalise();
    }

    const std::string& output = outputStrm.str();
    LOG_DEBUG(<< "Output is: " << output);
    BOOST_REQUIRE_EQUAL(std::string("[]"), output);
}

BOOST_AUTO_TEST_CASE(testHandleControlMessages) {
    model::CLimits limits;
    CFieldConfig config;
    BOOST_TEST_REQUIRE(config.initFromFile("testfiles/new_persist_categorization.conf"));

    std::ostringstream outputStrm;
    {
        CNullOutput nullOutput;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
        CJsonOutputWriter writer("job", wrappedOutputStream);

        CTestFieldDataCategorizer categorizer("job", config, limits, nullOutput,
                                              writer, nullptr);

        CFieldDataCategorizer::TStrStrUMap dataRowFields;
        dataRowFields["."] = "f7";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        categorizer.finalise();
    }

    const std::string& output = outputStrm.str();
    LOG_DEBUG(<< "Output is: " << output);
    BOOST_REQUIRE_EQUAL(std::string::size_type(0),
                        output.find("[{\"flush\":{\"id\":\"7\",\"last_finalized_bucket_end\":0}}"));
}

BOOST_AUTO_TEST_CASE(testRestoreStateFailsWithEmptyState) {
    model::CLimits limits;
    CFieldConfig config;
    BOOST_TEST_REQUIRE(config.initFromFile("testfiles/new_persist_categorization.conf"));

    std::ostringstream outputStrm;
    CNullOutput nullOutput;
    core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
    CJsonOutputWriter writer("job", wrappedOutputStream);
    CTestFieldDataCategorizer categorizer("job", config, limits, nullOutput, writer, nullptr);

    core_t::TTime completeToTime(0);
    CEmptySearcher restoreSearcher;
    BOOST_TEST_REQUIRE(categorizer.restoreState(restoreSearcher, completeToTime) == false);
}

BOOST_AUTO_TEST_CASE(flushWritesOnlyChangedCategories) {
    model::CLimits limits;
    CFieldConfig config;
    BOOST_TEST_REQUIRE(config.initFromFile("testfiles/new_persist_categorization.conf"));

    std::ostringstream outputStrm;
    {
        CNullOutput nullOutput;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
        CJsonOutputWriter writer("job", wrappedOutputStream);

        CTestFieldDataCategorizer categorizer("job", config, limits, nullOutput, writer);

        CFieldDataCategorizer::TStrStrUMap dataRowFields;
        dataRowFields["message"] = "Node 1 started";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        dataRowFields["message"] = "Node 2 started";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        dataRowFields["message"] = "Somethingelse my message";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        CFieldDataCategorizer::TStrStrUMap flush;
        flush["."] = "f42";

        //! should write to the output buffer and the num_matches will end up being 2
        BOOST_TEST_REQUIRE(categorizer.handleRecord(flush));

        dataRowFields["message"] = "Node 2 started";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        BOOST_TEST_REQUIRE(categorizer.handleRecord(flush));
    }
    const std::string& output = outputStrm.str();
    LOG_DEBUG(<< "Output is: " << output);

    auto findOccurrences = [](const std::string& str, const std::string& substr) {
        int occurrences = 0;
        std::string::size_type start = 0;
        while ((start = str.find(substr, start)) != std::string::npos) {
            ++occurrences;
            start += substr.length();
        }
        return occurrences;
    };
    //! Output should have category_id 1 3 times. 2 for the first two calls, and one for the flush
    BOOST_REQUIRE_EQUAL(findOccurrences(output, "\"category_id\":1"), 3);

    //! Output should only have the initial persistence as it did not change after the flush
    BOOST_REQUIRE_EQUAL(findOccurrences(output, "\"category_id\":2"), 1);
}

BOOST_AUTO_TEST_CASE(finalizeWritesOnlyChangedCategories) {
    model::CLimits limits;
    CFieldConfig config;
    BOOST_TEST_REQUIRE(config.initFromFile("testfiles/new_persist_categorization.conf"));

    std::ostringstream outputStrm;
    {
        CNullOutput nullOutput;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
        CJsonOutputWriter writer("job", wrappedOutputStream);

        CTestFieldDataCategorizer categorizer("job", config, limits, nullOutput, writer);

        CFieldDataCategorizer::TStrStrUMap dataRowFields;
        dataRowFields["message"] = "Node 1 started";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        dataRowFields["message"] = "Node 2 started";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        dataRowFields["message"] = "Somethingelse my message";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));
        categorizer.finalise();

        dataRowFields["message"] = "Node 2 started";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        categorizer.finalise();
    }
    auto findOccurrences = [](const std::string& str, const std::string& substr) {
        int occurrences = 0;
        std::string::size_type start = 0;
        while ((start = str.find(substr, start)) != std::string::npos) {
            ++occurrences;
            start += substr.length();
        }
        return occurrences;
    };
    const std::string& output = outputStrm.str();
    LOG_DEBUG(<< "Output is: " << output);
    //! Output should have category_id 1 3 times. 2 for the first two calls, and one for the finalize
    BOOST_REQUIRE_EQUAL(findOccurrences(output, "\"category_id\":1"), 3);

    //! Output should only have the initial persistence as it did not change after the finalize
    BOOST_REQUIRE_EQUAL(findOccurrences(output, "\"category_id\":2"), 1);
}

BOOST_AUTO_TEST_SUITE_END()
