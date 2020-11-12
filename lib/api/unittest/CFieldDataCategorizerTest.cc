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
#include <api/CSimpleOutputWriter.h>

#include "CTestFieldDataCategorizer.h"

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <set>
#include <sstream>
#include <tuple>

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
    TIStreamP search(size_t /*currentDocNum*/, size_t /*limit*/) override {
        return TIStreamP(new std::istringstream());
    }
};

class CTestChainedProcessor : public CDataProcessor {
public:
    using TIntSet = std::set<int>;

public:
    void finalise() override { m_Finalised = true; }

    bool hasFinalised() const { return m_Finalised; }

    bool handleRecord(const TStrStrUMap& dataRowFields, TOptionalTime /*time*/) override {
        auto iter = dataRowFields.find(".");
        if (iter != dataRowFields.end() && iter->second.empty() == false) {
            ++m_NumControlMessages;
            return true;
        }
        std::string categoryIdStr;
        iter = dataRowFields.find("mlcategory");
        if (iter != dataRowFields.end()) {
            categoryIdStr = iter->second;
        }
        int categoryId{0};
        if (categoryIdStr.empty() == false &&
            core::CStringUtils::stringToType(categoryIdStr, categoryId) != false &&
            categoryId > 0) {
            m_CategoryIdsHandled.insert(categoryId);
        }
        ++m_NumRecordsHandled;
        return true;
    }

    bool restoreState(core::CDataSearcher& /*restoreSearcher*/,
                      core_t::TTime& /*completeToTime*/) override {
        return true;
    }

    bool persistStateInForeground(core::CDataAdder& /*persister*/,
                                  const std::string& /*descriptionPrefix*/) override {
        return true;
    }

    std::uint64_t numRecordsHandled() const override {
        return m_NumRecordsHandled;
    }

    std::uint64_t numControlMessagesHandled() const {
        return m_NumControlMessages;
    }

    bool isPersistenceNeeded(const std::string& /*description*/) const override {
        return false;
    }

    const TIntSet& categoryIdsHandled() const { return m_CategoryIdsHandled; }

private:
    bool m_Finalised = false;

    std::uint64_t m_NumRecordsHandled = 0;
    std::uint64_t m_NumControlMessages = 0;

    TIntSet m_CategoryIdsHandled;
};

class CTestDataSearcher : public core::CDataSearcher {
public:
    CTestDataSearcher(const std::string& data)
        : m_Stream(new std::istringstream(data)) {}

    TIStreamP search(size_t /*currentDocNum*/, size_t /*limit*/) override {
        return m_Stream;
    }

private:
    TIStreamP m_Stream;
};

class CTestDataAdder : public core::CDataAdder {
public:
    CTestDataAdder() : m_Stream(new std::ostringstream) {}

    TOStreamP addStreamed(const std::string& /*id*/) override {
        return m_Stream;
    }

    bool streamComplete(TOStreamP& /*strm*/, bool /*force*/) override {
        return true;
    }

    TOStreamP getStream() { return m_Stream; }

private:
    TOStreamP m_Stream;
};

std::size_t countOccurrences(const std::string& str, const std::string& substr) {
    std::size_t occurrences{0};
    for (std::size_t start = str.find(substr); start != std::string::npos;
         start = str.find(substr, start + substr.length())) {
        ++occurrences;
    }
    return occurrences;
}

std::string setupPerPartitionStopOnWarnTest(bool stopOnWarnAtInit,
                                            bool sendControlMessageAfter50,
                                            bool controlMessageContent) {
    model::CLimits limits;
    CFieldConfig config;
    BOOST_TEST_REQUIRE(config.initFromFile("testfiles/new_persist_per_partition_categorization.conf"));

    std::ostringstream outputStrm;
    {
        core::CJsonOutputStreamWrapper wrappedOutputStream{outputStrm};

        CTestFieldDataCategorizer categorizer{
            "job",   config,          limits, nullptr, wrappedOutputStream,
            nullptr, stopOnWarnAtInit};

        CFieldDataCategorizer::TStrStrUMap dataRowFieldsPartition1;
        categorizer.registerMutableField(
            CFieldDataCategorizer::MLCATEGORY_NAME,
            dataRowFieldsPartition1[CFieldDataCategorizer::MLCATEGORY_NAME]);
        dataRowFieldsPartition1["event.dataset"] = "nodes";
        dataRowFieldsPartition1["message"] = "Node 1 started";

        CFieldDataCategorizer::TStrStrUMap dataRowFieldsPartition2;
        dataRowFieldsPartition2["event.dataset"] = "random but changing";
        dataRowFieldsPartition2["message"] = "fff fff fff";

        // The 100th message in partition "nodes" should cause the
        // categorization status to change to "warn" for that partition.  This
        // is because there is only one category, which is a red flag, but we
        // don't report warning status until we've seen 100 messages.  Then we
        // should stop categorizing in partition "nodes" if stop-on-warn is in
        // force at that time.  However, since the "random but changing"
        // partition does not enter "warn" status, categorization should
        // continue for it.
        for (std::size_t count = 1; count <= 200; ++count) {
            BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFieldsPartition1));
            if (count % 10 == 1) {
                std::string& message{dataRowFieldsPartition2["message"]};
                char oldChar{message[0]};
                char newChar{static_cast<char>(oldChar + 1)};
                std::replace(message.begin(), message.end(), oldChar, newChar);
            }
            BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFieldsPartition2));
            if (count == 50 && sendControlMessageAfter50) {
                CFieldDataCategorizer::TStrStrUMap control;
                control["."] = "c" + ml::core::CStringUtils::typeToString(controlMessageContent);
                BOOST_TEST_REQUIRE(categorizer.handleRecord(control));
            }
        }

        categorizer.finalise();
    }
    return outputStrm.str();
}
}

BOOST_AUTO_TEST_CASE(testWithoutPerPartitionCategorization) {
    model::CLimits limits;
    CFieldConfig config;
    BOOST_TEST_REQUIRE(config.initFromFile("testfiles/new_persist_categorization.conf"));
    CTestChainedProcessor testChainedProcessor;

    std::ostringstream outputStrm;
    core::CJsonOutputStreamWrapper wrappedOutputStream{outputStrm};

    CTestFieldDataCategorizer categorizer{"job", config, limits, &testChainedProcessor,
                                          wrappedOutputStream};

    BOOST_REQUIRE_EQUAL(false, testChainedProcessor.hasFinalised());
    BOOST_REQUIRE_EQUAL(0, categorizer.numRecordsHandled());

    CFieldDataCategorizer::TStrStrUMap dataRowFields;
    categorizer.registerMutableField(CFieldDataCategorizer::MLCATEGORY_NAME,
                                     dataRowFields[CFieldDataCategorizer::MLCATEGORY_NAME]);
    dataRowFields["message"] = "thing";
    dataRowFields["two"] = "other";

    BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

    BOOST_REQUIRE_EQUAL(1, categorizer.numRecordsHandled());
    BOOST_REQUIRE_EQUAL(categorizer.numRecordsHandled(),
                        testChainedProcessor.numRecordsHandled());
    BOOST_REQUIRE_EQUAL("[1]", core::CContainerPrinter::print(
                                   testChainedProcessor.categoryIdsHandled()));

    // try a couple of erroneous cases
    dataRowFields.clear();
    categorizer.registerMutableField(CFieldDataCategorizer::MLCATEGORY_NAME,
                                     dataRowFields[CFieldDataCategorizer::MLCATEGORY_NAME]);
    BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

    dataRowFields["thing"] = "bling";
    dataRowFields["thang"] = "wing";
    BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

    dataRowFields["message"] = "";
    BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

    BOOST_REQUIRE_EQUAL(4, categorizer.numRecordsHandled());
    BOOST_REQUIRE_EQUAL(categorizer.numRecordsHandled(),
                        testChainedProcessor.numRecordsHandled());
    // Still only 1, as all the other input was invalid
    BOOST_REQUIRE_EQUAL("[1]", core::CContainerPrinter::print(
                                   testChainedProcessor.categoryIdsHandled()));

    dataRowFields["message"] = "and another thing";
    BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

    BOOST_REQUIRE_EQUAL(5, categorizer.numRecordsHandled());
    BOOST_REQUIRE_EQUAL(categorizer.numRecordsHandled(),
                        testChainedProcessor.numRecordsHandled());
    BOOST_REQUIRE_EQUAL("[1, 2]", core::CContainerPrinter::print(
                                      testChainedProcessor.categoryIdsHandled()));

    categorizer.finalise();
    BOOST_TEST_REQUIRE(testChainedProcessor.hasFinalised());

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
        std::ostringstream outputStrm2;
        core::CJsonOutputStreamWrapper wrappedOutputStream2{outputStrm2};

        CTestFieldDataCategorizer newCategorizer{"job", config, limits2,
                                                 nullptr, wrappedOutputStream2};
        CTestDataSearcher restorer{origJson};
        core_t::TTime time{0};
        newCategorizer.restoreState(restorer, time);

        CTestDataAdder adder;
        newCategorizer.persistStateInForeground(adder, "");
        std::ostringstream& ss{dynamic_cast<std::ostringstream&>(*adder.getStream())};
        newJson = ss.str();
    }
    BOOST_REQUIRE_EQUAL(origJson, newJson);
}

BOOST_AUTO_TEST_CASE(testWithPerPartitionCategorization) {
    model::CLimits limits;
    CFieldConfig config;
    BOOST_TEST_REQUIRE(config.initFromFile("testfiles/new_persist_per_partition_categorization.conf"));
    CTestChainedProcessor testChainedProcessor;

    std::ostringstream outputStrm;
    core::CJsonOutputStreamWrapper wrappedOutputStream{outputStrm};

    CTestFieldDataCategorizer categorizer{"job", config, limits, &testChainedProcessor,
                                          wrappedOutputStream};

    BOOST_REQUIRE_EQUAL(false, testChainedProcessor.hasFinalised());
    BOOST_REQUIRE_EQUAL(0, categorizer.numRecordsHandled());

    CFieldDataCategorizer::TStrStrUMap dataRowFields;
    categorizer.registerMutableField(CFieldDataCategorizer::MLCATEGORY_NAME,
                                     dataRowFields[CFieldDataCategorizer::MLCATEGORY_NAME]);
    dataRowFields["message"] = "thing";
    dataRowFields["event.dataset"] = "elasticsearch";
    dataRowFields["two"] = "other";
    categorizer.registerMutableField(CFieldDataCategorizer::MLCATEGORY_NAME,
                                     dataRowFields[CFieldDataCategorizer::MLCATEGORY_NAME]);

    BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

    BOOST_REQUIRE_EQUAL(1, categorizer.numRecordsHandled());
    BOOST_REQUIRE_EQUAL(categorizer.numRecordsHandled(),
                        testChainedProcessor.numRecordsHandled());
    BOOST_REQUIRE_EQUAL("[1]", core::CContainerPrinter::print(
                                   testChainedProcessor.categoryIdsHandled()));

    dataRowFields["event.dataset"] = "kibana";
    BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

    BOOST_REQUIRE_EQUAL(2, categorizer.numRecordsHandled());
    BOOST_REQUIRE_EQUAL(categorizer.numRecordsHandled(),
                        testChainedProcessor.numRecordsHandled());
    // Now two categories, because even though message was identical, the
    // partition was different
    BOOST_REQUIRE_EQUAL("[1, 2]", core::CContainerPrinter::print(
                                      testChainedProcessor.categoryIdsHandled()));

    // try a couple of erroneous cases
    dataRowFields.clear();
    categorizer.registerMutableField(CFieldDataCategorizer::MLCATEGORY_NAME,
                                     dataRowFields[CFieldDataCategorizer::MLCATEGORY_NAME]);
    BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

    dataRowFields["thing"] = "bling";
    dataRowFields["thang"] = "wing";
    BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

    dataRowFields["message"] = "";
    BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

    BOOST_REQUIRE_EQUAL(5, categorizer.numRecordsHandled());
    BOOST_REQUIRE_EQUAL(categorizer.numRecordsHandled(),
                        testChainedProcessor.numRecordsHandled());
    // Still only 2, as all the other input was invalid
    BOOST_REQUIRE_EQUAL("[1, 2]", core::CContainerPrinter::print(
                                      testChainedProcessor.categoryIdsHandled()));

    dataRowFields["message"] = "and another thing";
    dataRowFields["event.dataset"] = "elasticsearch";
    BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

    BOOST_REQUIRE_EQUAL(6, categorizer.numRecordsHandled());
    BOOST_REQUIRE_EQUAL(categorizer.numRecordsHandled(),
                        testChainedProcessor.numRecordsHandled());
    BOOST_REQUIRE_EQUAL("[1, 2, 3]", core::CContainerPrinter::print(
                                         testChainedProcessor.categoryIdsHandled()));

    BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

    BOOST_REQUIRE_EQUAL(7, categorizer.numRecordsHandled());
    BOOST_REQUIRE_EQUAL(categorizer.numRecordsHandled(),
                        testChainedProcessor.numRecordsHandled());
    // Still 3, as the message and partition were identical so won't have
    // created a new category
    BOOST_REQUIRE_EQUAL("[1, 2, 3]", core::CContainerPrinter::print(
                                         testChainedProcessor.categoryIdsHandled()));

    dataRowFields["event.dataset"] = "kibana";
    BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

    BOOST_REQUIRE_EQUAL(8, categorizer.numRecordsHandled());
    BOOST_REQUIRE_EQUAL(categorizer.numRecordsHandled(),
                        testChainedProcessor.numRecordsHandled());
    // Now 4, as the message was the same but the partition changed
    BOOST_REQUIRE_EQUAL("[1, 2, 3, 4]", core::CContainerPrinter::print(
                                            testChainedProcessor.categoryIdsHandled()));

    categorizer.finalise();
    BOOST_TEST_REQUIRE(testChainedProcessor.hasFinalised());

    // do a persist / restore
    std::string origJson;
    {
        CTestDataAdder adder;
        categorizer.persistStateInForeground(adder, "");
        std::ostringstream& ss{dynamic_cast<std::ostringstream&>(*adder.getStream())};
        origJson = ss.str();
    }

    std::string newJson;
    LOG_DEBUG(<< "origJson = " << origJson);
    {
        model::CLimits limits2;
        std::ostringstream outputStrm2;
        core::CJsonOutputStreamWrapper wrappedOutputStream2{outputStrm2};

        CTestFieldDataCategorizer newCategorizer{"job", config, limits2,
                                                 nullptr, wrappedOutputStream2};
        CTestDataSearcher restorer{origJson};
        core_t::TTime time{0};
        newCategorizer.restoreState(restorer, time);

        CTestDataAdder adder;
        newCategorizer.persistStateInForeground(adder, "");
        std::ostringstream& ss{dynamic_cast<std::ostringstream&>(*adder.getStream())};
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
        core::CJsonOutputStreamWrapper wrappedOutputStream{outputStrm};

        CTestFieldDataCategorizer categorizer{"job", config, limits, nullptr, wrappedOutputStream};

        CFieldDataCategorizer::TStrStrUMap dataRowFields;
        categorizer.registerMutableField(CFieldDataCategorizer::MLCATEGORY_NAME,
                                         dataRowFields[CFieldDataCategorizer::MLCATEGORY_NAME]);
        dataRowFields["message"] = "Node 1 started";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        dataRowFields["message"] = "Node 2 started";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        categorizer.finalise();
    }

    const std::string& output{outputStrm.str()};
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
        core::CJsonOutputStreamWrapper wrappedOutputStream{outputStrm};

        CTestFieldDataCategorizer categorizer{"job", config, limits, nullptr, wrappedOutputStream};

        CFieldDataCategorizer::TStrStrUMap dataRowFields;
        categorizer.registerMutableField(CFieldDataCategorizer::MLCATEGORY_NAME,
                                         dataRowFields[CFieldDataCategorizer::MLCATEGORY_NAME]);
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

    const std::string& output{outputStrm.str()};
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

    CTestChainedProcessor testChainedProcessor;

    std::ostringstream outputStrm;
    {
        core::CJsonOutputStreamWrapper wrappedOutputStream{outputStrm};

        CTestFieldDataCategorizer categorizer{
            "job", config, limits, &testChainedProcessor, wrappedOutputStream};

        CFieldDataCategorizer::TStrStrUMap dataRowFields;
        categorizer.registerMutableField(CFieldDataCategorizer::MLCATEGORY_NAME,
                                         dataRowFields[CFieldDataCategorizer::MLCATEGORY_NAME]);
        dataRowFields["."] = "f7";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        categorizer.finalise();
    }

    const std::string& output{outputStrm.str()};
    LOG_DEBUG(<< "Output is: " << output);
    BOOST_REQUIRE_EQUAL("[]", output);

    BOOST_REQUIRE_EQUAL(0, testChainedProcessor.categoryIdsHandled().size());
    BOOST_REQUIRE_EQUAL(0, testChainedProcessor.numRecordsHandled());
    BOOST_REQUIRE_EQUAL(1, testChainedProcessor.numControlMessagesHandled());
    BOOST_REQUIRE_EQUAL(true, testChainedProcessor.hasFinalised());
}

BOOST_AUTO_TEST_CASE(testHandleControlMessages) {
    model::CLimits limits;
    CFieldConfig config;
    BOOST_TEST_REQUIRE(config.initFromFile("testfiles/new_persist_categorization.conf"));

    std::ostringstream outputStrm;
    {
        core::CJsonOutputStreamWrapper wrappedOutputStream{outputStrm};

        CTestFieldDataCategorizer categorizer{"job", config, limits, nullptr, wrappedOutputStream};

        CFieldDataCategorizer::TStrStrUMap dataRowFields;
        categorizer.registerMutableField(CFieldDataCategorizer::MLCATEGORY_NAME,
                                         dataRowFields[CFieldDataCategorizer::MLCATEGORY_NAME]);
        dataRowFields["."] = "f7";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        categorizer.finalise();
    }

    const std::string& output{outputStrm.str()};
    LOG_DEBUG(<< "Output is: " << output);
    BOOST_REQUIRE_EQUAL(0, output.find("[{\"flush\":{\"id\":\"7\",\"last_finalized_bucket_end\":0}}"));
}

BOOST_AUTO_TEST_CASE(testRestoreStateFailsWithEmptyState) {
    model::CLimits limits;
    CFieldConfig config;
    BOOST_TEST_REQUIRE(config.initFromFile("testfiles/new_persist_categorization.conf"));

    std::ostringstream outputStrm;
    core::CJsonOutputStreamWrapper wrappedOutputStream{outputStrm};
    CTestFieldDataCategorizer categorizer{"job", config, limits, nullptr, wrappedOutputStream};

    core_t::TTime completeToTime{0};
    CEmptySearcher restoreSearcher;
    BOOST_TEST_REQUIRE(categorizer.restoreState(restoreSearcher, completeToTime) == false);
}

BOOST_AUTO_TEST_CASE(testFlushWritesOnlyChangedCategories) {
    model::CLimits limits;
    CFieldConfig config;
    BOOST_TEST_REQUIRE(config.initFromFile("testfiles/new_persist_categorization.conf"));

    std::ostringstream outputStrm;
    {
        core::CJsonOutputStreamWrapper wrappedOutputStream{outputStrm};

        CTestFieldDataCategorizer categorizer{"job", config, limits, nullptr, wrappedOutputStream};

        CFieldDataCategorizer::TStrStrUMap dataRowFields;
        categorizer.registerMutableField(CFieldDataCategorizer::MLCATEGORY_NAME,
                                         dataRowFields[CFieldDataCategorizer::MLCATEGORY_NAME]);
        dataRowFields["message"] = "Node 1 started";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        dataRowFields["message"] = "Node 2 started";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        dataRowFields["message"] = "Somethingelse my message";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        CFieldDataCategorizer::TStrStrUMap flush;
        flush["."] = "f1";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(flush));

        dataRowFields["message"] = "Node 2 started";

        // This will not add a new example, as the message is exactly the
        // same as a previous example.  So the number of matches for
        // category_id 1 will increase to 3, but the category will not
        // immediately be updated.
        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        flush["."] = "f2";

        // Should write the updated category_id 1 to the output again with
        // num_matches: 3
        BOOST_TEST_REQUIRE(categorizer.handleRecord(flush));
    }
    const std::string& output{outputStrm.str()};
    LOG_DEBUG(<< "Output is: " << output);

    // Output should have a category definition for category_id 1 three times:
    // two for the first two calls, and one for the flush
    BOOST_REQUIRE_EQUAL(3, countOccurrences(output, "\"category_id\":1"));

    // Stats object on first flush should have been written when there were
    // a total of 3 categorized messages
    BOOST_REQUIRE_EQUAL(1, countOccurrences(output, "\"categorized_doc_count\":3"));

    // Output should only have the initial write of category definition for
    // category_id 2 as it did not change after the flush
    BOOST_REQUIRE_EQUAL(1, countOccurrences(output, "\"category_id\":2"));

    // Stats object on second flush should have been written when there were
    // a total of 4 categorized messages
    BOOST_REQUIRE_EQUAL(1, countOccurrences(output, "\"categorized_doc_count\":4"));
}

BOOST_AUTO_TEST_CASE(testFinalizeWritesOnlyChangedCategories) {
    model::CLimits limits;
    CFieldConfig config;
    BOOST_TEST_REQUIRE(config.initFromFile("testfiles/new_persist_categorization.conf"));

    std::ostringstream outputStrm;
    {
        core::CJsonOutputStreamWrapper wrappedOutputStream{outputStrm};

        CTestFieldDataCategorizer categorizer{"job", config, limits, nullptr, wrappedOutputStream};

        CFieldDataCategorizer::TStrStrUMap dataRowFields;
        categorizer.registerMutableField(CFieldDataCategorizer::MLCATEGORY_NAME,
                                         dataRowFields[CFieldDataCategorizer::MLCATEGORY_NAME]);
        dataRowFields["message"] = "Node 1 started";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        dataRowFields["message"] = "Node 2 started";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        dataRowFields["message"] = "Somethingelse my message";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        dataRowFields["message"] = "Node 2 started";

        BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));

        categorizer.finalise();
    }
    const std::string& output{outputStrm.str()};
    LOG_DEBUG(<< "Output is: " << output);

    // Output should have a category definition for category_id 1 three times:
    // two for the first two calls, and one for the finalise
    BOOST_REQUIRE_EQUAL(3, countOccurrences(output, "\"category_id\":1"));

    // Output should only have the initial write of category definition for
    // category_id 2 as it was completely up-to-date when finalise was called
    BOOST_REQUIRE_EQUAL(1, countOccurrences(output, "\"category_id\":2"));

    // Stats object on finalize should have been written when there were
    // a total of 4 categorized messages
    BOOST_REQUIRE_EQUAL(1, countOccurrences(output, "\"categorized_doc_count\":4"));
}

BOOST_AUTO_TEST_CASE(testWarnStatusCausesUrgentStatsWrite) {
    model::CLimits limits;
    CFieldConfig config;
    BOOST_TEST_REQUIRE(config.initFromFile("testfiles/new_persist_categorization.conf"));

    std::ostringstream outputStrm;
    {
        core::CJsonOutputStreamWrapper wrappedOutputStream{outputStrm};

        CTestFieldDataCategorizer categorizer{"job", config, limits, nullptr, wrappedOutputStream};

        CFieldDataCategorizer::TStrStrUMap dataRowFields;
        categorizer.registerMutableField(CFieldDataCategorizer::MLCATEGORY_NAME,
                                         dataRowFields[CFieldDataCategorizer::MLCATEGORY_NAME]);
        dataRowFields["message"] = "Node 1 started";

        // The 100th message should cause an urgent stats write with a
        // categorization status of "warn".  This is because there is only one
        // category, which is a red flag, but we don't report warning status
        // until we've seen 100 messages.
        for (std::size_t count = 1; count <= 101; ++count) {
            BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));
        }

        // This should cause another stats write, as the total number of messages
        // categorized has increased to 101 since the previous stats write.
        categorizer.finalise();
    }
    const std::string& output{outputStrm.str()};
    LOG_DEBUG(<< "Output is: " << output);

    // Output should have a category definition for category_id 1 twice:
    // one for the first time it was seen, and another for the finalise
    BOOST_REQUIRE_EQUAL(2, countOccurrences(output, "\"category_id\":1"));

    // Both stats objects should have categorization status warn
    BOOST_REQUIRE_EQUAL(2, countOccurrences(output, "\"categorization_status\":\"warn\""));

    // One stats object should have 100 total count and the other 101
    BOOST_REQUIRE_EQUAL(1, countOccurrences(output, "\"categorized_doc_count\":100"));
    BOOST_REQUIRE_EQUAL(1, countOccurrences(output, "\"categorized_doc_count\":101"));

    // We should have got an annotation on the change to "warn" status
    BOOST_REQUIRE_EQUAL(1, countOccurrences(output, "\"annotation\":\"Categorization status changed to 'warn'\""));
}

BOOST_AUTO_TEST_CASE(testStopCategorizingOnWarnStatusSingleCategorizer) {
    model::CLimits limits;
    CFieldConfig config;
    BOOST_TEST_REQUIRE(config.initFromFile("testfiles/new_persist_categorization.conf"));

    std::ostringstream outputStrm;
    {
        core::CJsonOutputStreamWrapper wrappedOutputStream{outputStrm};

        CTestFieldDataCategorizer categorizer{
            "job", config, limits, nullptr, wrappedOutputStream, nullptr, true};

        CFieldDataCategorizer::TStrStrUMap dataRowFields;
        categorizer.registerMutableField(CFieldDataCategorizer::MLCATEGORY_NAME,
                                         dataRowFields[CFieldDataCategorizer::MLCATEGORY_NAME]);
        dataRowFields["message"] = "Node 1 started";

        // The 100th message should cause the categorization status to change to
        // "warn".  This is because there is only one category, which is a red
        // flag, but we don't report warning status until we've seen 100
        // messages.  Then we should stop categorizing, as we passed
        // stop-on-warn=true to the constructor in this test.
        for (std::size_t count = 1; count < 200; ++count) {
            BOOST_TEST_REQUIRE(categorizer.handleRecord(dataRowFields));
        }

        categorizer.finalise();
    }
    const std::string& output{outputStrm.str()};
    LOG_DEBUG(<< "Output is: " << output);

    // Output should have a category definition for category_id 1 twice:
    // one for the first time it was seen, and another for the finalise
    BOOST_REQUIRE_EQUAL(2, countOccurrences(output, "\"category_id\":1"));
    BOOST_REQUIRE_EQUAL(1, countOccurrences(output, "\"num_matches\":1,"));
    BOOST_REQUIRE_EQUAL(1, countOccurrences(output, "\"num_matches\":100"));

    // The only stats object should have 100 total count and categorization
    // status "warn"; the stats should not have changed again when finalise()
    // was called because we should have stopped categorizing
    BOOST_REQUIRE_EQUAL(1, countOccurrences(output, "\"categorization_status\":\"warn\""));
    BOOST_REQUIRE_EQUAL(1, countOccurrences(output, "\"categorized_doc_count\":100"));

    // We should have got an annotation on the change to "warn" status
    BOOST_REQUIRE_EQUAL(1, countOccurrences(output, "\"annotation\":\"Categorization status changed to 'warn'\""));
}

BOOST_AUTO_TEST_CASE(testStopCategorizingOnWarnStatusPerPartition) {

    // All these combinations of initial state and control message changes
    // should result in the same behaviour
    using TBoolBoolBoolTuple = std::tuple<bool, bool, bool>;
    auto combinations = {TBoolBoolBoolTuple{true, false, false},
                         TBoolBoolBoolTuple{true, true, true},
                         TBoolBoolBoolTuple{false, true, true}};
    for (const auto& args : combinations) {
        const std::string& output{setupPerPartitionStopOnWarnTest(
            std::get<0>(args), std::get<1>(args), std::get<2>(args))};
        LOG_DEBUG(<< "Output is: " << output);

        // We should have a total of 21 categories, each written when they had a
        // count of one
        BOOST_REQUIRE_EQUAL(21, countOccurrences(output, "\"num_matches\":1,"));

        // Output should have a category definition for category_id 1 twice:
        // one for the first time it was seen, and another for the finalise
        BOOST_REQUIRE_EQUAL(2, countOccurrences(output, "\"category_id\":1,"));
        BOOST_REQUIRE_EQUAL(1, countOccurrences(output, "\"num_matches\":100"));

        // The categories for partition "random but changing" should have also been
        // updated on finalise()
        BOOST_REQUIRE_EQUAL(2, countOccurrences(output, "\"category_id\":21"));
        BOOST_REQUIRE_EQUAL(20, countOccurrences(output, "\"num_matches\":10,"));

        // The stats object for partition "nodes" should have 100 total count and
        // categorization status "warn"; the stats should not have changed again
        // when finalise() was called because we should have stopped categorizing
        BOOST_REQUIRE_EQUAL(1, countOccurrences(output, "\"categorization_status\":\"warn\""));
        BOOST_REQUIRE_EQUAL(1, countOccurrences(output, "\"categorized_doc_count\":100"));

        // The stats object for partition "random but changing" should have 200
        // total count and categorization status "ok"
        BOOST_REQUIRE_EQUAL(1, countOccurrences(output, "\"categorization_status\":\"ok\""));
        BOOST_REQUIRE_EQUAL(1, countOccurrences(output, "\"categorized_doc_count\":200"));

        // We should have got an annotation on the change to "warn" status
        BOOST_REQUIRE_EQUAL(1, countOccurrences(output, "\"annotation\":\"Categorization status changed to 'warn' for 'event.dataset' 'nodes'\""));
    }
}

BOOST_AUTO_TEST_CASE(testStopCategorizingOnWarnStatusPerPartitionDisabledByControlMessage) {

    const std::string& output{setupPerPartitionStopOnWarnTest(true, true, false)};
    LOG_DEBUG(<< "Output is: " << output);

    // We should have a total of 21 categories, each written when they had a
    // count of one
    BOOST_REQUIRE_EQUAL(21, countOccurrences(output, "\"num_matches\":1,"));

    // Output should have a category definition for category_id 1 twice:
    // one for the first time it was seen, and another for the finalise
    BOOST_REQUIRE_EQUAL(2, countOccurrences(output, "\"category_id\":1,"));
    BOOST_REQUIRE_EQUAL(1, countOccurrences(output, "\"num_matches\":200"));

    // The categories for partition "random but changing" should have also been
    // updated on finalise()
    BOOST_REQUIRE_EQUAL(2, countOccurrences(output, "\"category_id\":21"));
    BOOST_REQUIRE_EQUAL(20, countOccurrences(output, "\"num_matches\":10,"));

    // A stats object for partition "nodes" should have been written with 100
    // total count and categorization status "warn"; the stats should have
    // changed again when finalise() was called because we should have kept
    // categorizing despite the "warn" status, due to the control message
    // disabling stop-on-warn
    BOOST_REQUIRE_EQUAL(1, countOccurrences(output, "\"categorized_doc_count\":100"));
    BOOST_REQUIRE_EQUAL(2, countOccurrences(output, "\"categorization_status\":\"warn\""));

    // The stats object for partition "random but changing" should have
    // categorization status "ok"
    BOOST_REQUIRE_EQUAL(1, countOccurrences(output, "\"categorization_status\":\"ok\""));

    // The final stats objects for both partitiosn should have 200 total count
    BOOST_REQUIRE_EQUAL(2, countOccurrences(output, "\"categorized_doc_count\":200"));

    // We should have got an annotation on the change to "warn" status
    BOOST_REQUIRE_EQUAL(1, countOccurrences(output, "\"annotation\":\"Categorization status changed to 'warn' for 'event.dataset' 'nodes'\""));
}

BOOST_AUTO_TEST_SUITE_END()
