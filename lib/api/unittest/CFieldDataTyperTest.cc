/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#include "CFieldDataTyperTest.h"

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CDataAdder.h>
#include <core/CDataSearcher.h>

#include <model/CLimits.h>

#include <api/CFieldConfig.h>
#include <api/CFieldDataTyper.h>
#include <api/CJsonOutputWriter.h>
#include <api/CNullOutput.h>
#include <api/COutputChainer.h>
#include <api/COutputHandler.h>

#include "CMockDataProcessor.h"

#include <sstream>

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
        CTestOutputHandler(void) : COutputHandler(), m_NewStream(false),
                                   m_Finalised(false), m_Records(0) {
        }

        virtual ~CTestOutputHandler(void) {
        }

        virtual void finalise(void) {
            m_Finalised = true;
        }

        bool hasFinalised(void) const {
            return m_Finalised;
        }

        virtual void newOutputStream(void) {
            m_NewStream = true;
        }

        bool isNewStream(void) const {
            return m_NewStream;
        }

        virtual bool fieldNames(const TStrVec & /*fieldNames*/,
                                const TStrVec & /*extraFieldNames*/) {
            return true;
        }

        virtual const TStrVec &fieldNames(void) const {
            return m_FieldNames;
        }

        virtual bool writeRow(const TStrStrUMap & /*dataRowFields*/,
                              const TStrStrUMap & /*overrideDataRowFields*/) {
            m_Records++;
            return true;
        }

        uint64_t getNumRows(void) const {
            return m_Records;
        }

    private:
        TStrVec m_FieldNames;

        bool m_NewStream;

        bool m_Finalised;

        uint64_t m_Records;
};

class CTestDataSearcher : public core::CDataSearcher {
    public:
        CTestDataSearcher(const std::string &data)
            : m_Stream(new std::istringstream(data)) {
        }

        virtual TIStreamP search(size_t /*currentDocNum*/, size_t /*limit*/) {
            return m_Stream;
        }

    private:
        TIStreamP m_Stream;
};

class CTestDataAdder : public core::CDataAdder {
    public:
        CTestDataAdder(void)
            : m_Stream(new std::ostringstream) {
        }

        virtual TOStreamP addStreamed(const std::string & /*index*/,
                                      const std::string & /*id*/) {
            return m_Stream;
        }

        virtual bool streamComplete(TOStreamP & /*strm*/, bool /*force*/) {
            return true;
        }

        TOStreamP getStream(void) {
            return m_Stream;
        }

    private:
        TOStreamP m_Stream;
};

}

void CFieldDataTyperTest::testAll(void) {
    model::CLimits limits;
    CFieldConfig   config;
    CPPUNIT_ASSERT(config.initFromFile("testfiles/new_persist_categorization.conf"));
    CTestOutputHandler handler;

    std::ostringstream             outputStrm;
    core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
    CJsonOutputWriter              writer("job", wrappedOutputStream);

    CFieldDataTyper typer("job", config, limits, handler, writer);
    CPPUNIT_ASSERT_EQUAL(false, handler.isNewStream());
    typer.newOutputStream();
    CPPUNIT_ASSERT_EQUAL(true, handler.isNewStream());

    CPPUNIT_ASSERT_EQUAL(false, handler.hasFinalised());
    CPPUNIT_ASSERT_EQUAL(uint64_t(0), typer.numRecordsHandled());

    CFieldDataTyper::TStrStrUMap dataRowFields;
    dataRowFields["message"] = "thing";
    dataRowFields["two"] = "other";

    CPPUNIT_ASSERT(typer.handleRecord(dataRowFields));

    CPPUNIT_ASSERT_EQUAL(uint64_t(1), typer.numRecordsHandled());
    CPPUNIT_ASSERT_EQUAL(typer.numRecordsHandled(), handler.getNumRows());

    // try a couple of erroneous cases
    dataRowFields.clear();
    CPPUNIT_ASSERT(typer.handleRecord(dataRowFields));

    dataRowFields["thing"] = "bling";
    dataRowFields["thang"] = "wing";
    CPPUNIT_ASSERT(typer.handleRecord(dataRowFields));

    dataRowFields["message"] = "";
    dataRowFields["thang"] = "wing";
    CPPUNIT_ASSERT(typer.handleRecord(dataRowFields));

    CPPUNIT_ASSERT_EQUAL(uint64_t(4), typer.numRecordsHandled());
    CPPUNIT_ASSERT_EQUAL(typer.numRecordsHandled(), handler.getNumRows());

    typer.finalise();
    CPPUNIT_ASSERT(handler.hasFinalised());

    // do a persist / restore
    std::string origJson;
    {
        CTestDataAdder adder;
        typer.persistState(adder);
        std::ostringstream &ss = dynamic_cast<std::ostringstream &>(*adder.getStream());
        origJson = ss.str();
    }

    std::string newJson;
    LOG_DEBUG("origJson = " << origJson);
    {
        model::CLimits                 limits2;
        CFieldConfig                   config2("x", "y");
        CTestOutputHandler             handler2;
        std::ostringstream             outputStrm2;
        core::CJsonOutputStreamWrapper wrappedOutputStream2 (outputStrm2);
        CJsonOutputWriter              writer2("job", wrappedOutputStream2);

        CFieldDataTyper   newTyper("job", config2, limits2, handler2, writer2);
        CTestDataSearcher restorer(origJson);
        core_t::TTime     time = 0;
        newTyper.restoreState(restorer, time);

        CTestDataAdder adder;
        newTyper.persistState(adder);
        std::ostringstream &ss = dynamic_cast<std::ostringstream &>(*adder.getStream());
        newJson = ss.str();
    }
    CPPUNIT_ASSERT_EQUAL(origJson, newJson);
}

void CFieldDataTyperTest::testNodeReverseSearch(void) {
    model::CLimits limits;
    CFieldConfig   config;
    CPPUNIT_ASSERT(config.initFromFile("testfiles/new_persist_categorization.conf"));

    std::ostringstream outputStrm;
    {
        CNullOutput                    nullOutput;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
        CJsonOutputWriter              writer("job", wrappedOutputStream);

        CFieldDataTyper typer("job", config, limits, nullOutput, writer);

        CFieldDataTyper::TStrStrUMap dataRowFields;
        dataRowFields["message"] = "Node 1 started";

        CPPUNIT_ASSERT(typer.handleRecord(dataRowFields));

        dataRowFields["message"] = "Node 2 started";

        CPPUNIT_ASSERT(typer.handleRecord(dataRowFields));

        typer.finalise();
    }

    const std::string &output = outputStrm.str();
    LOG_DEBUG("Output is: " << output);

    // Assert that the reverse search contains all expected tokens when
    // categorization is run end-to-end (obviously computation of categories and
    // reverse search creation are tested more thoroughly in the unit tests for
    // their respective classes, but this test helps to confirm that they work
    // together)
    CPPUNIT_ASSERT(output.find("\"terms\":\"Node started\"") != std::string::npos);
    CPPUNIT_ASSERT(output.find("\"regex\":\".*?Node.+?started.*\"") != std::string::npos);
    // The input data should NOT be in the output
    CPPUNIT_ASSERT(output.find("\"message\"") == std::string::npos);
}

void CFieldDataTyperTest::testPassOnControlMessages(void) {
    model::CLimits limits;
    CFieldConfig   config;
    CPPUNIT_ASSERT(config.initFromFile("testfiles/new_persist_categorization.conf"));

    std::ostringstream outputStrm;
    {
        CNullOutput                    nullOutput;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
        CJsonOutputWriter              writer("job", wrappedOutputStream);

        CMockDataProcessor mockProcessor(nullOutput);
        COutputChainer     outputChainer(mockProcessor);
        CFieldDataTyper    typer("job", config, limits, outputChainer, writer);

        CFieldDataTyper::TStrStrUMap dataRowFields;
        dataRowFields["."] = "f7";

        CPPUNIT_ASSERT(typer.handleRecord(dataRowFields));

        typer.finalise();
    }

    const std::string &output = outputStrm.str();
    LOG_DEBUG("Output is: " << output);
    CPPUNIT_ASSERT_EQUAL(std::string("[]"), output);
}

void CFieldDataTyperTest::testHandleControlMessages(void) {
    model::CLimits limits;
    CFieldConfig   config;
    CPPUNIT_ASSERT(config.initFromFile("testfiles/new_persist_categorization.conf"));

    std::ostringstream outputStrm;
    {
        CNullOutput                    nullOutput;
        core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
        CJsonOutputWriter              writer("job", wrappedOutputStream);

        CFieldDataTyper typer("job", config, limits, nullOutput, writer, nullptr);

        CFieldDataTyper::TStrStrUMap dataRowFields;
        dataRowFields["."] = "f7";

        CPPUNIT_ASSERT(typer.handleRecord(dataRowFields));

        typer.finalise();
    }

    const std::string &output = outputStrm.str();
    LOG_DEBUG("Output is: " << output);
    CPPUNIT_ASSERT_EQUAL(std::string::size_type(0),
                         output.find("[{\"flush\":{\"id\":\"7\",\"last_finalized_bucket_end\":0}}"));
}

void CFieldDataTyperTest::testRestoreStateFailsWithEmptyState(void) {
    model::CLimits limits;
    CFieldConfig   config;
    CPPUNIT_ASSERT(config.initFromFile("testfiles/new_persist_categorization.conf"));

    std::ostringstream             outputStrm;
    CNullOutput                    nullOutput;
    core::CJsonOutputStreamWrapper wrappedOutputStream(outputStrm);
    CJsonOutputWriter              writer("job", wrappedOutputStream);
    CFieldDataTyper                typer("job", config, limits, nullOutput, writer, nullptr);

    core_t::TTime  completeToTime(0);
    CEmptySearcher restoreSearcher;
    CPPUNIT_ASSERT(typer.restoreState(restoreSearcher, completeToTime) == false);
}

CppUnit::Test* CFieldDataTyperTest::suite() {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CFieldDataTyperTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CFieldDataTyperTest>(
                               "CFieldDataTyperTest::testAll",
                               &CFieldDataTyperTest::testAll) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CFieldDataTyperTest>(
                               "CFieldDataTyperTest::testNodeReverseSearch",
                               &CFieldDataTyperTest::testNodeReverseSearch) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CFieldDataTyperTest>(
                               "CFieldDataTyperTest::testPassOnControlMessages",
                               &CFieldDataTyperTest::testPassOnControlMessages) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CFieldDataTyperTest>(
                               "CFieldDataTyperTest::testHandleControlMessages",
                               &CFieldDataTyperTest::testHandleControlMessages) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CFieldDataTyperTest>(
                               "CFieldDataTyperTest::testRestoreStateFailsWithEmptyState",
                               &CFieldDataTyperTest::testRestoreStateFailsWithEmptyState) );
    return suiteOfTests;
}

