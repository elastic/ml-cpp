/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_test_CMockDataAdder_h
#define INCLUDED_ml_test_CMockDataAdder_h

#include <core/CDataAdder.h>

#include <cppunit/TestAssert.h>

#include <string>

namespace ml {
namespace test {

class CMockDataAdder : public ml::core::CDataAdder {
public:
    using TSizeStrMap = std::map<std::size_t, std::string>;
    using TOStreamP = ml::core::CDataAdder::TOStreamP;

public:
    CMockDataAdder(std::size_t maxDocSize)
        : m_CurrentDocNum(0), m_MaxDocumentSize(maxDocSize) {}

    virtual TOStreamP addStreamed(const std::string& /*index*/, const std::string& /*id*/) {
        ++m_CurrentDocNum;
        m_CurrentStream = TOStreamP(new std::ostringstream);
        return m_CurrentStream;
    }

    virtual bool streamComplete(TOStreamP& strm, bool /*force*/) {
        CPPUNIT_ASSERT_EQUAL(m_CurrentStream, strm);
        std::ostringstream* ss =
            dynamic_cast<std::ostringstream*>(m_CurrentStream.get());
        CPPUNIT_ASSERT(ss);
        LOG_TRACE(<< ss->str());
        m_Data[m_CurrentDocNum] = ss->str();
        LOG_TRACE(<< m_Data[m_CurrentDocNum]);
        return true;
    }

    virtual std::size_t maxDocumentSize() const { return m_MaxDocumentSize; }

    const TSizeStrMap& data() const { return m_Data; }

private:
    TSizeStrMap m_Data;
    std::size_t m_CurrentDocNum;
    TOStreamP m_CurrentStream;
    std::size_t m_MaxDocumentSize;
};
}
}

#endif // INCLUDED_ml_test_CMockDataAdder_h
