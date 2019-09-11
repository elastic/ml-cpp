/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_test_CMockDataSearcher_h
#define INCLUDED_ml_test_CMockDataSearcher_h

#include <core/CDataSearcher.h>

#include <test/CMockDataAdder.h>


#include <map>


namespace ml {
namespace test {

class CMockDataSearcher : public ml::core::CDataSearcher {
public:
    using TSizeStrMap = std::map<std::size_t, std::string>;
    using TSizeStrMapCItr = TSizeStrMap::const_iterator;
    using TOStreamP = ml::core::CDataAdder::TOStreamP;
public:
    CMockDataSearcher(CMockDataAdder &adder) : m_Adder(adder), m_AskedFor(0) {}

    virtual TIStreamP search(size_t /*currentDocNum*/, size_t /*limit*/) {
        TIStreamP stream;
        const TSizeStrMap &events = m_Adder.data();

        TSizeStrMapCItr iter = events.find(m_AskedFor + 1);
        if (iter == events.end()) {
            // return a stream here that is in the fail state
            stream.reset(new std::stringstream);
            stream->setstate(std::ios_base::failbit);
        } else {
            stream.reset(new std::stringstream(iter->second));
            ++m_AskedFor;
        }
        return stream;
    }

    std::size_t totalDocs() const { return m_Adder.data().size(); }

    std::size_t askedFor() const { return m_AskedFor; }

private:
    CMockDataAdder &m_Adder;
    std::size_t m_AskedFor;
};

}
}
#endif // INCLUDED_ml_test_CMockDataSearcher_h
