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

#include "CStateCompressorTest.h"

#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>
#include <core/CStateCompressor.h>
#include <core/CStateDecompressor.h>

#include <boost/random.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/generator_iterator.hpp>

#include <iostream>

using namespace ml;
using namespace core;

typedef boost::mt19937 TRandom;
typedef boost::uniform_int<> TDistribution;
typedef boost::random::variate_generator<TRandom&, TDistribution> TGenerator;
typedef boost::generator_iterator<TGenerator> TGeneratorItr;

namespace {

typedef std::map<std::size_t, std::string> TSizeStrMap;
typedef TSizeStrMap::const_iterator TSizeStrMapCItr;
typedef core::CDataAdder::TOStreamP TOStreamP;
typedef core::CDataSearcher::TIStreamP TIStreamP;

void insert3rdLevel(ml::core::CStatePersistInserter &inserter) {
    inserter.insertValue("ssdrgad", 99999, ml::core::CIEEE754::E_SinglePrecision);
    inserter.insertValue("bbvczcvbdfb", "rrtw");
}

void insert2ndLevel(ml::core::CStatePersistInserter &inserter) {
    inserter.insertValue("eerwq_dsf_dfsgh_h5dafg", 3.14, ml::core::CIEEE754::E_SinglePrecision);
    inserter.insertValue("level2B", 'z');
    for (std::size_t i = 0; i < 50; i++) {
        inserter.insertLevel("hiawat" + core::CStringUtils::typeToString(i), &insert3rdLevel);
    }
}

void insert1stLevel(ml::core::CStatePersistInserter &inserter, std::size_t n) {
    inserter.insertValue("theFirstThing", "a");
    inserter.insertValue("anItemThatComesNext", 25);
    for (std::size_t i = 0; i < n; i++) {
        inserter.insertLevel("levelC" + core::CStringUtils::typeToString(i), &insert2ndLevel);
    }
}

class CMockDataAdder : public ml::core::CDataAdder {
    public:
        CMockDataAdder(std::size_t maxDocSize)
            : m_CurrentDocNum(0),
              m_MaxDocumentSize(maxDocSize) {
        }

        virtual TOStreamP addStreamed(const std::string & /*index*/,
                                      const std::string & /*id*/) {
            ++m_CurrentDocNum;
            m_CurrentStream = TOStreamP(new std::ostringstream);
            return m_CurrentStream;
        }

        virtual bool streamComplete(TOStreamP &strm,
                                    bool /*force*/) {
            CPPUNIT_ASSERT_EQUAL(m_CurrentStream, strm);
            std::ostringstream *ss = dynamic_cast<std::ostringstream *>(m_CurrentStream.get());
            CPPUNIT_ASSERT(ss);
            LOG_TRACE(ss->str());
            m_Data[m_CurrentDocNum] = ss->str();
            LOG_TRACE(m_Data[m_CurrentDocNum]);
            return true;
        }

        virtual std::size_t maxDocumentSize(void) const {
            return m_MaxDocumentSize;
        }

        const TSizeStrMap &data(void) const {
            return m_Data;
        }

    private:
        TSizeStrMap       m_Data;
        std::size_t       m_CurrentDocNum;
        TOStreamP         m_CurrentStream;
        std::size_t       m_MaxDocumentSize;
};

class CMockDataSearcher : public ml::core::CDataSearcher {
    public:
        CMockDataSearcher(CMockDataAdder &adder) : m_Adder(adder), m_AskedFor(0) {
        }

        virtual TIStreamP search(size_t /*currentDocNum*/, size_t /*limit*/) {
            TIStreamP         stream;
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

        std::size_t totalDocs(void) const {
            return m_Adder.data().size();
        }

        std::size_t askedFor(void) const {
            return m_AskedFor;
        }

    private:

        CMockDataAdder &m_Adder;
        std::size_t    m_AskedFor;
};

}


void CStateCompressorTest::testForApiNoKey(void) {
    // This test verifies the basic operation of compressing and decompressing
    // some JSON data, using two simultaneous streams: one regular stringstream,
    // and one compress/decompress stream

    std::ostringstream referenceStream;
    ::CMockDataAdder   mockKvAdder(3000);
    {
        ml::core::CStateCompressor compressor(mockKvAdder);

        TOStreamP strm = compressor.addStreamed("1", "");
        {
            CJsonStatePersistInserter inserter(*strm);
            CJsonStatePersistInserter referenceInserter(referenceStream);

            insert1stLevel(inserter, 1001);
            insert1stLevel(referenceInserter, 1001);
        }
        compressor.streamComplete(strm, true);
    }

    std::string ref(referenceStream.str());
    std::string restored;
    {
        LOG_DEBUG("Restoring data");
        CMockDataSearcher            mockKvSearcher(mockKvAdder);
        ml::core::CStateDecompressor decompressor(mockKvSearcher);
        decompressor.setStateRestoreSearch("1", "");
        TIStreamP istrm = decompressor.search(1, 1);

        std::istreambuf_iterator<char> eos;
        restored.assign(std::istreambuf_iterator<char>(*istrm), eos);
    }

    LOG_DEBUG("Size: " << restored.size());
    LOG_DEBUG("Reference size: " << ref.size());

    CPPUNIT_ASSERT_EQUAL(ref.size(), restored.size());
}

void CStateCompressorTest::testStreaming(void) {
    // The purpose of this test is to add a reasonable block of data to the
    // compressed store, then read it back out and show that the data is
    // read in stream chunks, not all at once. CMockDataSearcher has a
    // method to return how much of the stored content has been accessed.

    ::CMockDataAdder mockKvAdder(3000);
    {
        // Add lots of data to the mock datastore (compressed on the way)
        ml::core::CStateCompressor compressor(mockKvAdder);

        TOStreamP strm = compressor.addStreamed("1", "");
        {
            CJsonStatePersistInserter inserter(*strm);

            insert1stLevel(inserter, 10001);
        }
        compressor.streamComplete(strm, true);
    }

    {
        // Read the datastore via a JSON traverser, and show that the
        // data is streamed, not read all at once
        std::size_t         lastAskedFor = 0;
        ::CMockDataSearcher mockKvSearcher(mockKvAdder);
        LOG_TRACE("After compression, there are " << mockKvSearcher.totalDocs() << " docs, asked for " << mockKvSearcher.askedFor());
        ml::core::CStateDecompressor decompressor(mockKvSearcher);
        decompressor.setStateRestoreSearch("1", "");
        TIStreamP istrm = decompressor.search(1, 1);

        CPPUNIT_ASSERT_EQUAL(std::size_t(0), mockKvSearcher.askedFor());

        CJsonStateRestoreTraverser traverser(*istrm);
        traverser.next();
        CPPUNIT_ASSERT(mockKvSearcher.askedFor() > lastAskedFor);
        lastAskedFor = mockKvSearcher.askedFor();

        for (std::size_t i = 0; i < 5000; i++) {
            traverser.next();
        }
        CPPUNIT_ASSERT(mockKvSearcher.askedFor() > lastAskedFor);
        lastAskedFor = mockKvSearcher.askedFor();

        for (std::size_t i = 0; i < 5000; i++) {
            traverser.next();
        }
        CPPUNIT_ASSERT(mockKvSearcher.askedFor() > lastAskedFor);
        lastAskedFor = mockKvSearcher.askedFor();

        for (std::size_t i = 0; i < 5000; i++) {
            traverser.next();
        }
        CPPUNIT_ASSERT(mockKvSearcher.askedFor() > lastAskedFor);
        lastAskedFor = mockKvSearcher.askedFor();

        for (std::size_t i = 0; i < 5000; i++) {
            traverser.next();
        }
        CPPUNIT_ASSERT(mockKvSearcher.askedFor() > lastAskedFor);
        lastAskedFor = mockKvSearcher.askedFor();
        while (traverser.next()) {
        }
        ;
        LOG_TRACE("Asked for: " << mockKvSearcher.askedFor());
        CPPUNIT_ASSERT_EQUAL(mockKvSearcher.askedFor(), mockKvAdder.data().size());
    }
}


void CStateCompressorTest::testChunking(void) {
    // Put arbitrary string data into the stream, and stress different sizes
    // check CMockDataAdder with max doc sizes from 500 to 500000

    TRandom       rng(1849434026ul);
    TGenerator    generator(rng, TDistribution(0, 254));
    TGeneratorItr randItr(&generator);

    for (std::size_t i = 500; i < 5000001; i *= 10) {
        // check string data from sizes 1 to 200000
        for (std::size_t j = 1; j < 2570000; j *= 4) {
            CMockDataAdder     adder(i);
            std::ostringstream ss;
            std::string        decompressed;
            try {
                {
                    ml::core::CStateCompressor      compressor(adder);
                    ml::core::CDataAdder::TOStreamP strm = compressor.addStreamed("1", "");
                    for (std::size_t k = 0; k < j; k++) {
                        char c = char(*randItr++);
                        ss << c;
                        (*strm) << c;
                    }
                }
                {
                    CMockDataSearcher            searcher(adder);
                    ml::core::CStateDecompressor decompressor(searcher);
                    decompressor.setStateRestoreSearch("1", "");
                    ml::core::CDataSearcher::TIStreamP strm = decompressor.search(1, 1);
                    std::istreambuf_iterator<char>     eos;
                    decompressed.assign(std::istreambuf_iterator<char>(*strm), eos);
                }
            } catch (std::exception &e) {
                LOG_DEBUG("Error in test case " << i << " / " << j << ": " << e.what());
                LOG_DEBUG("String is: " << ss.str());
                CPPUNIT_ASSERT(false);
            }
            CPPUNIT_ASSERT_EQUAL(ss.str(), decompressed);
        }
        LOG_DEBUG("Done chunk size " << i);
    }

    // Do a really big string test
    {
        CMockDataAdder     adder(0xffffffff);
        std::ostringstream ss;
        std::string        decompressed;
        try {
            {
                ml::core::CStateCompressor      compressor(adder);
                ml::core::CDataAdder::TOStreamP strm = compressor.addStreamed("1", "");
                for (std::size_t k = 0; k < 100000000; k++) {
                    char c = char(*randItr++);
                    ss << c;
                    (*strm) << c;
                }
            }
            {
                CMockDataSearcher            searcher(adder);
                ml::core::CStateDecompressor decompressor(searcher);
                decompressor.setStateRestoreSearch("1", "");
                ml::core::CDataSearcher::TIStreamP strm = decompressor.search(1, 1);
                std::istreambuf_iterator<char>     eos;
                decompressed.assign(std::istreambuf_iterator<char>(*strm), eos);
            }
        } catch (std::exception &e) {
            LOG_DEBUG("Error in test case " << e.what());
            LOG_DEBUG("String is: " << ss.str());
            CPPUNIT_ASSERT(false);
        }
        CPPUNIT_ASSERT_EQUAL(ss.str(), decompressed);
    }
}

CppUnit::Test* CStateCompressorTest::suite() {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CStateCompressorTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CStateCompressorTest>(
                               "CStateCompressorTest::testForApiNoKey",
                               &CStateCompressorTest::testForApiNoKey) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CStateCompressorTest> (
                               "CStateCompressorTest::testStreaming",
                               &CStateCompressorTest::testStreaming) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CStateCompressorTest> (
                               "CStateCompressorTest::testChunking",
                               &CStateCompressorTest::testChunking) );

    return suiteOfTests;
}
