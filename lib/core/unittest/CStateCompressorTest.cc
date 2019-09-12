/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CStateCompressorTest.h"

#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>
#include <core/CStateCompressor.h>
#include <core/CStateDecompressor.h>

#include <test/CMockDataAdder.h>
#include <test/CMockDataSearcher.h>

#include <boost/generator_iterator.hpp>
#include <boost/random.hpp>
#include <boost/random/uniform_int.hpp>

#include <iostream>

using namespace ml;
using namespace core;

using TRandom = boost::mt19937;
using TDistribution = boost::uniform_int<>;
using TGenerator = boost::random::variate_generator<TRandom&, TDistribution>;
using TGeneratorItr = boost::generator_iterator<TGenerator>;

namespace {

using TSizeStrMap = std::map<std::size_t, std::string>;
using TSizeStrMapCItr = TSizeStrMap::const_iterator;
using TOStreamP = core::CDataAdder::TOStreamP;
using TIStreamP = core::CDataSearcher::TIStreamP;

void insert3rdLevel(ml::core::CStatePersistInserter& inserter) {
    inserter.insertValue("ssdrgad", 99999, ml::core::CIEEE754::E_SinglePrecision);
    inserter.insertValue("bbvczcvbdfb", "rrtw");
}

void insert2ndLevel(ml::core::CStatePersistInserter& inserter) {
    inserter.insertValue("eerwq_dsf_dfsgh_h5dafg", 3.14, ml::core::CIEEE754::E_SinglePrecision);
    inserter.insertValue("level2B", 'z');
    for (std::size_t i = 0; i < 50; i++) {
        inserter.insertLevel("hiawat" + core::CStringUtils::typeToString(i), &insert3rdLevel);
    }
}

void insert1stLevel(ml::core::CStatePersistInserter& inserter, std::size_t n) {
    inserter.insertValue("theFirstThing", "a");
    inserter.insertValue("anItemThatComesNext", 25);
    for (std::size_t i = 0; i < n; i++) {
        inserter.insertLevel("levelC" + core::CStringUtils::typeToString(i), &insert2ndLevel);
    }
}
}

void CStateCompressorTest::testForApiNoKey() {
    // This test verifies the basic operation of compressing and decompressing
    // some JSON data, using two simultaneous streams: one regular stringstream,
    // and one compress/decompress stream

    std::ostringstream referenceStream;
    ml::test::CMockDataAdder mockKvAdder(3000);
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
        LOG_DEBUG(<< "Restoring data");
        ml::test::CMockDataSearcher mockKvSearcher(mockKvAdder);
        ml::core::CStateDecompressor decompressor(mockKvSearcher);
        decompressor.setStateRestoreSearch("1", "");
        TIStreamP istrm = decompressor.search(1, 1);

        std::istreambuf_iterator<char> eos;
        restored.assign(std::istreambuf_iterator<char>(*istrm), eos);
    }

    LOG_DEBUG(<< "Size: " << restored.size());
    LOG_DEBUG(<< "Reference size: " << ref.size());

    CPPUNIT_ASSERT_EQUAL(ref.size(), restored.size());
}

void CStateCompressorTest::testStreaming() {
    // The purpose of this test is to add a reasonable block of data to the
    // compressed store, then read it back out and show that the data is
    // read in stream chunks, not all at once. CMockDataSearcher has a
    // method to return how much of the stored content has been accessed.

    ml::test::CMockDataAdder mockKvAdder(3000);
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
        std::size_t lastAskedFor = 0;
        ml::test::CMockDataSearcher mockKvSearcher(mockKvAdder);
        LOG_TRACE(<< "After compression, there are " << mockKvSearcher.totalDocs()
                  << " docs, asked for " << mockKvSearcher.askedFor());
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
        };
        LOG_TRACE(<< "Asked for: " << mockKvSearcher.askedFor());
        CPPUNIT_ASSERT_EQUAL(mockKvSearcher.askedFor(), mockKvAdder.data().size());
    }
}

void CStateCompressorTest::testChunking() {
    // Put arbitrary string data into the stream, and stress different sizes
    // check CMockDataAdder with max doc sizes from 500 to 500000

    TRandom rng(1849434026ul);
    TGenerator generator(rng, TDistribution(0, 254));
    TGeneratorItr randItr(&generator);

    for (std::size_t i = 500; i < 5000001; i *= 10) {
        // check string data from sizes 1 to 200000
        for (std::size_t j = 1; j < 2570000; j *= 4) {
            ml::test::CMockDataAdder adder(i);
            std::ostringstream ss;
            std::string decompressed;
            try {
                {
                    ml::core::CStateCompressor compressor(adder);
                    ml::core::CDataAdder::TOStreamP strm = compressor.addStreamed("1", "");
                    for (std::size_t k = 0; k < j; k++) {
                        char c = char(*randItr++);
                        ss << c;
                        (*strm) << c;
                    }
                }
                {
                    ml::test::CMockDataSearcher searcher(adder);
                    ml::core::CStateDecompressor decompressor(searcher);
                    decompressor.setStateRestoreSearch("1", "");
                    ml::core::CDataSearcher::TIStreamP strm = decompressor.search(1, 1);
                    std::istreambuf_iterator<char> eos;
                    decompressed.assign(std::istreambuf_iterator<char>(*strm), eos);
                }
            } catch (std::exception& e) {
                LOG_DEBUG(<< "Error in test case " << i << " / " << j << ": " << e.what());
                LOG_DEBUG(<< "String is: " << ss.str());
                CPPUNIT_ASSERT(false);
            }
            CPPUNIT_ASSERT_EQUAL(ss.str(), decompressed);
        }
        LOG_DEBUG(<< "Done chunk size " << i);
    }

    // Do a really big string test
    {
        ml::test::CMockDataAdder adder(0xffffffff);
        std::ostringstream ss;
        std::string decompressed;
        try {
            {
                ml::core::CStateCompressor compressor(adder);
                ml::core::CDataAdder::TOStreamP strm = compressor.addStreamed("1", "");
                for (std::size_t k = 0; k < 100000000; k++) {
                    char c = char(*randItr++);
                    ss << c;
                    (*strm) << c;
                }
            }
            {
                ml::test::CMockDataSearcher searcher(adder);
                ml::core::CStateDecompressor decompressor(searcher);
                decompressor.setStateRestoreSearch("1", "");
                ml::core::CDataSearcher::TIStreamP strm = decompressor.search(1, 1);
                std::istreambuf_iterator<char> eos;
                decompressed.assign(std::istreambuf_iterator<char>(*strm), eos);
            }
        } catch (std::exception& e) {
            LOG_DEBUG(<< "Error in test case " << e.what());
            LOG_DEBUG(<< "String is: " << ss.str());
            CPPUNIT_ASSERT(false);
        }
        CPPUNIT_ASSERT_EQUAL(ss.str(), decompressed);
    }
}

CppUnit::Test* CStateCompressorTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CStateCompressorTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CStateCompressorTest>(
        "CStateCompressorTest::testForApiNoKey", &CStateCompressorTest::testForApiNoKey));
    suiteOfTests->addTest(new CppUnit::TestCaller<CStateCompressorTest>(
        "CStateCompressorTest::testStreaming", &CStateCompressorTest::testStreaming));
    suiteOfTests->addTest(new CppUnit::TestCaller<CStateCompressorTest>(
        "CStateCompressorTest::testChunking", &CStateCompressorTest::testChunking));

    return suiteOfTests;
}
