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

#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>
#include <core/CStateCompressor.h>
#include <core/CStateDecompressor.h>

#include <boost/generator_iterator.hpp>
#include <boost/random.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/test/unit_test.hpp>

#include <iostream>

BOOST_AUTO_TEST_SUITE(CStateCompressorTest)

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

class CMockDataAdder : public ml::core::CDataAdder {
public:
    CMockDataAdder(std::size_t maxDocSize)
        : m_CurrentDocNum{0}, m_MaxDocumentSize{maxDocSize} {}

    TOStreamP addStreamed(const std::string& /*id*/) override {
        ++m_CurrentDocNum;
        m_CurrentStream = TOStreamP(new std::ostringstream);
        return m_CurrentStream;
    }

    bool streamComplete(TOStreamP& strm, bool /*force*/) override {
        BOOST_REQUIRE_EQUAL(m_CurrentStream, strm);
        std::ostringstream* ss =
            dynamic_cast<std::ostringstream*>(m_CurrentStream.get());
        BOOST_TEST_REQUIRE(ss);
        LOG_TRACE(<< ss->str());
        m_Data[m_CurrentDocNum] = ss->str();
        LOG_TRACE(<< m_Data[m_CurrentDocNum]);
        return true;
    }

    std::size_t maxDocumentSize() const override { return m_MaxDocumentSize; }

    const TSizeStrMap& data() const { return m_Data; }

private:
    TSizeStrMap m_Data;
    std::size_t m_CurrentDocNum;
    TOStreamP m_CurrentStream;
    std::size_t m_MaxDocumentSize;
};

class CMockDataSearcher : public ml::core::CDataSearcher {
public:
    CMockDataSearcher(CMockDataAdder& adder) : m_Adder{adder}, m_AskedFor{0} {}

    TIStreamP search(std::size_t /*currentDocNum*/, std::size_t /*limit*/) override {
        TIStreamP stream;
        const TSizeStrMap& events = m_Adder.data();

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
    CMockDataAdder& m_Adder;
    std::size_t m_AskedFor;
};
}

BOOST_AUTO_TEST_CASE(testForApiNoKey) {
    // This test verifies the basic operation of compressing and decompressing
    // some JSON data, using two simultaneous streams: one regular stringstream,
    // and one compress/decompress stream

    std::ostringstream referenceStream;
    CMockDataAdder mockKvAdder(3000);
    {
        ml::core::CStateCompressor compressor(mockKvAdder);

        TOStreamP strm = compressor.addStreamed("");
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
        CMockDataSearcher mockKvSearcher(mockKvAdder);
        ml::core::CStateDecompressor decompressor(mockKvSearcher);
        TIStreamP istrm = decompressor.search(1, 1);

        std::istreambuf_iterator<char> eos;
        restored.assign(std::istreambuf_iterator<char>(*istrm), eos);
    }

    LOG_DEBUG(<< "Size: " << restored.size());
    LOG_DEBUG(<< "Reference size: " << ref.size());

    BOOST_REQUIRE_EQUAL(ref.size(), restored.size());
}

BOOST_AUTO_TEST_CASE(testStreaming) {
    // The purpose of this test is to add a reasonable block of data to the
    // compressed store, then read it back out and show that the data is
    // read in stream chunks, not all at once. CMockDataSearcher has a
    // method to return how much of the stored content has been accessed.

    CMockDataAdder mockKvAdder(3000);
    {
        // Add lots of data to the mock datastore (compressed on the way)
        ml::core::CStateCompressor compressor(mockKvAdder);

        TOStreamP strm = compressor.addStreamed("");
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
        CMockDataSearcher mockKvSearcher(mockKvAdder);
        LOG_TRACE(<< "After compression, there are " << mockKvSearcher.totalDocs()
                  << " docs, asked for " << mockKvSearcher.askedFor());
        ml::core::CStateDecompressor decompressor(mockKvSearcher);
        TIStreamP istrm = decompressor.search(1, 1);

        BOOST_REQUIRE_EQUAL(0, mockKvSearcher.askedFor());

        CJsonStateRestoreTraverser traverser(*istrm);
        traverser.next();
        BOOST_TEST_REQUIRE(mockKvSearcher.askedFor() > lastAskedFor);
        lastAskedFor = mockKvSearcher.askedFor();

        for (std::size_t i = 0; i < 5000; i++) {
            traverser.next();
        }
        BOOST_TEST_REQUIRE(mockKvSearcher.askedFor() > lastAskedFor);
        lastAskedFor = mockKvSearcher.askedFor();

        for (std::size_t i = 0; i < 5000; i++) {
            traverser.next();
        }
        BOOST_TEST_REQUIRE(mockKvSearcher.askedFor() > lastAskedFor);
        lastAskedFor = mockKvSearcher.askedFor();

        for (std::size_t i = 0; i < 5000; i++) {
            traverser.next();
        }
        BOOST_TEST_REQUIRE(mockKvSearcher.askedFor() > lastAskedFor);
        lastAskedFor = mockKvSearcher.askedFor();

        for (std::size_t i = 0; i < 5000; i++) {
            traverser.next();
        }
        BOOST_TEST_REQUIRE(mockKvSearcher.askedFor() > lastAskedFor);
        lastAskedFor = mockKvSearcher.askedFor();
        while (traverser.next()) {
        };
        LOG_TRACE(<< "Asked for: " << mockKvSearcher.askedFor());
        BOOST_REQUIRE_EQUAL(mockKvSearcher.askedFor(), mockKvAdder.data().size());
    }
}

BOOST_AUTO_TEST_CASE(testChunking) {
    // Put arbitrary string data into the stream, and stress different sizes
    // check CMockDataAdder with max doc sizes from 500 to 500000

    TRandom rng(1849434026ul);
    TGenerator generator(rng, TDistribution(0, 254));
    TGeneratorItr randItr(&generator);

    for (std::size_t i = 500; i < 5000001; i *= 10) {
        // check string data from sizes 1 to 200000
        for (std::size_t j = 1; j < 2570000; j *= 4) {
            CMockDataAdder adder(i);
            std::ostringstream ss;
            std::string decompressed;
            try {
                {
                    ml::core::CStateCompressor compressor(adder);
                    ml::core::CDataAdder::TOStreamP strm = compressor.addStreamed("");
                    for (std::size_t k = 0; k < j; k++) {
                        char c = char(*randItr++);
                        ss << c;
                        (*strm) << c;
                    }
                }
                {
                    CMockDataSearcher searcher(adder);
                    ml::core::CStateDecompressor decompressor(searcher);
                    ml::core::CDataSearcher::TIStreamP strm = decompressor.search(1, 1);
                    std::istreambuf_iterator<char> eos;
                    decompressed.assign(std::istreambuf_iterator<char>(*strm), eos);
                }
            } catch (std::exception& e) {
                LOG_DEBUG(<< "Error in test case " << i << " / " << j << ": " << e.what());
                LOG_DEBUG(<< "String is: " << ss.str());
                BOOST_TEST_REQUIRE(false);
            }
            BOOST_REQUIRE_EQUAL(ss.str(), decompressed);
        }
        LOG_DEBUG(<< "Done chunk size " << i);
    }

    // Do a really big string test
    {
        CMockDataAdder adder(0xffffffff);
        std::ostringstream ss;
        std::string decompressed;
        try {
            {
                ml::core::CStateCompressor compressor(adder);
                ml::core::CDataAdder::TOStreamP strm = compressor.addStreamed("");
                for (std::size_t k = 0; k < 100000000; k++) {
                    char c = char(*randItr++);
                    ss << c;
                    (*strm) << c;
                }
            }
            {
                CMockDataSearcher searcher(adder);
                ml::core::CStateDecompressor decompressor(searcher);
                ml::core::CDataSearcher::TIStreamP strm = decompressor.search(1, 1);
                std::istreambuf_iterator<char> eos;
                decompressed.assign(std::istreambuf_iterator<char>(*strm), eos);
            }
        } catch (std::exception& e) {
            LOG_DEBUG(<< "Error in test case " << e.what());
            LOG_DEBUG(<< "String is: " << ss.str());
            BOOST_TEST_REQUIRE(false);
        }
        BOOST_REQUIRE_EQUAL(ss.str(), decompressed);
    }
}

BOOST_AUTO_TEST_SUITE_END()
