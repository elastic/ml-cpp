/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CBase64Filter.h>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/test/unit_test.hpp>

#include <boost/generator_iterator.hpp>
#include <boost/random.hpp>
#include <boost/random/uniform_int.hpp>

BOOST_AUTO_TEST_SUITE(CBase64FilterTest)

using TRandom = boost::mt19937;
using TDistribution = boost::uniform_int<>;
using TGenerator = boost::random::variate_generator<TRandom&, TDistribution>;
using TGeneratorItr = boost::generator_iterator<TGenerator>;

using namespace ml;
using namespace core;

namespace {

using TFilteredOutput = boost::iostreams::filtering_stream<boost::iostreams::output>;
using TFilteredInput = boost::iostreams::filtering_stream<boost::iostreams::input>;

// Implements the boost::iostreams Source template interface
class CMockSource {
public:
    using char_type = char;

    struct category : public boost::iostreams::source_tag {};

public:
    CMockSource(const std::string& s) : m_Data(s), m_Read(0) {}

    std::streamsize read(char* s, std::streamsize n) {
        if (m_Read >= std::streamsize(m_Data.size())) {
            return -1;
        }
        std::streamsize toCopy = std::min(std::streamsize(m_Data.size() - m_Read), n);
        LOG_TRACE(<< "Read " << toCopy << ": "
                  << std::string(m_Data.c_str() + m_Read, toCopy));
        memcpy(s, m_Data.c_str() + m_Read, toCopy);
        m_Read += toCopy;
        return toCopy;
    }

    void close() {}

private:
    std::string m_Data;
    std::streamsize m_Read;
};

// Implements the boost::iostreams Sink template interface
class CMockSink {
public:
    using char_type = char;

    struct category : public boost::iostreams::sink_tag,
                      public boost::iostreams::closable_tag {};

public:
    CMockSink() {}

    std::streamsize write(const char* s, std::streamsize n) {
        m_Data.append(s, n);
        return n;
    }

    void close() {}

    const std::string getData() const { return m_Data; }

private:
    void writeInternal(const char* s, std::streamsize& written, std::streamsize& n);

private:
    std::string m_Data;
};

void testEncodeDecode(const std::string& input) {
    CMockSink sink;
    {
        TFilteredOutput filter;
        filter.push(CBase64Encoder());
        filter.push(boost::ref(sink));
        filter << input;
    }
    CMockSource source(sink.getData());
    {
        TFilteredInput filter;
        filter.push(CBase64Decoder());
        filter.push(boost::ref(source));

        std::istreambuf_iterator<char> eos;
        std::string s(std::istreambuf_iterator<char>(filter), eos);
        BOOST_REQUIRE_EQUAL(input, s);
    }
}
}

BOOST_AUTO_TEST_CASE(testEncode) {
    {
        // Test encode ability, with known test data

        // From Wikipedia
        CMockSink sink;
        {
            TFilteredOutput filter;
            filter.push(CBase64Encoder());
            filter.push(boost::ref(sink));
            filter << "Man is distinguished, not only by his reason, but by this singular passion from ";
            filter << "other animals, which is a lust of the mind, that by a perseverance of delight ";
            filter << "in the continued and indefatigable generation of knowledge, exceeds the short ";
            filter << "vehemence of any carnal pleasure.";
        }
        std::string expected =
            "TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0aGlz"
            "IHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1c3Qgb2Yg"
            "dGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmFuY2Ugb2YgZGVsaWdodCBpbiB0aGUgY29udGlu"
            "dWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdlLCBleGNlZWRzIHRo"
            "ZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4=";
        BOOST_REQUIRE_EQUAL(expected, sink.getData());
    }
    {
        // Test a much longer string
        CMockSink sink;
        {
            TFilteredOutput filter;
            filter.push(CBase64Encoder());
            filter.push(boost::ref(sink));
            for (std::size_t i = 0; i < 50000; i++) {
                filter << "OneTwoThreeFourFiveSixSevenEightNineTen";
            }
        }
        std::ostringstream result;
        for (std::size_t i = 0; i < 50000; i++) {
            result << "T25lVHdvVGhyZWVGb3VyRml2ZVNpeFNldmVuRWlnaHROaW5lVGVu";
        }
        BOOST_REQUIRE_EQUAL(result.str(), sink.getData());
    }
}

BOOST_AUTO_TEST_CASE(testDecode) {
    {
        // Test decoding
        std::string encoded =
            "TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0aGlz"
            "IHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1c3Qgb2Yg"
            "dGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmFuY2Ugb2YgZGVsaWdodCBpbiB0aGUgY29udGlu"
            "dWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdlLCBleGNlZWRzIHRo"
            "ZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4=";
        std::string expected =
            "Man is distinguished, not only by his reason, but by this singular passion from "
            "other animals, which is a lust of the mind, that by a perseverance of delight "
            "in the continued and indefatigable generation of knowledge, exceeds the short "
            "vehemence of any carnal pleasure.";
        CMockSource source(encoded);
        TFilteredInput filter;
        filter.push(CBase64Decoder());
        filter.push(boost::ref(source));
        std::istreambuf_iterator<char> eos;
        std::string s(std::istreambuf_iterator<char>(filter), eos);
        BOOST_REQUIRE_EQUAL(expected, s);
    }
    {
        // Test decoding
        std::string encoded = "0";
        std::string expected = "";
        CMockSource source(encoded);
        TFilteredInput filter;
        filter.push(CBase64Decoder());
        filter.push(boost::ref(source));
        std::istreambuf_iterator<char> eos;
        std::string s(std::istreambuf_iterator<char>(filter), eos);
        BOOST_REQUIRE_EQUAL(expected, s);
    }
}

BOOST_AUTO_TEST_CASE(testBoth) {
    {
        testEncodeDecode("a");
        testEncodeDecode("aa");
        testEncodeDecode("aaa");
        testEncodeDecode("aaaa");
        testEncodeDecode("aaaaa");
        testEncodeDecode("aaaaaa");
        testEncodeDecode("aaaaaaa");
        testEncodeDecode("aaaaaaaa");
        testEncodeDecode("OneTwoThreeFourFiveSixSevenEightNineTen");
    }
    {
        TRandom rng(3632638936UL);
        TGenerator generator(rng, TDistribution(0, 255));
        TGeneratorItr randItr(&generator);

        std::ostringstream ss;
        for (std::size_t i = 0; i < 5000000; i++) {
            ss << char(*randItr++);
        }
        testEncodeDecode(ss.str());
    }
}

BOOST_AUTO_TEST_SUITE_END()
