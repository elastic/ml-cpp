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

#include <core/CJsonStateRestoreTraverser.h>

#include <boost/json.hpp>
#include <boost/test/unit_test.hpp>

#include <sstream>

// This file must be manually included when
// using basic_parser to implement a parser.
#include <boost/json/basic_parser_impl.hpp>

using namespace std;
using namespace boost::json;

// The null parser discards all the data
class null_parser {
    struct handler {
        constexpr static std::size_t max_object_size = std::size_t(-1);
        constexpr static std::size_t max_array_size = std::size_t(-1);
        constexpr static std::size_t max_key_size = std::size_t(-1);
        constexpr static std::size_t max_string_size = std::size_t(-1);

        bool on_document_begin(boost::json::error_code&) {
            LOG_INFO(<< "on_document_begin");
            return true;
        }
        bool on_document_end(boost::json::error_code&) {
            LOG_INFO(<< "on_document_end");
            return true;
        }
        bool on_object_begin(boost::json::error_code&) {
            LOG_INFO(<< "on_object_begin");
            return true;
        }
        bool on_object_end(std::size_t, boost::json::error_code&) {
            LOG_INFO(<< "on_object_end");
            return true;
        }
        bool on_array_begin(boost::json::error_code&) {
            LOG_INFO(<< "on_array_begin");
            return true;
        }
        bool on_array_end(std::size_t, boost::json::error_code&) {
            LOG_INFO(<< "on_array_end");
            return true;
        }
        bool on_key_part(std::string_view str, std::size_t, boost::json::error_code&) {
            LOG_INFO(<< "on_key_part: " << str);
            return true;
        }
        bool on_key(std::string_view str, std::size_t n, boost::json::error_code&) {
            LOG_INFO(<< "on_key: '" << str << "'"
                     << ", size: " << n);
            return true;
        }
        bool on_string_part(std::string_view str, std::size_t, boost::json::error_code&) {
            LOG_INFO(<< "on_string_part: " << str);
            return true;
        }
        bool on_string(std::string_view str, std::size_t n, boost::json::error_code&) {
            LOG_INFO(<< "on_string: '" << str << "'"
                     << ", size: " << n);
            return true;
        }
        bool on_number_part(std::string_view str, boost::json::error_code&) {
            LOG_INFO(<< "on_number_part: " << str);
            return true;
        }
        bool on_int64(std::int64_t, std::string_view str, boost::json::error_code&) {
            LOG_INFO(<< "on_int64: " << str);
            return true;
        }
        bool on_uint64(std::uint64_t, std::string_view str, boost::json::error_code&) {
            LOG_INFO(<< "on_uint64: " << str);
            return true;
        }
        bool on_double(double, std::string_view str, boost::json::error_code&) {
            LOG_INFO(<< "on_double: " << str);
            return true;
        }
        bool on_bool(bool b, boost::json::error_code&) {
            LOG_INFO(<< "on_bool: " << std::boolalpha << b);
            return true;
        }
        bool on_null(boost::json::error_code&) {
            LOG_INFO(<< "on_null: ");
            return true;
        }
        bool on_comment_part(std::string_view str, boost::json::error_code&) {
            LOG_INFO(<< "on_comment_part: " << str);
            return true;
        }
        bool on_comment(std::string_view str, boost::json::error_code&) {
            LOG_INFO(<< "on_comment: " << str);
            return true;
        }
    };

    basic_parser<handler> p_;
    value m_Doc;

public:
    null_parser() : p_(parse_options()) {}

    ~null_parser() {}

    std::size_t write(char const* data, std::size_t size, boost::json::error_code& ec) {
        std::size_t n{0};
        for (int i = 0; i < size - 1; i++) {
            const char* ptr = &data[i];
            n += p_.write_some(true, ptr, 1, ec);
            if (ec) {
                LOG_ERROR(<< "Json parsing of [" << data << "] failed at offset << "
                          << i << ". Parser message: " << ec.message());
                break;
            }
        }

        if (ec) {
            LOG_ERROR(<< "Json parsing failed: " << ec.message());
            return n;
        }

        const char* ptr = &data[size - 1];

        n += p_.write_some(false, ptr, 1, ec);
        if (!ec && n < size) {
            ec = error::extra_data;
        }
        if (ec) {
            LOG_ERROR(<< "Json parsing failed: " << ec.message() << ": " << data);
            return n;
        }

        //        auto const n = p_.write_some( false, data, size, ec );
        //        if(! ec && n < size)
        //            ec = error::extra_data;
        return n;
    }
};

bool validate(std::string_view s) {
    // Parse with the null parser and return false on error
    null_parser p;
    boost::json::error_code ec;
    p.write(s.data(), s.size(), ec);
    if (ec) {
        LOG_ERROR(<< "Error parsing json: " << ec.message());
        return false;
    }

    // The string is valid JSON.
    return true;
}

bool validate(std::istringstream& strm) {
    // Parse with the null parser and return false on error
    const std::string& s = strm.rdbuf()->str();
    null_parser p;
    boost::json::error_code ec;
    p.write(s.c_str(), s.size(), ec);
    if (ec) {
        LOG_ERROR(<< "Error parsing json: " << ec.message());
        return false;
    }

    // The string is valid JSON.
    return true;
}

BOOST_AUTO_TEST_SUITE(CJsonStateRestoreTraverserTest)

namespace {

bool traverse2ndLevel(ml::core::CStateRestoreTraverser& traverser) {
    BOOST_REQUIRE_EQUAL(std::string("level2A"), traverser.name());
    BOOST_REQUIRE_EQUAL(std::string("3.14"), traverser.value());
    BOOST_TEST_REQUIRE(!traverser.hasSubLevel());
    BOOST_TEST_REQUIRE(traverser.next());
    BOOST_REQUIRE_EQUAL(std::string("level2B"), traverser.name());
    BOOST_REQUIRE_EQUAL(std::string("z"), traverser.value());
    BOOST_TEST_REQUIRE(!traverser.hasSubLevel());
    BOOST_TEST_REQUIRE(!traverser.next());

    return true;
}

bool traverse1stLevel1(ml::core::CStateRestoreTraverser& traverser) {
    BOOST_REQUIRE_EQUAL(std::string("level1A"), traverser.name());
    BOOST_REQUIRE_EQUAL(std::string("a"), traverser.value());
    BOOST_TEST_REQUIRE(!traverser.hasSubLevel());
    BOOST_TEST_REQUIRE(traverser.next());
    BOOST_REQUIRE_EQUAL(std::string("level1B"), traverser.name());
    BOOST_REQUIRE_EQUAL(std::string("25"), traverser.value());
    BOOST_TEST_REQUIRE(!traverser.hasSubLevel());
    BOOST_TEST_REQUIRE(traverser.next());
    BOOST_REQUIRE_EQUAL(std::string("level1C"), traverser.name());
    BOOST_TEST_REQUIRE(traverser.hasSubLevel());
    BOOST_TEST_REQUIRE(traverser.traverseSubLevel(&traverse2ndLevel));
    BOOST_TEST_REQUIRE(!traverser.next());

    return true;
}

bool traverse1stLevel2(ml::core::CStateRestoreTraverser& traverser) {
    BOOST_REQUIRE_EQUAL(std::string("level1A"), traverser.name());
    BOOST_REQUIRE_EQUAL(std::string("a"), traverser.value());
    BOOST_TEST_REQUIRE(!traverser.hasSubLevel());
    BOOST_TEST_REQUIRE(traverser.next());
    BOOST_REQUIRE_EQUAL(std::string("level1B"), traverser.name());
    BOOST_REQUIRE_EQUAL(std::string("25"), traverser.value());
    BOOST_TEST_REQUIRE(!traverser.hasSubLevel());
    BOOST_TEST_REQUIRE(traverser.next());
    BOOST_REQUIRE_EQUAL(std::string("level1C"), traverser.name());
    BOOST_TEST_REQUIRE(traverser.hasSubLevel());
    BOOST_TEST_REQUIRE(traverser.traverseSubLevel(&traverse2ndLevel));
    BOOST_TEST_REQUIRE(traverser.next());
    BOOST_REQUIRE_EQUAL(std::string("level1D"), traverser.name());
    BOOST_REQUIRE_EQUAL(std::string("afterAscending"), traverser.value());
    BOOST_TEST_REQUIRE(!traverser.hasSubLevel());
    BOOST_TEST_REQUIRE(!traverser.next());

    return true;
}

bool traverse2ndLevelEmpty(ml::core::CStateRestoreTraverser& traverser) {
    BOOST_TEST_REQUIRE(traverser.name().empty());
    BOOST_TEST_REQUIRE(traverser.value().empty());
    BOOST_TEST_REQUIRE(!traverser.hasSubLevel());
    BOOST_TEST_REQUIRE(!traverser.next());

    return true;
}

bool traverse1stLevel3(ml::core::CStateRestoreTraverser& traverser) {
    BOOST_REQUIRE_EQUAL(std::string("level1A"), traverser.name());
    BOOST_REQUIRE_EQUAL(std::string("a"), traverser.value());
    BOOST_TEST_REQUIRE(!traverser.hasSubLevel());
    BOOST_TEST_REQUIRE(traverser.next());
    BOOST_REQUIRE_EQUAL(std::string("level1B"), traverser.name());
    BOOST_REQUIRE_EQUAL(std::string("25"), traverser.value());
    BOOST_TEST_REQUIRE(!traverser.hasSubLevel());
    BOOST_TEST_REQUIRE(traverser.next());
    BOOST_REQUIRE_EQUAL(std::string("level1C"), traverser.name());
    BOOST_TEST_REQUIRE(traverser.hasSubLevel());
    BOOST_TEST_REQUIRE(traverser.traverseSubLevel(&traverse2ndLevelEmpty));
    BOOST_TEST_REQUIRE(traverser.next());
    BOOST_REQUIRE_EQUAL(std::string("level1D"), traverser.name());
    BOOST_REQUIRE_EQUAL(std::string("afterAscending"), traverser.value());
    BOOST_TEST_REQUIRE(!traverser.hasSubLevel());
    BOOST_TEST_REQUIRE(!traverser.next());

    return true;
}

bool traverse1stLevel4(ml::core::CStateRestoreTraverser& traverser) {
    BOOST_REQUIRE_EQUAL(std::string("level1A"), traverser.name());
    BOOST_REQUIRE_EQUAL(std::string("a"), traverser.value());
    BOOST_TEST_REQUIRE(!traverser.hasSubLevel());
    BOOST_TEST_REQUIRE(traverser.next());
    BOOST_REQUIRE_EQUAL(std::string("level1B"), traverser.name());
    BOOST_REQUIRE_EQUAL(std::string("25"), traverser.value());
    BOOST_TEST_REQUIRE(!traverser.hasSubLevel());
    BOOST_TEST_REQUIRE(traverser.next());
    BOOST_REQUIRE_EQUAL(std::string("level1C"), traverser.name());
    BOOST_TEST_REQUIRE(traverser.hasSubLevel());
    // For this test we ignore the contents of the sub-level
    BOOST_TEST_REQUIRE(!traverser.next());

    return true;
}
}

BOOST_AUTO_TEST_CASE(testRestore1) {
    std::string json("{\"_source\":{\"level1A\":\"a\",\"level1B\":\"25\",\"level1C\":{\"level2A\":\"3.14\",\"level2B\":\"z\"}}}");

    BOOST_TEST_REQUIRE(validate(json));

    std::istringstream strm(json);

    ml::core::CJsonStateRestoreTraverser traverser(strm);

    BOOST_REQUIRE_EQUAL(std::string("_source"), traverser.name());
    BOOST_TEST_REQUIRE(traverser.hasSubLevel());
    BOOST_TEST_REQUIRE(traverser.traverseSubLevel(&traverse1stLevel1));
    BOOST_TEST_REQUIRE(!traverser.next());
}

BOOST_AUTO_TEST_CASE(testRestore2) {
    std::string json("{\"_source\":{\"level1A\":\"a\",\"level1B\":\"25\",\"level1C\":{\"level2A\":\"3.14\",\"level2B\":\"z\"},\"level1D\":"
                     "\"afterAscending\"}}");
    std::istringstream strm(json);

    ml::core::CJsonStateRestoreTraverser traverser(strm);

    BOOST_REQUIRE_EQUAL(std::string("_source"), traverser.name());
    BOOST_TEST_REQUIRE(traverser.hasSubLevel());
    BOOST_TEST_REQUIRE(traverser.traverseSubLevel(&traverse1stLevel2));
    BOOST_TEST_REQUIRE(!traverser.next());
}

BOOST_AUTO_TEST_CASE(testRestore3) {
    std::string json("{\"_source\":{\"level1A\":\"a\",\"level1B\":\"25\",\"level1C\":{},\"level1D\":\"afterAscending\"}}");
    std::istringstream strm(json);

    ml::core::CJsonStateRestoreTraverser traverser(strm);

    BOOST_REQUIRE_EQUAL(std::string("_source"), traverser.name());
    BOOST_TEST_REQUIRE(traverser.hasSubLevel());
    BOOST_TEST_REQUIRE(traverser.traverseSubLevel(&traverse1stLevel3));
    BOOST_TEST_REQUIRE(!traverser.next());
}

BOOST_AUTO_TEST_CASE(testRestore4) {
    std::string json("{\"_source\":{\"level1A\":\"a\",\"level1B\":\"25\",\"level1C\":{\"level2A\":\"3.14\",\"level2B\":\"z\"}}}");
    std::istringstream strm(json);

    ml::core::CJsonStateRestoreTraverser traverser(strm);

    BOOST_REQUIRE_EQUAL(std::string("_source"), traverser.name());
    BOOST_TEST_REQUIRE(traverser.hasSubLevel());
    BOOST_TEST_REQUIRE(traverser.traverseSubLevel(&traverse1stLevel4));
    BOOST_TEST_REQUIRE(!traverser.next());
}

BOOST_AUTO_TEST_CASE(testParsingBooleanFields) {
    // Even though the parser doesn't handle boolean fields it should not hiccup over them
    std::string json =
        std::string("{\"_index\" : \"categorization-test\", \"_type\" : \"categorizerState\",") +
        std::string("\"_id\" : \"1\",  \"_version\" : 2, \"found\" : true, ") +
        std::string("\"_source\":{\"a\" :\"1\"}");

    std::istringstream strm(json);

    ml::core::CJsonStateRestoreTraverser traverser(strm);

    BOOST_REQUIRE_EQUAL(std::string("_index"), traverser.name());
    BOOST_REQUIRE_EQUAL(std::string("categorization-test"), traverser.value());
    BOOST_TEST_REQUIRE(traverser.next());
    BOOST_REQUIRE_EQUAL(std::string("_type"), traverser.name());
    BOOST_REQUIRE_EQUAL(std::string("categorizerState"), traverser.value());
    BOOST_TEST_REQUIRE(traverser.next());
    BOOST_REQUIRE_EQUAL(std::string("_id"), traverser.name());
    BOOST_REQUIRE_EQUAL(std::string("1"), traverser.value());
    BOOST_TEST_REQUIRE(traverser.next());
    BOOST_REQUIRE_EQUAL(std::string("_version"), traverser.name());
    BOOST_REQUIRE_EQUAL(std::string("2"), traverser.value());
    BOOST_TEST_REQUIRE(traverser.next());
    BOOST_REQUIRE_EQUAL(std::string("found"), traverser.name());
    BOOST_REQUIRE_EQUAL(std::string("true"), traverser.value());
    BOOST_TEST_REQUIRE(traverser.next());
    BOOST_REQUIRE_EQUAL(std::string("_source"), traverser.name());
    BOOST_TEST_REQUIRE(traverser.hasSubLevel());
}

BOOST_AUTO_TEST_CASE(testRestore1IgnoreArrays) {
    std::string json("{\"_source\":{\"level1A\":\"a\",\"someArray\":[42],\"level1B\":\"25\",\"level1C\":{\"level2A\":\"3.14\",\"level2B\":\"z\"}}}");
    std::istringstream strm(json);

    ml::core::CJsonStateRestoreTraverser traverser(strm);

    BOOST_REQUIRE_EQUAL(std::string("_source"), traverser.name());
    BOOST_TEST_REQUIRE(traverser.hasSubLevel());
    BOOST_TEST_REQUIRE(traverser.traverseSubLevel(&traverse1stLevel1));
    BOOST_TEST_REQUIRE(!traverser.next());
}

BOOST_AUTO_TEST_CASE(testRestore1IgnoreArraysNested) {
    std::string json("{\"_source\":{\"level1A\":\"a\",\"someArray\":[{\"nestedArray\":[42]}],\"level1B\":\"25\",\"level1C\":{\"level2A\":"
                     "\"3.14\",\"level2B\":\"z\"}}}");
    std::istringstream strm(json);

    ml::core::CJsonStateRestoreTraverser traverser(strm);

    BOOST_REQUIRE_EQUAL(std::string("_source"), traverser.name());
    BOOST_TEST_REQUIRE(traverser.hasSubLevel());
    BOOST_TEST_REQUIRE(traverser.traverseSubLevel(&traverse1stLevel1));
    BOOST_TEST_REQUIRE(!traverser.next());
}

BOOST_AUTO_TEST_SUITE_END()
