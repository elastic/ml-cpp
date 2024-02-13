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
#ifndef INCLUDED_ml_core_CBoostJsonParser_h
#define INCLUDED_ml_core_CBoostJsonParser_h

#include <core/CLogger.h>
#include <core/ImportExport.h>

#include <boost/json.hpp>

namespace json = boost::json;

namespace ml {
namespace core {

//! \brief
//! Simple wrapper around the boost::json::parse function
//!
//! DESCRIPTION:\n
//! Simple wrapper around the boost::json::parse function
//!
//! https://www.boost.org/doc/libs/1_83_0/libs/json/doc/html/
//!
//! IMPLEMENTATION DECISIONS:\n
//! Makes use of small monotonic stack buffer similar to example:
//! https://www.boost.org/doc/libs/1_83_0/libs/json/doc/html/json/allocators/storage_ptr.html
//!
class CORE_EXPORT CBoostJsonParser {
public:
    static constexpr std::size_t JSON_PARSE_BUFFER_SIZE{8192};

    static bool parse(const std::string& jsonString, json::value& doc) {
        unsigned char buffer[JSON_PARSE_BUFFER_SIZE]; // Small stack buffer to avoid most allocations during parse
        json::monotonic_resource mr(buffer); // This resource will use our local buffer first
        json::error_code ec;
        doc = json::parse(jsonString, ec, &mr);
        if (ec) {
            LOG_ERROR(<< "An error occurred while parsing JSON: \""
                      << jsonString << "\" :" << ec.message());
            return false;
        }
        return true;
    }

    static json::error_code parse(std::istream& istream, json::value& doc) {
        json::stream_parser p;

        unsigned char buf[JSON_PARSE_BUFFER_SIZE]; // Now we need a buffer to hold the actual JSON values
        json::monotonic_resource mr(buf); // The static resource is monotonic, using only a caller-provided buffer
        p.reset(&mr); // Use the static resource for producing the value

        json::error_code ec;
        std::string line;
        while (std::getline(istream, line)) {
            LOG_TRACE(<< "write_some: " << line);
            p.write_some(line);
        }
        p.finish(ec);

        doc = p.release();

        return ec;
    }

    static json::error_code parse(char* begin, std::size_t length, json::value& doc) {
        json::stream_parser p;

        unsigned char buf[JSON_PARSE_BUFFER_SIZE]; // Now we need a buffer to hold the actual JSON values
        json::monotonic_resource mr(buf); // The static resource is monotonic, using only a caller-provided buffer
        p.reset(&mr); // Use the static resource for producing the value

        json::error_code ec;
        std::size_t written{0};
        p.reset();
        while (written < length) {
            written += p.write_some(begin, length, ec);
        }

        p.finish();

        doc = p.release();

        return ec;
    }
};
}
}

#endif // INCLUDED_ml_core_CBoostJsonParser_h
