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

#include "CCommandParser.h"

#include <core/CLogger.h>

#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <istream>
#include <sstream>
#include <string>

namespace rapidjson {

std::ostream& operator<<(std::ostream& os, const rapidjson::Document& doc) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);
    return os << buffer.GetString();
}
}

namespace ml {
namespace torch {

const std::string CCommandParser::REQUEST_ID{"request_id"};
const std::string CCommandParser::TOKENS{"tokens"};
const std::string CCommandParser::VAR_ARG_PREFIX{"arg_"};
const std::string CCommandParser::UNKNOWN_ID;

CCommandParser::CCommandParser(std::istream& strmIn) : m_StrmIn(strmIn) {
}

bool CCommandParser::ioLoop(const TRequestHandlerFunc& requestHandler,
                            const TErrorHandlerFunc& errorHandler) {

    rapidjson::IStreamWrapper isw{m_StrmIn};

    while (true) {
        rapidjson::Document doc;
        rapidjson::ParseResult parseResult =
            doc.ParseStream<rapidjson::kParseStopWhenDoneFlag>(isw);

        if (static_cast<bool>(parseResult) == false) {
            if (m_StrmIn.eof()) {
                break;
            }

            std::ostringstream ss;
            ss << "Error parsing command from JSON: "
               << rapidjson::GetParseError_En(parseResult.Code())
               << ". At offset: " << parseResult.Offset();

            errorHandler(UNKNOWN_ID, ss.str());

            return false;
        }

        if (validateJson(doc, errorHandler) == false) {
            continue;
        }

        LOG_TRACE(<< "Inference command: " << doc);
        jsonToRequest(doc);
        if (requestHandler(m_Request) == false) {
            LOG_ERROR(<< "Request handler forced exit");
            return false;
        }
    }

    return true;
}

bool CCommandParser::validateJson(const rapidjson::Document& doc,
                                  const TErrorHandlerFunc& errorHandler) const {
    if (doc.HasMember(REQUEST_ID) == false) {
        errorHandler(UNKNOWN_ID, "Invalid command: missing field [" + REQUEST_ID + "]");
        return false;
    }

    if (doc[REQUEST_ID].IsString() == false) {
        errorHandler(UNKNOWN_ID, "Invalid command: [" + REQUEST_ID + "] field is not a string");
        return false;
    }

    if (doc.HasMember(TOKENS)) {
        const rapidjson::Value& tokens = doc[TOKENS];
        if (tokens.IsArray() == false) {
            errorHandler(doc[REQUEST_ID].GetString(),
                         "Invalid command: expected an array [" + TOKENS + "]");
            return false;
        }

        for (auto itr = tokens.Begin(); itr != tokens.End(); ++itr) {
            const rapidjson::Value::ConstArray& innerArray = itr->GetArray();
            if (checkArrayContainsUInts(innerArray) == false) {
                errorHandler(doc[REQUEST_ID].GetString(),
                             "Invalid command: array [" + TOKENS +
                                 "] contains values that are not unsigned integers");
                return false;
            }
        }
    } else {
        errorHandler(doc[REQUEST_ID].GetString(),
                     "Invalid command: missing field [" + TOKENS + "]");
        return false;
    }

    // check optional args
    std::uint64_t varCount{1};
    std::string varArgName = VAR_ARG_PREFIX + std::to_string(varCount);
    while (doc.HasMember(varArgName)) {
        const rapidjson::Value& value = doc[varArgName];
        if (value.IsArray() == false) {
            errorHandler(doc[REQUEST_ID].GetString(),
                         "Invalid command: argument [" + varArgName + "] is not an array");
            return false;
        }

        for (auto itr = value.Begin(); itr != value.End(); ++itr) {
            const rapidjson::Value::ConstArray& innerArray = itr->GetArray();

            if (checkArrayContainsUInts(innerArray) == false) {
                errorHandler(doc[REQUEST_ID].GetString(),
                             "Invalid command: array [" + varArgName +
                                 "] contains values that are not unsigned integers");
                return false;
            }
        }

        ++varCount;
        varArgName = VAR_ARG_PREFIX + std::to_string(varCount);
    }

    return true;
}

bool CCommandParser::checkArrayContainsUInts(const rapidjson::Value::ConstArray& arr) {
    bool allInts{true};

    for (auto itr = arr.Begin(); itr != arr.End(); ++itr) {
        allInts = allInts && itr->IsUint64();
    }

    return allInts;
}

bool CCommandParser::checkArrayContainsDoubles(const rapidjson::Value::ConstArray& arr) {
    bool allDoubles{true};

    for (auto itr = arr.Begin(); itr != arr.End(); ++itr) {
        allDoubles = allDoubles && itr->IsDouble();
    }

    return allDoubles;
}

void CCommandParser::jsonToRequest(const rapidjson::Document& doc) {

    m_Request.s_RequestId = doc[REQUEST_ID].GetString();

    // wipe any previous
    m_Request.s_Tokens.clear();
    if (doc.HasMember(TOKENS)) {
        // read 2D array into contiguous memory
        const rapidjson::Value& arr = doc[TOKENS];
        m_Request.h = arr.Size();
        for (auto itr = arr.Begin(); itr != arr.End(); ++itr) {
            const auto innerArray = itr->GetArray();
            m_Request.w = innerArray.Size();
            m_Request.s_Tokens.reserve(m_Request.h * m_Request.w);

            for (const auto* innerItr = innerArray.Begin();
                 innerItr != innerArray.End(); ++innerItr) {
                m_Request.s_Tokens.push_back(innerItr->GetUint64());
            }
        }
    }

    std::uint64_t varCount{1};
    std::string varArgName = VAR_ARG_PREFIX + std::to_string(varCount);

    // wipe any previous
    m_Request.s_SecondaryArguments.clear();
    while (doc.HasMember(varArgName)) {

        const rapidjson::Value& arr = doc[varArgName];
        TUint64Vec arg;
        arg.reserve(m_Request.h * m_Request.w);
        for (auto itr = arr.Begin(); itr != arr.End(); ++itr) {
            const auto innerArray = itr->GetArray();
            for (const auto* innerItr = innerArray.Begin();
                 innerItr != innerArray.End(); ++innerItr) {
                arg.push_back(innerItr->GetUint64());
            }
        }
        m_Request.s_SecondaryArguments.push_back(std::move(arg));
        ++varCount;
        varArgName = VAR_ARG_PREFIX + std::to_string(varCount);
    }
}

bool CCommandParser::SRequest::hasTokens() const {
    return s_Tokens.empty() == false;
}
}
}
