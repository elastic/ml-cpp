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

    if (doc.HasMember(TOKENS) == false) {
        errorHandler(doc[REQUEST_ID].GetString(),
                     "Invalid command: missing field [" + TOKENS + "]");
        return false;
    }

    const rapidjson::Value& tokens = doc[TOKENS];
    if (tokens.IsArray() == false) {
        errorHandler(doc[REQUEST_ID].GetString(),
                     "Invalid command: expected an array of [" + TOKENS + "]");
        return false;
    }

    const rapidjson::Value::ConstArray& outerArray = tokens.GetArray();
    for (const auto& val : outerArray) {
        if (val.IsArray() == false) {
            errorHandler(doc[REQUEST_ID].GetString(),
                         "Invalid command: expected an array of arrays of [" + TOKENS + "]");
            return false;
        }

        const rapidjson::Value::ConstArray& innerArray = val.GetArray();
        if (checkArrayContainsUInts(innerArray) == false) {
            errorHandler(doc[REQUEST_ID].GetString(),
                         "Invalid command: array [" + TOKENS +
                             "] contains values that are not unsigned integers");
            return false;
        }
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

        const rapidjson::Value::ConstArray& outerArgArray = value.GetArray();
        for (const auto& val : outerArgArray) {
            if (val.IsArray() == false) {
                errorHandler(doc[REQUEST_ID].GetString(),
                             "Invalid command: expected an array of arrays of [" +
                                 varArgName + "]");
                return false;
            }

            const rapidjson::Value::ConstArray& innerArgArray = val.GetArray();

            if (checkArrayContainsUInts(innerArgArray) == false) {
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
    return std::find_if(arr.Begin(), arr.End(), [](const auto& i) {
               return i.IsUint64() == false;
           }) == arr.End();
}

bool CCommandParser::checkArrayContainsDoubles(const rapidjson::Value::ConstArray& arr) {
    return std::find_if(arr.Begin(), arr.End(), [](const auto& i) {
               return i.IsDouble() == false;
           }) == arr.End();
}

void CCommandParser::jsonToRequest(const rapidjson::Document& doc) {
    // wipe any previous
    m_Request.reset();
    m_Request.s_RequestId = doc[REQUEST_ID].GetString();

    // read 2D array into contiguous memory
    const rapidjson::Value::ConstArray& tokens = doc[TOKENS].GetArray();
    m_Request.s_NumberInferences = tokens.Size();
    for (const auto& itr : tokens) {
        const auto& innerArray = itr.GetArray();
        m_Request.s_NumberInputTokens = innerArray.Size();
        m_Request.s_Tokens.reserve(m_Request.s_NumberInferences * m_Request.s_NumberInputTokens);

        for (const auto& val : innerArray) {
            m_Request.s_Tokens.push_back(val.GetUint64());
        }
    }

    std::uint64_t varCount{1};
    std::string varArgName = VAR_ARG_PREFIX + std::to_string(varCount);
    
    while (doc.HasMember(varArgName)) {

        const rapidjson::Value::ConstArray& outer = doc[varArgName].GetArray();
        TUint64Vec arg;
        arg.reserve(m_Request.s_NumberInferences * m_Request.s_NumberInputTokens);
        for (const auto& val : outer) {
            const auto& innerArray = val.GetArray();
            for (const auto& e : innerArray) {
                arg.push_back(e.GetUint64());
            }
        }
        m_Request.s_SecondaryArguments.push_back(std::move(arg));
        ++varCount;
        varArgName = VAR_ARG_PREFIX + std::to_string(varCount);
    }
}

void CCommandParser::SRequest::reset() {
    s_NumberInputTokens = 0;
    s_NumberInferences = 0;
    s_RequestId.clear();
    s_Tokens.clear();
    s_SecondaryArguments.clear();
}
}
}
