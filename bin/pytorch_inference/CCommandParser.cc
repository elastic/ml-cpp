/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CCommandParser.h"

#include <core/CLogger.h>

#include <iostream>

#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace ml {
namespace torch {

namespace {
void debug(const rapidjson::Document& doc) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);
    LOG_TRACE(<< buffer.GetString());
}
}

const std::string CCommandParser::REQUEST_ID{"request_id"};
const std::string CCommandParser::TOKENS{"tokens"};
const std::string CCommandParser::VAR_ARG_PREFIX{"arg_"};

CCommandParser::CCommandParser(std::istream& strmIn) : m_StrmIn(strmIn) {
}

bool CCommandParser::ioLoop(const TRequestHandlerFunc& requestHandler) const {

    rapidjson::IStreamWrapper isw(m_StrmIn);

    while (true) {
        rapidjson::Document doc;
        rapidjson::ParseResult parseResult =
            doc.ParseStream<rapidjson::kParseStopWhenDoneFlag>(isw);

        if (static_cast<bool>(parseResult) == false) {
            if (m_StrmIn.eof()) {
                break;
            }

            LOG_ERROR(<< "Error parsing command from JSON: "
                      << rapidjson::GetParseError_En(parseResult.Code())
                      << ". At offset: " << parseResult.Offset());

            return false;
        }

        if (validateJson(doc) == false) {
            continue;
        }

        // TODO if logger.trace_enabled then
        debug(doc);
        CCommandParser::SRequest request = jsonToRequest(doc);
        if (requestHandler(request) == false) {
            LOG_ERROR(<< "Request handler forced exit");
            return false;
        }
    }

    return true;
}

bool CCommandParser::validateJson(const rapidjson::Document& doc) const {
    if (doc.HasMember(REQUEST_ID) == false) {
        LOG_ERROR(<< "Invalid command: missing field [" << REQUEST_ID << "]");
        return false;
    }

    if (doc.HasMember(TOKENS) == false) {
        LOG_ERROR(<< "Invalid command: missing field [" << TOKENS << "]");
        return false;
    }

    const rapidjson::Value& tokens = doc[TOKENS];
    if (tokens.IsArray() == false) {
        LOG_ERROR(<< "Invalid command: expected an array [" << TOKENS << "]");
        return false;
    }

    // check optional args
    std::uint64_t varCount{1};
    std::string varArgName = VAR_ARG_PREFIX + std::to_string(varCount);
    while (doc.HasMember(varArgName)) {
        const rapidjson::Value& value = doc[varArgName];
        if (value.IsArray() == false) {
            LOG_ERROR(<< "Invalid command: argument [" << varArgName << "] is not an array");
            return false;
        }

        ++varCount;
        varArgName = VAR_ARG_PREFIX + std::to_string(varCount);
    }

    return true;
}

CCommandParser::SRequest CCommandParser::jsonToRequest(const rapidjson::Document& doc) const {
    TUint32Vec tokens;
    const rapidjson::Value& arr = doc[TOKENS];
    for (auto itr = arr.Begin(); itr != arr.End(); ++itr) {
        tokens.push_back(itr->GetUint());
    }

    std::uint64_t varCount{1};
    std::string varArgName = VAR_ARG_PREFIX + std::to_string(varCount);
    TUint32VecVec args;

    while (doc.HasMember(varArgName)) {
        TUint32Vec arg;
        const rapidjson::Value& v = doc[varArgName];
        for (auto itr = v.Begin(); itr != v.End(); ++itr) {
            arg.push_back(itr->GetUint());
        }

        args.push_back(arg);
        ++varCount;
        varArgName = VAR_ARG_PREFIX + std::to_string(varCount);
    }
    return {doc[REQUEST_ID].GetString(), tokens, args};
}
}
}
