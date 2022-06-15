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

#include <core/CCompressedLfuCache.h>
#include <core/CLogger.h>
#include <core/CRapidJsonUnbufferedIStreamWrapper.h>

#include <rapidjson/error/en.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <chrono>
#include <istream>
#include <sstream>
#include <string>

#ifdef Windows
// rapidjson::Writer<rapidjson::StringBuffer> gets instantiated in the core
// library, and on Windows it gets exported too, because
// CRapidJsonConcurrentLineWriter inherits from it and is also exported.
// To avoid breaching the one-definition rule we must reuse this exported
// instantiation, as deduplication of template instantiations doesn't work
// across DLLs.  To make this even more confusing, this is only strictly
// necessary when building without optimisation, because with optimisation
// enabled the instantiation in this library gets inlined to the extent that
// there are no clashing symbols.
template class CORE_EXPORT rapidjson::Writer<rapidjson::StringBuffer>;
#endif

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

const std::string CCommandParser::CONTROL{"control"};
const std::string CCommandParser::NUM_ALLOCATIONS{"num_allocations"};
const std::string CCommandParser::RESERVED_REQUEST_ID{"ignore"};
const std::string CCommandParser::REQUEST_ID{"request_id"};
const std::string CCommandParser::TOKENS{"tokens"};
const std::string CCommandParser::VAR_ARG_PREFIX{"arg_"};
const std::string CCommandParser::UNKNOWN_ID;

CCommandParser::CCommandParser(std::istream& strmIn, std::size_t cacheMemoryLimitBytes)
    : m_StrmIn{strmIn} {
    if (cacheMemoryLimitBytes > 0) {
        m_RequestCache = std::make_unique<CRequestCache>(cacheMemoryLimitBytes);
    } else {
        m_RequestCache = std::make_unique<CRequestCacheStub>();
    }
}

bool CCommandParser::ioLoop(const TRequestHandlerFunc& requestHandler,
                            const TControlHandlerFunc& controlHandler,
                            const TErrorHandlerFunc& errorHandler) {

    core::CRapidJsonUnbufferedIStreamWrapper isw{m_StrmIn};

    while (true) {
        rapidjson::Document doc;
        rapidjson::ParseResult parseResult{
            doc.ParseStream<rapidjson::kParseStopWhenDoneFlag>(isw)};

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

        LOG_TRACE(<< "Inference command: " << doc);
        switch (validateJson(doc, errorHandler)) {
        case EMessageType::E_InferenceRequest:
            if (requestHandler(*m_RequestCache, jsonToInferenceRequest(doc)) == false) {
                LOG_ERROR(<< "Request handler forced exit");
                return false;
            }
            break;
        case EMessageType::E_ControlMessage:
            controlHandler(jsonToControlMessage(doc));
            break;
        case EMessageType::E_MalformedMessage:
            continue;
        }
    }

    return true;
}

CCommandParser::EMessageType
CCommandParser::validateJson(const rapidjson::Document& doc,
                             const TErrorHandlerFunc& errorHandler) {
    if (doc.HasMember(REQUEST_ID) == false) {
        errorHandler(UNKNOWN_ID, "Invalid command: missing field [" + REQUEST_ID + "]");
        return EMessageType::E_MalformedMessage;
    }

    if (doc[REQUEST_ID].IsString() == false) {
        errorHandler(UNKNOWN_ID, "Invalid command: [" + REQUEST_ID + "] field is not a string");
        return EMessageType::E_MalformedMessage;
    }

    if (doc.HasMember(CONTROL)) {
        return validateControlMessageJson(doc, errorHandler);
    }

    return validateInferenceRequestJson(doc, errorHandler);
}

CCommandParser::EMessageType
CCommandParser::validateControlMessageJson(const rapidjson::Document& doc,
                                           const TErrorHandlerFunc& errorHandler) {

    const rapidjson::Value& control = doc[CONTROL];
    if (control.IsInt() == false || control.GetInt() < 0 ||
        control.GetInt() >= EControlMessageType::E_Unknown) {
        errorHandler(UNKNOWN_ID, "Invalid control message: unknown control message type");
        return EMessageType::E_MalformedMessage;
    }

    if (doc.HasMember(NUM_ALLOCATIONS) == false) {
        errorHandler(UNKNOWN_ID, "Invalid control message: missing field [" +
                                     NUM_ALLOCATIONS + "]");
        return EMessageType::E_MalformedMessage;
    }

    const rapidjson::Value& numThreads = doc[NUM_ALLOCATIONS];
    if (numThreads.IsInt() == false) {
        errorHandler(UNKNOWN_ID, "Invalid control message: field [" +
                                     NUM_ALLOCATIONS + "] is not an integer");
        return EMessageType::E_MalformedMessage;
    }

    return EMessageType::E_ControlMessage;
}

CCommandParser::EMessageType
CCommandParser::validateInferenceRequestJson(const rapidjson::Document& doc,
                                             const TErrorHandlerFunc& errorHandler) {
    if (doc.HasMember(TOKENS) == false) {
        errorHandler(doc[REQUEST_ID].GetString(),
                     "Invalid command: missing field [" + TOKENS + "]");
        return EMessageType::E_MalformedMessage;
    }

    const rapidjson::Value& tokens = doc[TOKENS];
    if (tokens.IsArray() == false) {
        errorHandler(doc[REQUEST_ID].GetString(),
                     "Invalid command: expected an array of [" + TOKENS + "]");
        return EMessageType::E_MalformedMessage;
    }

    const rapidjson::Value::ConstArray& outerArray = tokens.GetArray();
    for (const auto& val : outerArray) {
        if (val.IsArray() == false) {
            errorHandler(doc[REQUEST_ID].GetString(),
                         "Invalid command: expected an array of arrays of [" + TOKENS + "]");
            return EMessageType::E_MalformedMessage;
        }

        const rapidjson::Value::ConstArray& innerArray = val.GetArray();
        if (checkArrayContainsUInts(innerArray) == false) {
            errorHandler(doc[REQUEST_ID].GetString(),
                         "Invalid command: array [" + TOKENS +
                             "] contains values that are not unsigned integers");
            return EMessageType::E_MalformedMessage;
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
            return EMessageType::E_MalformedMessage;
        }

        const rapidjson::Value::ConstArray& outerArgArray = value.GetArray();
        for (const auto& val : outerArgArray) {
            if (val.IsArray() == false) {
                errorHandler(doc[REQUEST_ID].GetString(),
                             "Invalid command: expected an array of arrays of [" +
                                 varArgName + "]");
                return EMessageType::E_MalformedMessage;
            }

            const rapidjson::Value::ConstArray& innerArgArray = val.GetArray();

            if (checkArrayContainsUInts(innerArgArray) == false) {
                errorHandler(doc[REQUEST_ID].GetString(),
                             "Invalid command: array [" + varArgName +
                                 "] contains values that are not unsigned integers");
                return EMessageType::E_MalformedMessage;
            }
        }

        ++varCount;
        varArgName = VAR_ARG_PREFIX + std::to_string(varCount);
    }

    return EMessageType::E_InferenceRequest;
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

CCommandParser::SRequest
CCommandParser::jsonToInferenceRequest(const rapidjson::Document& doc) {
    SRequest request;
    request.s_RequestId = doc[REQUEST_ID].GetString();

    // Read 2D array into contiguous memory.
    const auto& tokens = doc[TOKENS].GetArray();
    request.s_NumberInferences = tokens.Size();
    for (const auto& vals : tokens) {
        const auto& innerArray = vals.GetArray();
        request.s_NumberInputTokens = innerArray.Size();
        request.s_Tokens.reserve(request.s_NumberInferences * request.s_NumberInputTokens);
        for (const auto& val : innerArray) {
            request.s_Tokens.push_back(val.GetUint64());
        }
    }

    std::uint64_t varCount{1};
    std::string varArgName{VAR_ARG_PREFIX + std::to_string(varCount)};

    while (doc.HasMember(varArgName)) {

        const auto& outerArray = doc[varArgName].GetArray();
        TUint64Vec arg;
        arg.reserve(request.s_NumberInferences * request.s_NumberInputTokens);
        for (const auto& vals : outerArray) {
            const auto& innerArray = vals.GetArray();
            for (const auto& val : innerArray) {
                arg.push_back(val.GetUint64());
            }
        }
        request.s_SecondaryArguments.push_back(std::move(arg));
        ++varCount;
        varArgName = VAR_ARG_PREFIX + std::to_string(varCount);
    }

    return request;
}

CCommandParser::SControlMessage
CCommandParser::jsonToControlMessage(const rapidjson::Document& doc) {
    return {static_cast<EControlMessageType>(doc[CONTROL].GetInt()),
            doc[NUM_ALLOCATIONS].GetInt(), doc[REQUEST_ID].GetString()};
}

CCommandParser::CRequestCache::CRequestCache(std::size_t memoryLimitBytes)
    : m_Impl{memoryLimitBytes, std::chrono::milliseconds{100},
             [](const auto& dictionary, const auto& request) {
                 auto translator = dictionary.translator();
                 translator.add(request.s_NumberInputTokens);
                 translator.add(request.s_NumberInferences);
                 translator.add(request.s_Tokens);
                 for (const auto& argument : request.s_SecondaryArguments) {
                     translator.add(argument);
                 }
                 return translator.word();
             }} {
}
}
}
