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

#include <core/CBoostJsonUnbufferedIStreamWrapper.h>
#include <core/CLogger.h>

#include <boost/json.hpp>

#include <chrono>
#include <istream>
#include <sstream>
#include <string>

namespace boost::json {

std::ostream& operator<<(std::ostream& os, const json::value& doc) {
    return os << json::serialize(doc);
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

//    core::CBoostJsonUnbufferedIStreamWrapper isw{m_StrmIn};

    while (true) {
        json::value doc;
        json::stream_parser p;
        json::error_code ec;
        std::string line;
        while(std::getline(m_StrmIn, line) && !ec) {
            p.write_some(line, ec);
        }

        if (ec) {
            if (m_StrmIn.eof()) {
                break;
            }

            std::ostringstream ss;
            ss << "Error parsing command from JSON: "
               << ec.message();

            errorHandler(UNKNOWN_ID, ss.str());

            return false;
        }

        LOG_TRACE(<< "Inference command: " << doc);
        switch (validateJson(doc.as_object(), errorHandler)) {
        case EMessageType::E_InferenceRequest:
            if (requestHandler(*m_RequestCache, jsonToInferenceRequest(doc.as_object())) == false) {
                LOG_ERROR(<< "Request handler forced exit");
                return false;
            }
            break;
        case EMessageType::E_ControlMessage:
            controlHandler(*m_RequestCache, jsonToControlMessage(doc.as_object()));
            break;
        case EMessageType::E_MalformedMessage:
            continue;
        }
    }

    return true;
}

CCommandParser::EMessageType
CCommandParser::validateJson(const json::object& doc,
                             const TErrorHandlerFunc& errorHandler) {
    if (doc.contains(REQUEST_ID) == false) {
        errorHandler(UNKNOWN_ID, "Invalid command: missing field [" + REQUEST_ID + "]");
        return EMessageType::E_MalformedMessage;
    }

    if (doc.at(REQUEST_ID).is_string() == false) {
        errorHandler(UNKNOWN_ID, "Invalid command: [" + REQUEST_ID + "] field is not a string");
        return EMessageType::E_MalformedMessage;
    }

    if (doc.contains(CONTROL)) {
        return validateControlMessageJson(doc, errorHandler);
    }

    return validateInferenceRequestJson(doc, errorHandler);
}

CCommandParser::EMessageType
CCommandParser::validateControlMessageJson(const json::object& doc,
                                           const TErrorHandlerFunc& errorHandler) {

    const json::value& control = doc.at(CONTROL);
    EControlMessageType controlMessageType =
        (control.is_int64() && control.as_int64() >= 0 &&
         control.as_int64() < EControlMessageType::E_Unknown)
            ? static_cast<EControlMessageType>(control.as_int64())
            : EControlMessageType::E_Unknown;

    switch (controlMessageType) {
    case E_NumberOfAllocations: {
        if (doc.contains(NUM_ALLOCATIONS) == false) {
            errorHandler(UNKNOWN_ID, "Invalid control message: missing field [" +
                                         NUM_ALLOCATIONS + "]");
            return EMessageType::E_MalformedMessage;
        }
        const json::value& numAllocations = doc.at(NUM_ALLOCATIONS);
        if (numAllocations.is_int64() == false) {
            errorHandler(UNKNOWN_ID, "Invalid control message: field [" +
                                         NUM_ALLOCATIONS + "] is not an integer");
            return EMessageType::E_MalformedMessage;
        }
        break;
    }
    case E_ClearCache:
    case E_ProcessStats:
        // No extra arguments needed
        break;
    case E_Unknown:
        errorHandler(UNKNOWN_ID, "Invalid control message: unknown control message type");
        return EMessageType::E_MalformedMessage;
    }

    return EMessageType::E_ControlMessage;
}

CCommandParser::EMessageType
CCommandParser::validateInferenceRequestJson(const json::object& doc,
                                             const TErrorHandlerFunc& errorHandler) {
    if (doc.contains(TOKENS) == false) {
        errorHandler(doc.at(REQUEST_ID).as_string(),
                     "Invalid command: missing field [" + TOKENS + "]");
        return EMessageType::E_MalformedMessage;
    }

    const json::value& tokens = doc.at(TOKENS);
    if (tokens.is_array() == false) {
        errorHandler(doc.at(REQUEST_ID).as_string(),
                     "Invalid command: expected an array of [" + TOKENS + "]");
        return EMessageType::E_MalformedMessage;
    }

    const json::array& outerArray = tokens.as_array();
    for (const auto& val : outerArray) {
        if (val.is_array() == false) {
            errorHandler(doc.at(REQUEST_ID).as_string(),
                         "Invalid command: expected an array of arrays of [" + TOKENS + "]");
            return EMessageType::E_MalformedMessage;
        }

        const json::array& innerArray = val.as_array();
        if (checkArrayContainsUInts(innerArray) == false) {
            errorHandler(doc.at(REQUEST_ID).as_string(),
                         "Invalid command: array [" + TOKENS +
                             "] contains values that are not unsigned integers");
            return EMessageType::E_MalformedMessage;
        }
    }

    // Check optional args.
    std::uint64_t varCount{1};
    std::string varArgName = VAR_ARG_PREFIX + std::to_string(varCount);
    while (doc.contains(varArgName)) {
        const json::value& value = doc.at(varArgName);
        if (value.is_array() == false) {
            errorHandler(doc.at(REQUEST_ID).as_string(),
                         "Invalid command: argument [" + varArgName + "] is not an array");
            return EMessageType::E_MalformedMessage;
        }

        const json::array& outerArgArray = value.as_array();
        for (const auto& val : outerArgArray) {
            if (val.is_array() == false) {
                errorHandler(doc.at(REQUEST_ID).as_string(),
                             "Invalid command: expected an array of arrays of [" +
                                 varArgName + "]");
                return EMessageType::E_MalformedMessage;
            }

            const json::array& innerArgArray = val.as_array();

            if (checkArrayContainsUInts(innerArgArray) == false) {
                errorHandler(doc.at(REQUEST_ID).as_string(),
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

bool CCommandParser::checkArrayContainsUInts(const json::array& arr) {
    return std::find_if(arr.begin(), arr.end(), [](const auto& i) {
               return i.is_uint64() == false;
           }) == arr.end();
}

CCommandParser::SRequest
CCommandParser::jsonToInferenceRequest(const json::object& doc) {
    SRequest request;
    request.s_RequestId = doc.at(REQUEST_ID).as_string();

    // Read 2D array into contiguous memory.
    const auto& tokens = doc.at(TOKENS).as_array();
    request.s_NumberInferences = tokens.size();
    for (const auto& vals : tokens) {
        const auto& innerArray = vals.as_array();
        request.s_NumberInputTokens = innerArray.size();
        request.s_Tokens.reserve(request.s_NumberInferences * request.s_NumberInputTokens);
        for (const auto& val : innerArray) {
            request.s_Tokens.push_back(val.as_uint64());
        }
    }

    std::uint64_t varCount{1};
    std::string varArgName{VAR_ARG_PREFIX + std::to_string(varCount)};

    while (doc.contains(varArgName)) {

        const auto& outerArray = doc.at(varArgName).as_array();
        TUint64Vec arg;
        arg.reserve(request.s_NumberInferences * request.s_NumberInputTokens);
        for (const auto& vals : outerArray) {
            const auto& innerArray = vals.as_array();
            for (const auto& val : innerArray) {
                arg.push_back(val.as_uint64());
            }
        }
        request.s_SecondaryArguments.push_back(std::move(arg));
        ++varCount;
        varArgName = VAR_ARG_PREFIX + std::to_string(varCount);
    }

    return request;
}

CCommandParser::SControlMessage
CCommandParser::jsonToControlMessage(const json::object& doc) {
    auto controlMessageType = static_cast<EControlMessageType>(doc.at(CONTROL).as_int64());
    switch (controlMessageType) {
    case E_NumberOfAllocations:
        return {controlMessageType, doc.at(NUM_ALLOCATIONS).as_int64(),
                doc.at(REQUEST_ID).as_string()};
    case E_ClearCache:
    case E_ProcessStats:
        return {controlMessageType, 0, doc.at(REQUEST_ID).as_string()};
    case E_Unknown:
        break;
    }

    LOG_ABORT(<< "Programmatic error - incorrect validation of control message");
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
