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

#ifndef INCLUDED_ml_torch_CCommandParser_h
#define INCLUDED_ml_torch_CCommandParser_h

#include <rapidjson/document.h>

#include <functional>
#include <iosfwd>
#include <string>
#include <vector>

namespace ml {
namespace torch {

//! \brief
//! Reads JSON documents from a stream calling the appropriate handler
//! for each parsed document.
//!
//! DESCRIPTION:\n
//!
//! IMPLEMENTATION DECISIONS:\n
//! Validation exists to prevent memory violations from malicious input,
//! but no more. The caller is responsible for sending input that will
//! not result in errors from libTorch and will produce meaningful results.
//!
//! RapidJSON will natively parse a stream of rootless JSON documents
//! given the correct parse flags. The documents may be separated by
//!	whitespace but no other delineator is allowed.
//!
//! The parsed request and control message are members of this class and
//! will be modified when a new command is parsed. The function handlers
//! passed to ioLoop must not keep a reference to the request objects beyond
//! the scope of the handle function as the request will change.
//!
//! The input stream is held by reference.  They must outlive objects of
//! this class, which, in practice, means that the CIoManager object managing
//! them must outlive this object.
//!
class CCommandParser {
public:
    static const std::string CONTROL;
    static const std::string NUM_ALLOCATIONS;
    static const std::string RESERVED_REQUEST_ID;
    static const std::string REQUEST_ID;
    static const std::string TOKENS;
    static const std::string VAR_ARG_PREFIX;
    static const std::string UNKNOWN_ID;

    using TUint64Vec = std::vector<std::uint64_t>;
    using TUint64VecVec = std::vector<TUint64Vec>;
    using TDoubleVec = std::vector<double>;

    enum EMessageType {
        E_InferenceRequest,
        E_ControlMessage,
        E_MalformedMessage
    };

    //! The incoming JSON requests contain a 2D array of tokens representing
    //! a batch of inference calls. To avoid copying, the input tensor
    //! should be created directly from contiguous data so the 2D token
    //! array is read into a 1D vector of size w * h where w & h are the
    //! dimensions of in the JSON input. The secondary arguments are
    //! treated in the same manner.
    struct SRequest {
        std::int64_t s_NumberInputTokens;
        std::int64_t s_NumberInferences;
        std::string s_RequestId;
        TUint64Vec s_Tokens;
        TUint64VecVec s_SecondaryArguments;

        void reset();
    };

    struct SControlMessage {
        enum EControlMessageType { E_NumberOfAllocations, E_Unknown };

        EControlMessageType s_MessageType;
        std::int32_t s_NumAllocations;
        std::string s_RequestId;

        void reset();
    };

    using TControlHandlerFunc = std::function<void(SControlMessage&)>;
    using TRequestHandlerFunc = std::function<bool(SRequest&)>;
    using TErrorHandlerFunc =
        std::function<void(const std::string& requestId, const std::string& message)>;

public:
    explicit CCommandParser(std::istream& strmIn);

    //! Pass input to the processor until it's consumed as much as it can.
    //! Parsed requests are passed to the requestHandler, control messages
    //! to the controlHandler and  errors such as a failed validation are
    //! passed to errorHandler
    bool ioLoop(const TRequestHandlerFunc& requestHandler,
                const TControlHandlerFunc& controlHandler,
                const TErrorHandlerFunc& errorHandler);

    CCommandParser(const CCommandParser&) = delete;
    CCommandParser& operator=(const CCommandParser&) = delete;

private:
    static EMessageType validateJson(const rapidjson::Document& doc,
                                     const TErrorHandlerFunc& errorHandler);
    static EMessageType validateInferenceRequestJson(const rapidjson::Document& doc,
                                                     const TErrorHandlerFunc& errorHandler);
    static EMessageType validateControlMessageJson(const rapidjson::Document& doc,
                                                   const TErrorHandlerFunc& errorHandler);
    static bool checkArrayContainsUInts(const rapidjson::Value::ConstArray& arr);
    static bool checkArrayContainsDoubles(const rapidjson::Value::ConstArray& arr);
    void jsonToInferenceRequest(const rapidjson::Document& doc);
    void jsonToControlMessage(const rapidjson::Document& doc);

private:
    std::istream& m_StrmIn;
    SRequest m_Request;
    SControlMessage m_ControlMessage;
};
}
}

#endif // INCLUDED_ml_torch_CCommandParser_h
