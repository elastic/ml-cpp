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

#include <core/CCompressedLfuCache.h>

#include <functional>
#include <iosfwd>
#include <memory>
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
//! not result in errors from LibTorch and will produce meaningful results.
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
    using TUint64Vec = std::vector<std::uint64_t>;
    using TUint64VecVec = std::vector<TUint64Vec>;
    struct SRequest;

    //! \brief Inference request cache interface.
    class CRequestCacheInterface {
    public:
        using TComputeResponse = std::function<std::string(SRequest)>;
        using TReadResponse = std::function<void(const std::string&)>;

    public:
        virtual ~CRequestCacheInterface() = default;
        virtual void resize(std::size_t memoryLimitBytes) = 0;
        virtual bool lookup(SRequest request,
                            const TComputeResponse& computeResponse,
                            const TReadResponse& readResponse) = 0;
        virtual void clear() = 0;
    };

    //! \brief Memory limited inference request LFU cache.
    class CRequestCache : public CRequestCacheInterface {
    public:
        explicit CRequestCache(std::size_t memoryLimitBytes);

        void resize(std::size_t memoryLimitBytes) override {
            m_Impl.resize(memoryLimitBytes);
        }

        bool lookup(SRequest request,
                    const TComputeResponse& computeResponse,
                    const TReadResponse& readResponse) override {
            return m_Impl.lookup(std::move(request), computeResponse, readResponse);
        }

        void clear() override { m_Impl.clear(); }

    private:
        using TConcurrentLfuCache = core::CConcurrentCompressedLfuCache<SRequest, std::string>;

    private:
        TConcurrentLfuCache m_Impl;
    };

    //! \brief Stub cache.
    class CRequestCacheStub : public CRequestCacheInterface {
    public:
        void resize(std::size_t) override {}

        bool lookup(SRequest request,
                    const TComputeResponse& computeResponse,
                    const TReadResponse& readResponse) override {
            readResponse(computeResponse(std::move(request)));
            return false;
        }

        void clear() override {}
    };

    using TRequestCachePtr = std::unique_ptr<CRequestCacheInterface>;

    enum EMessageType {
        E_InferenceRequest,
        E_ControlMessage,
        E_MalformedMessage
    };
    enum EControlMessageType { E_NumberOfAllocations, E_ClearCache, E_Unknown };

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
    };

    //! Controls the process behaviour.
    struct SControlMessage {
        EControlMessageType s_MessageType;
        std::int32_t s_NumAllocations;
        std::string s_RequestId;
    };

    using TControlHandlerFunc =
        std::function<void(CRequestCacheInterface&, const SControlMessage&)>;
    using TRequestHandlerFunc = std::function<bool(CRequestCacheInterface&, SRequest)>;
    using TErrorHandlerFunc =
        std::function<void(const std::string& requestId, const std::string& message)>;

public:
    static const std::string REQUEST_ID;
    static const std::string RESERVED_REQUEST_ID;
    static const std::string UNKNOWN_ID;

public:
    CCommandParser(std::istream& strmIn, std::size_t cacheMemoryLimitBytes);

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
    static const std::string CONTROL;
    static const std::string NUM_ALLOCATIONS;
    static const std::string TOKENS;
    static const std::string VAR_ARG_PREFIX;

private:
    static EMessageType validateJson(const rapidjson::Document& doc,
                                     const TErrorHandlerFunc& errorHandler);
    static EMessageType validateInferenceRequestJson(const rapidjson::Document& doc,
                                                     const TErrorHandlerFunc& errorHandler);
    static EMessageType validateControlMessageJson(const rapidjson::Document& doc,
                                                   const TErrorHandlerFunc& errorHandler);
    static bool checkArrayContainsUInts(const rapidjson::Value::ConstArray& arr);
    static bool checkArrayContainsDoubles(const rapidjson::Value::ConstArray& arr);
    static SRequest jsonToInferenceRequest(const rapidjson::Document& doc);
    static SControlMessage jsonToControlMessage(const rapidjson::Document& doc);

private:
    std::istream& m_StrmIn;
    TRequestCachePtr m_RequestCache;
};
}
}

#endif // INCLUDED_ml_torch_CCommandParser_h
