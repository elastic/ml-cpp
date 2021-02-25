/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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
//! Reads JSON documents from a stream calling the request handler
//! for each parsed document.
//!
//! DESCRIPTION:\n
//! Validation on the input documents is light. It is expected the input
//! comes from another process which tightly controls what is sent.
//! Input from an outside source that has not been sanitized should never
//! be sent.
//!
//! IMPLEMENTATION DECISIONS:\n
//! RapidJSON will natively parse a stream of rootless JSON documents
//! given the correct parse flags. The documents may be separated by
//!	whitespace but no other delineator is allowed.
//!
//! The parsed request is a member of this class and will be modified when
//! a new command is parsed. The function handler passed to ioLoop must
//! not keep a reference to the request object beyond the scope of the
//! handle function as the request will change.
//!
//! The input stream is held by reference.  They must outlive objects of
//! this class, which, in practice, means that the CIoManager object managing
//! them must outlive this object.
//!
class CCommandParser {
public:
    static const std::string REQUEST_ID;
    static const std::string TOKENS;
    static const std::string VAR_ARG_PREFIX;

    using TUint64Vec = std::vector<std::uint64_t>;
    using TUint64VecVec = std::vector<TUint64Vec>;

    struct SRequest {
        std::string s_RequestId;
        TUint64Vec s_Tokens;
        TUint64VecVec s_SecondaryArguments;

        void clear();
    };

    using TRequestHandlerFunc = std::function<bool(SRequest&)>;

public:
    CCommandParser(std::istream& strmIn);

    //! Pass input to the processor until it's consumed as much as it can.
    bool ioLoop(const TRequestHandlerFunc& requestHandler);

    CCommandParser(const CCommandParser&) = delete;
    CCommandParser& operator=(const CCommandParser&) = delete;

private:
    bool validateJson(const rapidjson::Document& doc) const;
    bool checkArrayContainsUInts(const rapidjson::Value& arr) const;
    void jsonToRequest(const rapidjson::Document& doc);

private:
    std::istream& m_StrmIn;
    SRequest m_Request;
};
}
}

#endif // INCLUDED_ml_torch_CCommandParser_h
