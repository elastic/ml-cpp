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

#ifndef INCLUDED_ml_core_CRapidJsonLineWriter_h
#define INCLUDED_ml_core_CRapidJsonLineWriter_h

#include <core/CBoostJsonWriterBase.h>

namespace ml {
namespace core {

//! Writes each Json object to a single line.
//! Not as verbose as rapidjson::prettywriter but it is still possible to
//! parse json data streamed in this format by reading one line at a time
//!
//! \tparam OUTPUT_STREAM Type of output stream.
//! \note implements Handler concept
template<typename OUTPUT_STREAM>
class CBoostJsonLineWriter
    : public CBoostJsonWriterBase<OUTPUT_STREAM> {
public:
    using TBoostJsonWriterBase = CBoostJsonWriterBase<OUTPUT_STREAM>;
    using TBoostJsonWriterBase::TBoostJsonWriterBase;

    CBoostJsonLineWriter(OUTPUT_STREAM&& os) : TBoostJsonWriterBase(os) {}

    //! Overwrites the Writer::StartObject in order to count nested objects
    bool StartObject() override {
        ++m_ObjectCount;
        return TBoostJsonWriterBase::StartObject();
    }

    //! Overwrites Writer::EndObject in order to inject new lines if:
    //! - it's the end of the json object or array
    //! - it's the end of a json object as part of an array
    bool EndObject(std::size_t memberCount = 0) override {
        bool baseReturnCode = TBoostJsonWriterBase::EndObject(memberCount);
        --m_ObjectCount;

        // put a new line if at top level or if inside an array
        if (this->topLevel() || m_ObjectCount == 0) {
            this->put('\n');
        }
        return baseReturnCode;
    }

    //! Add a pre-formatted key and value to the output.
    bool rawKeyAndValue(const std::string& keyAndValue) {
        // We achieve this by first splitting the keyAndValue string
        // on the first occurrence of ":". The substring before the first ":"
        // is the key and the substring after it is the value.
        auto pos = keyAndValue.find(":");
        if (pos == keyAndValue.npos) {
            LOG_ERROR(<< "Invalid JSON snippet: " << keyAndValue);
            return false;
        }
        std::string key = keyAndValue.substr(0, pos);
        std::string innerVal = keyAndValue.substr(pos+1, keyAndValue.length());

        // Unfortunately value_stack does not support pushing raw json objects so
        // we temporarily create innerVal as type string.
        this->Key(key);
        this->String(innerVal);

        // Then, once the parsing is completed and the json value created with
        // value_stack.release() it will be necessary to replace the placeholder
        // innerValue string with the parsed json object, with something like this.
        // json::value jInnerVal = json::parse(innerVal);
        // doc[innerValPath].emplace_object() = jInnerVal;

        return true;
    }

private:
    size_t m_ObjectCount = 0;
};
}
}

#endif // INCLUDED_ml_core_CRapidJsonLineWriter_h
