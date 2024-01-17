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
        if (m_ObjectCount++ == 0) {
            return this->StartDocument();
        }

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
        return this->WriteRawValue(keyAndValue);
    }

private:
    size_t m_ObjectCount = 0;
};
}
}

#endif // INCLUDED_ml_core_CRapidJsonLineWriter_h
