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

#ifndef INCLUDED_ml_core_CBoostJsonLineWriter_h
#define INCLUDED_ml_core_CBoostJsonLineWriter_h

#include <core/CBoostJsonWriterBase.h>

namespace ml {
namespace core {

//! Writes each Json object to a single line.
//! It is possible to parse json data streamed
//! in this format by reading one line at a time
//!
//! \tparam OUTPUT_STREAM Type of output stream.
//! \note implements Handler concept
template<typename OUTPUT_STREAM>
class CBoostJsonLineWriter : public CBoostJsonWriterBase<OUTPUT_STREAM> {
public:
    using TBoostJsonWriterBase = CBoostJsonWriterBase<OUTPUT_STREAM>;
    using TBoostJsonWriterBase::TBoostJsonWriterBase;

    CBoostJsonLineWriter(OUTPUT_STREAM&& os) : TBoostJsonWriterBase(os) {}

    //! Overwrites the Writer::onObjectBegin in order to count nested objects
    bool onObjectBegin() override {
        if (m_ObjectCount++ == 0) {
            return this->onDocumentBegin();
        }

        return TBoostJsonWriterBase::onObjectBegin();
    }

    //! Overwrites Writer::onObjectEnd in order to inject new lines if:
    //! - it's the end of the json object or array
    //! - it's the end of a json object as part of an array
    bool onObjectEnd(std::size_t memberCount = 0) override {
        bool baseReturnCode = TBoostJsonWriterBase::onObjectEnd(memberCount);
        --m_ObjectCount;

        // put a new line if at top level or if inside an array
        if (this->topLevel() || m_ObjectCount == 0) {
            this->put('\n');
        }
        return baseReturnCode;
    }

    //! Add a pre-formatted key and value to the output.
    bool rawKeyAndValue(const std::string& keyAndValue) {
        // This pre-supposes that the current container is an object.
        if (this->isObject() == false) {
            return false;
        }
        this->append((this->isComplete() ? "" : ","));
        this->append(keyAndValue);
        return true;
    }

private:
    size_t m_ObjectCount = 0;
};
}
}

#endif // INCLUDED_ml_core_CBoostJsonLineWriter_h
