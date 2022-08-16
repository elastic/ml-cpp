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

#include <core/CRapidJsonWriterBase.h>

namespace ml {
namespace core {

//! Writes each Json object to a single line.
//! Not as verbose as rapidjson::prettywriter but it is still possible to
//! parse json data streamed in this format by reading one line at a time
//!
//! \tparam OUTPUT_STREAM Type of output stream.
//! \tparam SOURCE_ENCODING Encoding of source string.
//! \tparam TARGET_ENCODING Encoding of output stream.
//! \tparam STACK_ALLOCATOR Type of allocator for allocating memory of stack.
//! \note implements Handler concept
template<typename OUTPUT_STREAM,
         typename SOURCE_ENCODING = rapidjson::UTF8<>,
         typename TARGET_ENCODING = rapidjson::UTF8<>,
         typename STACK_ALLOCATOR = rapidjson::CrtAllocator,
         unsigned WRITE_FLAGS = rapidjson::kWriteDefaultFlags>
class CRapidJsonLineWriter
    : public CRapidJsonWriterBase<OUTPUT_STREAM, SOURCE_ENCODING, TARGET_ENCODING, STACK_ALLOCATOR, WRITE_FLAGS, rapidjson::Writer> {
public:
    using TRapidJsonWriterBase =
        CRapidJsonWriterBase<OUTPUT_STREAM, SOURCE_ENCODING, TARGET_ENCODING, STACK_ALLOCATOR, WRITE_FLAGS, rapidjson::Writer>;
    using TRapidJsonWriterBase::TRapidJsonWriterBase;

    //! Overwrites the Writer::StartObject in order to count nested objects
    bool StartObject() {
        ++m_ObjectCount;
        return TRapidJsonWriterBase::StartObject();
    }

    //! Overwrites Writer::EndObject in order to inject new lines if:
    //! - it's the end of the json object or array
    //! - it's the end of a json object as part of an array
    bool EndObject(rapidjson::SizeType memberCount = 0) {
        bool baseReturnCode = TRapidJsonWriterBase::EndObject(memberCount);
        --m_ObjectCount;

        // put a new line if at top level or if inside an array
        if (TRapidJsonWriterBase::level_stack_.Empty() || m_ObjectCount == 0) {
            TRapidJsonWriterBase::os_->Put('\n');
        }
        return baseReturnCode;
    }

    //! Add a pre-formatted key and value to the output.
    bool rawKeyAndValue(const std::string& keyAndValue) {
        // We achieve this by pretending we're just adding the key, i.e.
        // a string, but since it's written raw it can contain both key
        // and value.
        if (this->RawValue(keyAndValue.c_str(), keyAndValue.length(), rapidjson::kStringType)) {
            // However, to avoiding tripping assertions we need to increment
            // the count of values within the level by an extra 1 so that it
            // includes the value that was bundled with the key. (The
            // RawValue() call above will have added 1 for the key.)
            TRapidJsonWriterBase::level_stack_
                .template Top<typename TRapidJsonWriterBase::Level>()
                ->valueCount++;
            return true;
        }
        return false;
    }

    //! Write JSON document to outputstream
    //! \note This overwrite is needed because the members of rapidjson::Writer
    //! are not virtual and we need to avoid "slicing" the writer to ensure that
    //! that the correct StartObject/EndObject functions are called when this is
    //! passed to \p doc Accept.
    void write(const rapidjson::Value& doc) override { doc.Accept(*this); }

private:
    size_t m_ObjectCount = 0;
};
}
}

#endif // INCLUDED_ml_core_CRapidJsonLineWriter_h
