/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#ifndef INCLUDED_ml_core_CRapidJsonLineWriter_h
#define INCLUDED_ml_core_CRapidJsonLineWriter_h

#include <core/CRapidJsonWriterBase.h>

namespace ml {
namespace core {

//! Writes each Json object to a single line.
//! Not as verbose as rapidjson::prettywriter but it is still possible to
//! parse json data streamed in this format by reading one line at a time
/*!
    \tparam OUTPUT_STREAM Type of output stream.
    \tparam SOURCE_ENCODING Encoding of source string.
    \tparam TARGET_ENCODING Encoding of output stream.
    \tparam STACK_ALLOCATOR Type of allocator for allocating memory of stack.
    \note implements Handler concept
*/
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

    //! inherit the constructors
    //! Note: VS2013 (see #205) does not compile with:
    //! using TRapidJsonWriterBase::TRapidJsonWriterBase;
    //! Please remove the following 3 constructors after #205

    //! Constructors
    /*! \param os Output stream.
        */
    explicit CRapidJsonLineWriter(OUTPUT_STREAM& os) : TRapidJsonWriterBase(os) {}

    explicit CRapidJsonLineWriter() : TRapidJsonWriterBase() {}

    CRapidJsonLineWriter(CRapidJsonLineWriter&& rhs) : TRapidJsonWriterBase(std::move(rhs)) {}

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

    //! Write JSON document to outputstream
    //! Note this non-virtual overwrite is needed to avoid slicing of the writer
    //! and hence ensure the correct StartObject/EndObject functions are called
    //! \p doc reference to rapidjson document value
    void write(rapidjson::Value& doc) { doc.Accept(*this); }

private:
    size_t m_ObjectCount = 0;
};
}
}

#endif /* INCLUDE_CORE_CRAPIDJSONLINEWRITER_H_ */
