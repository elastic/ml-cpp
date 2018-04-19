/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_core_CRapidJsonPrettyWriter_h
#define INCLUDED_ml_core_CRapidJsonPrettyWriter_h

#include <core/CRapidJsonWriterBase.h>

#include <rapidjson/prettywriter.h>

namespace ml {
namespace core {

//! Writes each Json object with indentation and spacing
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
class CRapidJsonPrettyWriter
    : public CRapidJsonWriterBase<OUTPUT_STREAM, SOURCE_ENCODING, TARGET_ENCODING, STACK_ALLOCATOR, WRITE_FLAGS, rapidjson::PrettyWriter> {
public:
    using TRapidJsonPrettyWriterBase =
        CRapidJsonWriterBase<OUTPUT_STREAM, SOURCE_ENCODING, TARGET_ENCODING, STACK_ALLOCATOR, WRITE_FLAGS, rapidjson::PrettyWriter>;

    //! inherit the constructors
    //! Note: VS2013 (see #205) does not compile with:
    //! using TRapidJsonPrettyWriterBase::TRapidJsonPrettyWriterBase;
    //! Please remove the following 3 constructors after #205

    //! Constructors
    /*! \param os Output stream.
        */
    explicit CRapidJsonPrettyWriter(OUTPUT_STREAM& os)
        : TRapidJsonPrettyWriterBase(os) {}

    explicit CRapidJsonPrettyWriter() : TRapidJsonPrettyWriterBase() {}

    CRapidJsonPrettyWriter(CRapidJsonPrettyWriter&& rhs)
        : TRapidJsonPrettyWriterBase(std::move(rhs)) {}
};
}
}

#endif /*  INCLUDED_ml_core_CRapidJsonPrettyWriter_h */
