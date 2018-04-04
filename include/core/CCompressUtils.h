/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CCompressUtils_h
#define INCLUDED_ml_core_CCompressUtils_h

#include <core/CNonCopyable.h>
#include <core/ImportExport.h>

#include <zlib.h>

#include <string>
#include <vector>


namespace ml
{
namespace core
{

//! \brief
//! Shrink wrap zlib calls.
//!
//! DESCRIPTION:\n
//! Shrink wrap zlib calls.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Data can be added incrementally and this 'finished' to
//! complete compression.
//!
//! This object retains in memory the entire compressed state
//! so it not good for file read/write.
//!
//! A single Z stream is used for the lifetime of the object,
//! so each object can only work on one task at a time.  In
//! a multi-threaded application it would be best to create
//! one object for each thread.
//!
class CORE_EXPORT CCompressUtils : private CNonCopyable
{
    public:
        //! The output type
        using TByteVec = std::vector<Bytef>;

    public:
        explicit CCompressUtils(bool lengthOnly,
                                int level = Z_DEFAULT_COMPRESSION);
        ~CCompressUtils();

        //! Add a string.  Multiple calls to this function without finishing the
        //! compression are equivalent to compressing the concatenation of the
        //! strings passed in the order they are passed.
        bool addString(const std::string &input);

        //! Get compressed representation.  This will fail if the lengthOnly
        //! constructor argument was set to true.
        //!
        //! \note The compressed representation is a byte array NOT a string,
        //! and hence not printable.
        //!
        //! If finish==false then retrieve partial compressed state.
        bool compressedData(bool finish, TByteVec &result);

        //! Get compressed string length.
        //!
        //! If finish==false then retrieve partial compressed length.
        bool compressedLength(bool finish, size_t &length);

        //! Reset the compressor.  This will happen automatically when adding a
        //! new string after having finished the previous compression, but
        //! sometimes, for example when recovering from an error, it may be
        //! desirable to explicitly reset the compressor state.
        void reset();

    private:
        bool doCompress(bool finish, const std::string &input);

    private:
        enum EState
        {
            E_Unused,
            E_Compressing,
            E_Finished
        };

        EState   m_State;

        //! Is this object only fit for getting compressed lengths?
        bool     m_LengthOnly;

        //! The output buffer when the compressed result is being stored
        TByteVec m_FullResult;

        //! The zlib data structure.
        z_stream m_ZlibStrm;
};

}
}

#endif // INCLUDED_ml_core_CCompressUtils_h

