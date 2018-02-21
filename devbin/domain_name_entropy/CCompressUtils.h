/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
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
#ifndef INCLUDED_ml_domain_name_entropy_CCompressUtils_h
#define INCLUDED_ml_domain_name_entropy_CCompressUtils_h

#include <string>
#include <vector>

#include <zlib.h>


#include <core/CNonCopyable.h>

namespace ml
{
namespace domain_name_entropy
{

//! \brief
//! Shrink wrap zlib calls.
//!
//! DESCRIPTION:\n
//! Shrink wrap zlib calls.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Implementation based on http://www.zlib.net/zpipe.c
//! 
//! Data can be added incrementally and this 'finished' to
//! complete compression.
//!
//! This object retains in memory the entire compressed state
//! so it not good for file read/write.
//!
class CCompressUtils : private core::CNonCopyable
{
    public:
        CCompressUtils(void);
        ~CCompressUtils(void);

        // --
        // COMPRESS INTERFACE
        // --
        //! Add string. If finish==true, the compressed state is
        //! completely calculated and no further state can be added.
        bool compressString(bool finish, const std::string &buffer);

        //! Get compressed representation
        //! NOTE: the compressed representation is a u_char array
        //! NOT a string.
        //! If finish==false retrieve partial compressed state
        bool compressedString(bool finish, std::string &buffer);

        //! Get compressed string length
        //! If finish==false retrieve partial compressed state
        bool compressedStringLength(bool finish, size_t &length);

    private:
        enum EState
        {
            E_Uninitialized,
            E_Compressing,
            E_Uncompressing,
            E_IsFinished
        };

        EState m_State;

        typedef std::vector<Bytef> TByteVec;

        TByteVec m_Buffer;

        z_stream m_ZlibStrm;
};

}
}

#endif // INCLUDED_ml_domain_name_entropy_CCompressUtils_h
