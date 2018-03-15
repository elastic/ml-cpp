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
#include "CCompressUtils.h"

#include <core/CLogger.h>

#include <iostream>

namespace ml {
namespace domain_name_entropy {

CCompressUtils::CCompressUtils(void) : m_State(E_Uninitialized) {
}

CCompressUtils::~CCompressUtils(void) {
    ::deflateEnd(&m_ZlibStrm);
}

// --
// COMPRESS INTERFACE
// --
bool CCompressUtils::compressString(bool finish, const std::string& str) {
    int level = Z_DEFAULT_COMPRESSION;

    switch (m_State) {
        case E_Compressing: {
            // fall through
            break;
        }
        case E_Uninitialized: {
            // allocate deflate state
            m_ZlibStrm.zalloc = Z_NULL;
            m_ZlibStrm.zfree = Z_NULL;
            m_ZlibStrm.opaque = Z_NULL;
            int ret = ::deflateInit(&m_ZlibStrm, level);
            if (ret != Z_OK) {
                LOG_ERROR("Error initializing Z stream: " << ::zError(ret));
                return false;
            }
            m_State = E_Compressing;
            break;
        }
        case E_Uncompressing: {
            LOG_ERROR("Can not compress uncompressed stream");
            return false;
        }
        case E_IsFinished: {
            LOG_ERROR("Can not compress finished stream");
            return false;
        }
    }

    m_ZlibStrm.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(str.data()));
    m_ZlibStrm.avail_in = static_cast<uInt>(str.size());

    static const size_t CHUNK = 16384;
    unsigned char out[CHUNK];

    int flush = Z_NO_FLUSH;
    if (finish == true) {
        flush = Z_FINISH;
    }

    do {
        m_ZlibStrm.next_out = out;
        m_ZlibStrm.avail_out = CHUNK;
        int ret = ::deflate(&m_ZlibStrm, flush); // no bad return value
        if (ret == Z_STREAM_ERROR) {
            LOG_ERROR("Error writing Z stream: " << ::zError(ret));
            return false;
        }
        size_t have = CHUNK - m_ZlibStrm.avail_out;

        m_Buffer.insert(m_Buffer.end(), &out[0], &out[have]);

    } while (m_ZlibStrm.avail_out == 0);

    if (finish == true) {
        (void)::deflateEnd(&m_ZlibStrm);
        m_State = E_IsFinished;
    }

    return true;
}

bool CCompressUtils::compressedString(bool finish, std::string& buffer) {
    if ((finish == true && m_State == E_IsFinished) ||
        (finish == false && m_State == E_Compressing)) {
        buffer.insert(0, reinterpret_cast<const char*>(&m_Buffer[0]), m_Buffer.size());
        return true;
    }

    if (this->compressString(finish, std::string()) == false) {
        return false;
    }

    buffer.insert(0, reinterpret_cast<const char*>(&m_Buffer[0]), m_Buffer.size());
    return true;
}

bool CCompressUtils::compressedStringLength(bool finish, size_t& length) {
    if ((finish == true && m_State == E_IsFinished) ||
        (finish == false && m_State == E_Compressing)) {
        length = m_Buffer.size();
        return true;
    }

    if (this->compressString(finish, std::string()) == false) {
        return false;
    }

    length = m_Buffer.size();
    return true;
}

// --
// UNCOMPRESS INTERFACE
// --
/*
bool CCompressUtils::uncompressString(const std::string &buffer)
{
    int level = Z_DEFAULT_COMPRESSION;

    switch (m_State)
    {
        case E_Uncompressing:
        {
            // fall through
            break;
        }
        case E_Uninitialized:
        {
            // allocate inflate state
            m_ZlibStrm.zalloc = Z_NULL;
            m_ZlibStrm.zfree = Z_NULL;
            m_ZlibStrm.opaque = Z_NULL;
            int ret = ::inflateInit(&m_ZlibStrm, level);
            if (ret != Z_OK)
            {
                LOG_ERROR("Error initializing Z stream: " << ::zError(ret));
                return false;
            }
            m_State = E_Uncompressing;
            break;
        }
        case E_Compressing:
        {
            LOG_ERROR("Can not compress uncompressed stream");
            return false;
        }
        case E_IsFinished:
        {
            LOG_ERROR("Can not uncompress finished stream");
            return false;
        }
    }

    m_ZlibStrm.next_out = reinterpret_cast<Bytef *>(const_cast<char *>(str.data()));
    m_ZlibStrm.avail_out = static_cast<uInt>(str.size());

    static const size_t CHUNK = 16384;
    unsigned char out[CHUNK];

    do {
        m_ZlibStrm.next_out = out;
        m_ZlibStrm.avail_out = CHUNK;
        int ret = ::inflate(&m_ZlibStrm, Z_NO_FLUSH);    // no bad return value
        if (ret == Z_STREAM_ERROR)
        {
            LOG_ERROR("Error reading Z stream: " << ::zError(ret));
            return false;
        }
        switch (ret)
        {
            case Z_NEED_DICT:
            case Z_DATA_ERROR:
            case Z_MEM_ERROR:
                LOG_ERROR("Error reading Z stream: " << ::zError(ret));
                return false;
        }

        size_t have = CHUNK - m_ZlibStrm.avail_out;

        m_Buffer.insert(m_Buffer.end(), &out[0], &out[have]);

    } while (m_ZlibStrm.avail_out == 0);

    if (finished)
    {
    }

    if (ret != Z_STREAM_END)
    {
        LOG_ERROR("Error reading Z stream: " << ::zError(ret));
        return false;
    }

    return true;
}
*/

/*
bool CCompressUtils::uncompressedString(bool finish, std::string &buffer)
{
}

bool CCompressUtils::uncompressedStringLength(bool finish, size_t &length)
{
}
*/
}
}
