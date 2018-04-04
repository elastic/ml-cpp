/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CCompressUtils.h>

#include <core/CLogger.h>

#include <string.h>

namespace ml {
namespace core {

CCompressUtils::CCompressUtils(bool lengthOnly, int level) : m_State(E_Unused), m_LengthOnly(lengthOnly) {
    ::memset(&m_ZlibStrm, 0, sizeof(z_stream));

    m_ZlibStrm.zalloc = Z_NULL;
    m_ZlibStrm.zfree = Z_NULL;

    int ret(::deflateInit(&m_ZlibStrm, level));
    if (ret != Z_OK) {
        LOG_ABORT("Error initialising Z stream: " << ::zError(ret));
    }
}

CCompressUtils::~CCompressUtils() {
    int ret(::deflateEnd(&m_ZlibStrm));
    if (ret != Z_OK) {
        LOG_ERROR("Error ending Z stream: " << ::zError(ret));
    }
}

bool CCompressUtils::addString(const std::string& str) {
    if (m_State == E_Finished) {
        // If the previous compression has finished and we're adding a new
        // string then we need to reset the stream so that a new compression
        // starts from scratch
        this->reset();
    }

    return this->doCompress(false, str);
}

bool CCompressUtils::compressedData(bool finish, TByteVec& result) {
    if (m_LengthOnly) {
        LOG_ERROR("Cannot get compressed data from length-only compressor");
        return false;
    }

    if (m_State == E_Unused) {
        LOG_ERROR("Cannot get compressed data - no strings added");
        return false;
    }

    if (finish && m_State == E_Compressing) {
        if (this->doCompress(finish, std::string()) == false) {
            LOG_ERROR("Cannot finish compression");
            return false;
        }
    }

    result = m_FullResult;

    return true;
}

bool CCompressUtils::compressedLength(bool finish, size_t& length) {
    if (m_State == E_Unused) {
        LOG_ERROR("Cannot get compressed data - no strings added");
        return false;
    }

    if (finish && m_State == E_Compressing) {
        if (this->doCompress(finish, std::string()) == false) {
            LOG_ERROR("Cannot finish compression");
            return false;
        }
    }

    length = m_ZlibStrm.total_out;

    return true;
}

void CCompressUtils::reset() {
    int ret(::deflateReset(&m_ZlibStrm));
    if (ret != Z_OK) {
        // deflateReset() will only fail if one or more of the critical members
        // of the current stream struct are NULL.  If this happens then memory
        // corruption must have occurred, because there's nowhere where we set
        // these pointers to NULL after initialisation, so it's reasonable to
        // abort.
        LOG_ABORT("Error reseting Z stream: " << ::zError(ret));
    }
    m_State = E_Unused;
}

bool CCompressUtils::doCompress(bool finish, const std::string& str) {
    if (str.empty() && m_State == E_Compressing && !finish) {
        return true;
    }

    m_State = E_Compressing;

    m_ZlibStrm.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(str.data()));
    m_ZlibStrm.avail_in = static_cast<uInt>(str.size());

    static const size_t CHUNK_SIZE = 4096;
    Bytef out[CHUNK_SIZE];

    int flush(finish ? Z_FINISH : Z_NO_FLUSH);
    do {
        m_ZlibStrm.next_out = out;
        m_ZlibStrm.avail_out = CHUNK_SIZE;
        int ret(::deflate(&m_ZlibStrm, flush));
        if (ret == Z_STREAM_ERROR) {
            LOG_ERROR("Error deflating: " << ::zError(ret));
            return false;
        }

        size_t have(CHUNK_SIZE - m_ZlibStrm.avail_out);
        if (!m_LengthOnly) {
            m_FullResult.insert(m_FullResult.end(), &out[0], &out[have]);
        }
    } while (m_ZlibStrm.avail_out == 0);

    m_State = finish ? E_Finished : E_Compressing;

    return true;
}
}
}
