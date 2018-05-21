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

CCompressUtils::CCompressUtils(EOperation operation, bool lengthOnly, int level)
    : m_State{E_Unused}, m_Operation{operation}, m_LengthOnly{lengthOnly} {
    ::memset(&m_ZlibStrm, 0, sizeof(z_stream));

    m_ZlibStrm.zalloc = Z_NULL;
    m_ZlibStrm.zfree = Z_NULL;

    int ret{Z_OK};
    switch (m_Operation) {
    case E_Deflate:
        ret = ::deflateInit(&m_ZlibStrm, level);
        break;
    case E_Inflate:
        ret = ::inflateInit(&m_ZlibStrm);
    }
    if (ret != Z_OK) {
        LOG_ABORT(<< "Error initialising Z stream: " << ::zError(ret));
    }
}

CCompressUtils::~CCompressUtils() {
    int ret{Z_OK};
    switch (m_Operation) {
    case E_Deflate:
        ret = ::deflateEnd(&m_ZlibStrm);
        break;
    case E_Inflate:
        ret = ::inflateEnd(&m_ZlibStrm);
        break;
    }
    if (ret != Z_OK) {
        LOG_ERROR(<< "Error ending Z stream: " << ::zError(ret));
    }
}

bool CCompressUtils::addString(const std::string& input) {
    if (m_State == E_Finished) {
        // If the last round of data processing has finished
        // and we're adding a new vector then we need to reset
        // the stream so that a new round starts from scratch.
        this->reset();
    }
    return this->processInput(false, input);
}

bool CCompressUtils::data(bool finish, TByteVec& result) {
    if (m_LengthOnly) {
        LOG_ERROR(<< "Cannot get data if asked for length-only");
        return false;
    }

    if (m_State == E_Unused) {
        LOG_ERROR(<< "Cannot get data - nothing added");
        return false;
    }

    if (finish && m_State == E_Active) {
        if (this->processInput(finish, std::string()) == false) {
            LOG_ERROR(<< "Failed to finish processing");
            return false;
        }
    }

    if (finish) {
        result = std::move(m_FullResult);
    } else {
        result = m_FullResult;
    }

    return true;
}

bool CCompressUtils::length(bool finish, size_t& length) {
    if (m_State == E_Unused) {
        LOG_ERROR(<< "Cannot get length - nothing added");
        return false;
    }

    if (finish && m_State == E_Active) {
        if (this->processInput(finish, std::string()) == false) {
            LOG_ERROR(<< "Cannot finish processing");
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
        LOG_ABORT(<< "Error reseting Z stream: " << ::zError(ret));
    }
    m_State = E_Unused;
}

bool CCompressUtils::processChunk(int flush) {
    m_ZlibStrm.next_out = m_Chunk;
    m_ZlibStrm.avail_out = CHUNK_SIZE;

    int ret{Z_OK};
    switch (m_Operation) {
    case E_Deflate:
        ret = ::deflate(&m_ZlibStrm, flush);
        break;
    case E_Inflate:
        ret = ::inflate(&m_ZlibStrm, flush);
        break;
    }
    if (ret == Z_STREAM_ERROR) {
        LOG_ERROR(<< "Error processing: " << ::zError(ret));
        return false;
    }

    size_t have(CHUNK_SIZE - m_ZlibStrm.avail_out);
    if (!m_LengthOnly) {
        m_FullResult.insert(m_FullResult.end(), &m_Chunk[0], &m_Chunk[have]);
    }
    return true;
}
}
}
