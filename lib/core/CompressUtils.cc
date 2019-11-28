/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CompressUtils.h>

#include <core/CLogger.h>
#include <core/CMemory.h>

#include <string.h>

namespace ml {
namespace core {

CCompressUtil::CCompressUtil(bool lengthOnly)
    : m_State{E_Unused}, m_LengthOnly{lengthOnly} {
    ::memset(&m_ZlibStrm, 0, sizeof(z_stream));
    m_ZlibStrm.zalloc = Z_NULL;
    m_ZlibStrm.zfree = Z_NULL;
}

bool CCompressUtil::addString(const std::string& input) {
    if (m_State == E_Finished) {
        // If the last round of data processing has finished
        // and we're adding a new vector then we need to reset
        // the stream so that a new round starts from scratch.
        this->reset();
    }
    return this->processInput(false, input);
}

bool CCompressUtil::data(bool finish, TByteVec& result) {
    if (this->prepareToReturnData(finish) == false) {
        return false;
    }
    result = m_FullResult;
    return true;
}

bool CCompressUtil::finishAndTakeData(TByteVec& result) {
    if (this->prepareToReturnData(true) == false) {
        return false;
    }
    result = std::move(m_FullResult);
    return true;
}

bool CCompressUtil::length(bool finish, std::size_t& length) {
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

void CCompressUtil::reset() {
    int ret{this->resetStream()};
    if (ret != Z_OK) {
        // resetStream() will only fail if one or more of the critical
        // members of the current z_stream struct are NULL.  If this
        // happens then memory corruption must have occurred, because
        // there's nowhere where we set these pointers to NULL after
        // initialisation, so it's reasonable to abort.
        LOG_ABORT(<< "Error reseting Z stream: " << ::zError(ret));
    }
    m_State = E_Unused;
}

z_stream& CCompressUtil::stream() {
    return m_ZlibStrm;
}

bool CCompressUtil::processChunk(int flush) {
    m_ZlibStrm.next_out = m_Chunk;
    m_ZlibStrm.avail_out = CHUNK_SIZE;

    int ret{this->streamProcessChunk(flush)};
    if (ret == Z_STREAM_ERROR) {
        LOG_ERROR(<< "Error processing: " << ::zError(ret));
        return false;
    }

    std::size_t have{CHUNK_SIZE - m_ZlibStrm.avail_out};
    if (!m_LengthOnly) {
        m_FullResult.insert(m_FullResult.end(), &m_Chunk[0], &m_Chunk[have]);
    }
    return true;
}

bool CCompressUtil::prepareToReturnData(bool finish) {
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
    return true;
}

void CCompressUtil::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CCompressUtil");
    core::CMemoryDebug::dynamicSize("m_FullResult", m_FullResult, mem);
    if (m_ZlibStrm.state != nullptr) {
        mem->addItem("m_ZlibStrm", 5952); // See below for where 5952 came from
    }
}

std::size_t CCompressUtil::memoryUsage() const {
    std::size_t mem = 0;
    mem += core::CMemory::dynamicSize(m_FullResult);
    if (m_ZlibStrm.state != nullptr) {
        // The value of 5952 was found using this program compiled in the zlib
        // 1.2.11 source directory:
        //
        // #include "deflate.h"
        // #include <iostream>
        // int main(int, char**) {
        //     std::cout << sizeof(internal_state) << '\n';
        //     return 0;
        // }
        //
        // There is no way to find this number dynamically in the ML code, as
        // it is a hidden implementation detail protected by an opaque pointer.
        // The size may vary between zlib versions, but probably not by enough
        // to be worth worrying about.
        mem += 5952;
    }
    return mem;
}

CDeflator::CDeflator(bool lengthOnly, int level) : CCompressUtil{lengthOnly} {
    int ret{::deflateInit(&this->stream(), level)};
    if (ret != Z_OK) {
        LOG_ABORT(<< "Error initialising Z stream: " << ::zError(ret));
    }
}

CDeflator::~CDeflator() {
    int ret{::deflateEnd(&this->stream())};
    if (ret != Z_OK) {
        LOG_ERROR(<< "Error ending Z stream: " << ::zError(ret));
    }
}

int CDeflator::streamProcessChunk(int flush) {
    return ::deflate(&this->stream(), flush);
}

int CDeflator::resetStream() {
    return ::deflateReset(&this->stream());
}

CInflator::CInflator(bool lengthOnly) : CCompressUtil{lengthOnly} {
    int ret{::inflateInit(&this->stream())};
    if (ret != Z_OK) {
        LOG_ABORT(<< "Error initialising Z stream: " << ::zError(ret));
    }
}

CInflator::~CInflator() {
    int ret{::inflateEnd(&this->stream())};
    if (ret != Z_OK) {
        LOG_ERROR(<< "Error ending Z stream: " << ::zError(ret));
    }
}

int CInflator::streamProcessChunk(int flush) {
    return ::inflate(&this->stream(), flush);
}

int CInflator::resetStream() {
    return ::inflateReset(&this->stream());
}
}
}
