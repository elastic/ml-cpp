/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CCompressOStream.h>

#include <core/CBase64Filter.h>
#include <core/CLogger.h>

#include <boost/iostreams/filter/gzip.hpp>

#include <iostream>

namespace ml {
namespace core {

CCompressOStream::CCompressOStream(CStateCompressor::CChunkFilter& filter)
    : std::ostream(&m_StreamBuf), m_UploadThread(*this, m_StreamBuf, filter) {

    if (m_UploadThread.start() == false) {
        this->setstate(std::ios_base::failbit | std::ios_base::badbit);
    }
}

CCompressOStream::~CCompressOStream() {
    this->close();
}

void CCompressOStream::close() {
    if (m_UploadThread.isStarted()) {
        LOG_TRACE(<< "Thread has been started, so stopping it");
        if (m_UploadThread.stop() == false) {
            this->setstate(std::ios_base::failbit | std::ios_base::badbit);
        }
    }
}

CCompressOStream::CCompressThread::CCompressThread(CCompressOStream& stream,
                                                   CDualThreadStreamBuf& streamBuf,
                                                   CStateCompressor::CChunkFilter& filter)
    : m_Stream(stream),
      m_StreamBuf(streamBuf),
      m_FilterSink(filter),
      m_OutFilter()

{
    m_OutFilter.push(boost::iostreams::gzip_compressor());
    m_OutFilter.push(CBase64Encoder());
    m_OutFilter.push(boost::ref(m_FilterSink));
}

void CCompressOStream::CCompressThread::run() {
    LOG_TRACE(<< "CompressThread run");

    char buf[4096];
    std::size_t bytesDone = 0;
    bool closeMe = false;
    while (closeMe == false) {
        std::streamsize n = m_StreamBuf.sgetn(buf, 4096);
        LOG_TRACE(<< "Read from in stream: " << n);
        if (n != -1) {
            bytesDone += n;
            m_OutFilter.write(buf, n);
        }

        if (m_StreamBuf.endOfFile() && (m_StreamBuf.in_avail() == 0)) {
            closeMe = true;
        }
    }
    LOG_TRACE(<< "CompressThread complete, written: " << bytesDone << ", bytes");
    boost::iostreams::close(m_OutFilter);
}

void CCompressOStream::CCompressThread::shutdown() {
    m_StreamBuf.signalEndOfFile();
    LOG_TRACE(<< "CompressThread shutdown called");
}

} // core
} // ml
