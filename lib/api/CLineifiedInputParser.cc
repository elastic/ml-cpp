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
#include <api/CLineifiedInputParser.h>

#include <core/CLogger.h>

#include <istream>

#include <string.h>


namespace ml {
namespace api {


// Initialise statics
const char CLineifiedInputParser::LINE_END('\n');
const size_t CLineifiedInputParser::WORK_BUFFER_SIZE(131072); // 128kB


CLineifiedInputParser::CLineifiedInputParser(std::istream &strmIn)
    : CInputParser(),
      m_StrmIn(strmIn),
      m_WorkBuffer(0),
      m_WorkBufferCapacity(0),
      m_WorkBufferPtr(0),
      m_WorkBufferEnd(0)
{}

CLineifiedInputParser::TCharPSizePr
CLineifiedInputParser::parseLine(void) {
    // For maximum performance, read the stream in large chunks that can be
    // moved around by memcpy().  Using memcpy() is an order of magnitude faster
    // than the naive approach of checking and copying one character at a time.
    // In modern versions of the GNU STL std::getline uses memchr() to search
    // for the delimiter and then memcpy() to transfer data to the target
    // std::string, but sadly this is not the case for the Microsoft and Apache
    // STLs.
    if (m_WorkBuffer.get() == 0) {
        m_WorkBuffer.reset(new char[WORK_BUFFER_SIZE]);
        m_WorkBufferCapacity = WORK_BUFFER_SIZE;
        m_WorkBufferPtr = m_WorkBuffer.get();
        m_WorkBufferEnd = m_WorkBufferPtr;
    }

    for (;;) {
        size_t avail(m_WorkBufferEnd - m_WorkBufferPtr);
        if (avail > 0) {
            char *delimPtr(reinterpret_cast<char *>(::memchr(m_WorkBufferPtr,
                                                             LINE_END,
                                                             avail)));
            if (delimPtr != 0) {
                *delimPtr = '\0';
                TCharPSizePr result(m_WorkBufferPtr, delimPtr - m_WorkBufferPtr);
                m_WorkBufferPtr = delimPtr + 1;
                return result;
            }

            if (m_WorkBufferPtr > m_WorkBuffer.get()) {
                // We didn't find a line ending, but we started part way through the
                // the buffer, so shuffle it up and refill it
                ::memmove(m_WorkBuffer.get(), m_WorkBufferPtr, avail);
            } else   {
                // We didn't find a line ending and started at the beginning of a
                // full buffer so expand it
                m_WorkBufferCapacity += WORK_BUFFER_SIZE;
                TScopedCharArray newBuffer(new char[m_WorkBufferCapacity]);
                ::memcpy(newBuffer.get(), m_WorkBufferPtr, avail);
                m_WorkBuffer.swap(newBuffer);
            }
            m_WorkBufferPtr = m_WorkBuffer.get();
            m_WorkBufferEnd = m_WorkBufferPtr + avail;
        }

        if (m_StrmIn.eof()) {
            // We have no lines in the buffered data and are already at the end
            // of the stream, so stop now
            break;
        }

        m_StrmIn.read(m_WorkBufferEnd,
                      static_cast<std::streamsize>(m_WorkBufferCapacity - avail));
        std::streamsize bytesRead(m_StrmIn.gcount());
        if (bytesRead == 0) {
            if (m_StrmIn.bad()) {
                LOG_ERROR("Input stream is bad");
            }
            // We needed to read more data and didn't get any, so stop
            break;
        }
        m_WorkBufferEnd += bytesRead;
    }

    return TCharPSizePr(static_cast<char *>(0), 0);
}

void CLineifiedInputParser::resetBuffer(void) {
    m_WorkBufferEnd = m_WorkBufferPtr;
}


}
}

