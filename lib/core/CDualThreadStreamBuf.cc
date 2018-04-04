/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CDualThreadStreamBuf.h>

#include <core/CLogger.h>
#include <core/CScopedLock.h>

#include <algorithm>

#include <string.h>


namespace ml
{
namespace core
{


// Initialise statics
const size_t CDualThreadStreamBuf::DEFAULT_BUFFER_CAPACITY(65536);


CDualThreadStreamBuf::CDualThreadStreamBuf(size_t bufferCapacity)
    : m_WriteBuffer(new char[bufferCapacity]),
      m_WriteBufferCapacity(bufferCapacity),
      m_ReadBuffer(new char[bufferCapacity]),
      m_ReadBufferCapacity(bufferCapacity),
      m_IntermediateBuffer(new char[bufferCapacity]),
      m_IntermediateBufferCapacity(bufferCapacity),
      m_IntermediateBufferEnd(m_IntermediateBuffer.get()),
      m_ReadBytesSwapped(0),
      m_WriteBytesSwapped(0),
      m_IntermediateBufferCondition(m_IntermediateBufferMutex),
      m_Eof(false),
      m_FatalError(false)
{
    // Initialise write buffer pointers to indicate an empty buffer
    char *begin(m_WriteBuffer.get());
    char *end(begin + m_WriteBufferCapacity);
    this->setp(begin, end);

    // Initialise read buffer pointers to indicate a buffer that has underflowed
    begin = m_ReadBuffer.get();
    end = begin + m_ReadBufferCapacity;
    this->setg(begin, end, end);
}

void CDualThreadStreamBuf::signalEndOfFile()
{
    CScopedLock lock(m_IntermediateBufferMutex);

    if (m_Eof)
    {
        return;
    }

    if (m_FatalError)
    {
        // If there's been a fatal error we don't care about losing data, so
        // just set the end-of-file flag and return
        m_Eof = true;

        return;
    }

    if (this->pptr() > this->pbase())
    {
        // Swapping the write buffer should wake up the reader thread
        if (this->swapWriteBuffer() == false)
        {
            LOG_ERROR("Failed to swap write buffer on setting end-of-file");
        }
    }
    else
    {
        // We don't need to swap the write buffer, but we do need to wake up
        // the reader thread
        m_IntermediateBufferCondition.signal();
    }

    // Assuming there hasn't been a fatal error, it's important that the
    // end-of-file flag isn't set until the write buffer has been swapped,
    // because otherwise the reader can act on it while the swapWriteBuffer()
    // method is waiting on m_IntermediateBufferCondition
    m_Eof = true;
}

bool CDualThreadStreamBuf::endOfFile() const
{
    return m_Eof;
}

void CDualThreadStreamBuf::signalFatalError()
{
    CScopedLock lock(m_IntermediateBufferMutex);

    // Chuck away the current read buffer
    char *begin(m_ReadBuffer.get());
    this->setg(begin, begin, begin);

    // Set a flag to indicate that future reads and writes should fail
    m_FatalError = true;

    m_IntermediateBufferCondition.signal();
}

bool CDualThreadStreamBuf::hasFatalError() const
{
    return m_FatalError;
}

std::streamsize CDualThreadStreamBuf::showmanyc()
{
    // Note that, unlike a file, we have no way of finding out what the total
    // amount of unread data is

    // Unread contents of read buffer
    std::streamsize ret(this->egptr() - this->gptr());

    CScopedLock lock(m_IntermediateBufferMutex);

    if (!m_FatalError)
    {
        // Add on unread contents of intermediate buffer
        ret += (m_IntermediateBufferEnd - m_IntermediateBuffer.get());
    }

    return ret;
}

int CDualThreadStreamBuf::sync()
{
    CScopedLock lock(m_IntermediateBufferMutex);

    if (m_FatalError)
    {
        return -1;
    }

    // If there is no data in the write buffer then sync is a no-op
    if (this->pptr() > this->pbase())
    {
        // Swapping the write buffer should wake up the reader thread
        if (this->swapWriteBuffer() == false)
        {
            LOG_ERROR("Failed to swap write buffer on sync");
            return -1;
        }
    }

    return 0;
}

std::streamsize CDualThreadStreamBuf::xsgetn(char *s, std::streamsize n)
{
    // Not locked; expected to be called only in the reader thread (see Doxygen
    // comments)

    std::streamsize ret(0);
    if (m_FatalError)
    {
        return ret;
    }

    while (ret < n)
    {
        std::streamsize bufLen(this->egptr() - this->gptr());
        if (bufLen > 0)
        {
            std::streamsize copyLen(std::min(bufLen, n - ret));
            ::memcpy(s, this->gptr(), static_cast<size_t>(copyLen));
            s += copyLen;
            ret += copyLen;
            this->gbump(static_cast<int>(copyLen));
        }
        else
        {
            // uflow() will call underflow(), so may block, but the buffers are
            // hopefully big enough that this should be rare
            int c(this->uflow());
            if (c == traits_type::eof())
            {
                break;
            }
            *s = char(c);
            ++s;
            ++ret;
        }
    }

    return ret;
}

int CDualThreadStreamBuf::underflow()
{
    CScopedLock lock(m_IntermediateBufferMutex);

    if (m_FatalError || this->swapReadBuffer() == false)
    {
        return traits_type::eof();
    }

    return int(m_ReadBuffer[0]);
}

int CDualThreadStreamBuf::pbackfail(int c)
{
    if (c == traits_type::eof())
    {
        // The standard says that pbackfail() may be called with an argument of
        // EOF to indicate that the current character at the ungotten position
        // should be retained.  Because this class does not support seeking, we
        // can't support this requirement either.  This effectively means we
        // don't reliably support sungetc().
        LOG_ERROR("pbackfail() not implemented for argument EOF");
        return c;
    }

    // The character being put back does not match the one at the putback
    // position.  Therefore we need to increase the size of the array to make
    // space for it.  THIS IS VERY EXPENSIVE and you wouldn't want to be doing
    // it often.
    ++m_ReadBufferCapacity;
    TScopedCharArray newReadBuffer(new char[m_ReadBufferCapacity]);

    std::streamsize countBeforeCurrent(this->gptr() - this->eback());
    std::streamsize countAfterCurrent(this->egptr() - this->gptr());

    char *newBegin(newReadBuffer.get());
    char *newCurrent(newBegin + countBeforeCurrent);
    char *newEnd(newCurrent + 1 + countAfterCurrent);

    if (countBeforeCurrent > 0)
    {
        ::memcpy(newBegin,
                 this->eback(),
                 static_cast<size_t>(countBeforeCurrent));
    }
    *newCurrent = char(c);
    if (countAfterCurrent > 0)
    {
        ::memcpy(newCurrent + 1,
                 this->gptr(),
                 static_cast<size_t>(countAfterCurrent));
    }

    m_ReadBuffer.swap(newReadBuffer);
    this->setg(newBegin, newCurrent, newEnd);

    return c;
}

std::streamsize CDualThreadStreamBuf::xsputn(const char *s, std::streamsize n)
{
    // Not locked; expected to be called only in the writer thread (see Doxygen
    // comments)

    std::streamsize ret(0);

    if (m_Eof)
    {
        LOG_ERROR("Inconsistency - trying to add data to stream buffer after end-of-file");
        return ret;
    }

    if (m_FatalError)
    {
        return ret;
    }

    while (ret < n)
    {
        std::streamsize bufAvail(this->epptr() - this->pptr());
        if (bufAvail > 0)
        {
            std::streamsize copyLen(std::min(bufAvail, n - ret));
            ::memcpy(this->pptr(), s, static_cast<size_t>(copyLen));
            s += copyLen;
            ret += copyLen;
            this->pbump(static_cast<int>(copyLen));
        }
        else
        {
            // overflow() may block, but the buffers are hopefully big enough
            // that this should be rare
            int c(this->overflow(int(*s)));
            if (c == traits_type::eof())
            {
                break;
            }
            ++s;
            ++ret;
        }
    }

    return ret;
}

int CDualThreadStreamBuf::overflow(int c)
{
    int ret(traits_type::eof());

    CScopedLock lock(m_IntermediateBufferMutex);

    if (m_Eof || m_FatalError || this->swapWriteBuffer() == false)
    {
        return ret;
    }

    if (c == ret)
    {
        m_Eof = true;
        // If the argument indicated EOF, we don't put it in the new buffer
        ret = traits_type::not_eof(c);
    }
    else
    {
        m_WriteBuffer[0] = char(c);
        this->pbump(1);
        ret = c;
    }

    return ret;
}

std::streampos CDualThreadStreamBuf::seekoff(std::streamoff off,
                                       std::ios_base::seekdir way,
                                       std::ios_base::openmode which)
{
    std::streampos pos(static_cast<std::streampos>(-1));

    if (off != 0)
    {
        LOG_ERROR("Seeking not supported on stream buffer");
        return pos;
    }

    if (way != std::ios_base::cur)
    {
        LOG_ERROR("Seeking from beginning or end not supported on stream buffer");
        return pos;
    }

    if (which == std::ios_base::in)
    {
        CScopedLock lock(m_IntermediateBufferMutex);
        pos = static_cast<std::streampos>(m_ReadBytesSwapped);
        pos -= (this->egptr() - this->gptr());
    }
    else if (which == std::ios_base::out)
    {
        CScopedLock lock(m_IntermediateBufferMutex);
        pos = static_cast<std::streampos>(m_WriteBytesSwapped);
        pos += (this->pptr() - this->pbase());
    }
    else
    {
        LOG_ERROR("Unexpected mode for seek on stream buffer: " << which);
    }

    return pos;
}

// NB: m_IntermediateBufferMutex MUST be locked when this method is called
bool CDualThreadStreamBuf::swapWriteBuffer()
{
    // Wait until the intermediate buffer is empty
    while (m_IntermediateBufferEnd > m_IntermediateBuffer.get())
    {
        m_IntermediateBufferCondition.wait();
        if (m_FatalError)
        {
            return false;
        }
    }

    m_WriteBytesSwapped += (this->pptr() - this->pbase());

    // Intermediate buffer now suitable for reading from
    m_IntermediateBufferEnd = this->pptr();
    m_WriteBuffer.swap(m_IntermediateBuffer);
    std::swap(m_WriteBufferCapacity, m_IntermediateBufferCapacity);
    char *begin(m_WriteBuffer.get());
    char *end(begin + m_WriteBufferCapacity);
    this->setp(begin, end);

    // Signal any waiting reader
    m_IntermediateBufferCondition.signal();

    return true;
}

// NB: m_IntermediateBufferMutex MUST be locked when this method is called
bool CDualThreadStreamBuf::swapReadBuffer()
{
    // Wait until the intermediate buffer contains data
    while (!m_Eof &&
           m_IntermediateBufferEnd == m_IntermediateBuffer.get())
    {
        m_IntermediateBufferCondition.wait();
        if (m_FatalError)
        {
            return false;
        }
    }

    char *begin(m_IntermediateBuffer.get());
    char *end(m_IntermediateBufferEnd);
    if (begin >= end)
    {
        if (!m_Eof)
        {
            LOG_ERROR("Inconsistency - intermediate buffer empty after wait "
                      "when not at end-of-file: begin = " <<
                      static_cast<void *>(begin) << " end = " <<
                      static_cast<void *>(end));
        }
        return false;
    }

    m_ReadBytesSwapped += (end - begin);

    // Intermediate buffer now suitable for reading from
    m_ReadBuffer.swap(m_IntermediateBuffer);
    std::swap(m_ReadBufferCapacity, m_IntermediateBufferCapacity);
    m_IntermediateBufferEnd = m_IntermediateBuffer.get();
    this->setg(begin, begin, end);

    // Signal any waiting writer
    m_IntermediateBufferCondition.signal();

    return true;
}


}
}

