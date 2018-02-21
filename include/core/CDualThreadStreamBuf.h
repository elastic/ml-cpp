/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CDualThreadStreamBuf_h
#define INCLUDED_ml_core_CDualThreadStreamBuf_h

#include <core/CCondition.h>
#include <core/CMutex.h>
#include <core/ImportExport.h>

#include <boost/scoped_array.hpp>

#include <streambuf>


namespace ml
{
namespace core
{

//! \brief
//! A stream buffer where reads and writes are processed in different threads.
//!
//! DESCRIPTION:\n
//! Provides a mechanism for streaming data when content is being
//! generated in one thread and needs to be consumed in another.
//!
//! Can be useful when streaming includes two expensive operations,
//! such as JSON encoding and compression.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This class is optimised for the case of one thread writing into
//! it whilst another thread reads from it.  It is NOT safe for
//! more than one thread to read from it at the same time, nor for
//! more than one thread to write to it at the same time.  To
//! achieve this there is a write buffer, a read buffer and an
//! intermediate buffer.  Only access to the intermediate buffer
//! is locked, and this is swapped with the read and write buffers
//! as they underflow or overflow.
//!
//! The member functions of this class mainly overload the virtual
//! protected members of the std::streambuf class, and hence have
//! names that aren't very Ml-like.
//!
//! Some members are not overloaded, namely setbuf(), seekpos(),
//! and imbue(), which means the class can't have its buffer
//! changed, isn't seekable, and isn't locale-aware.  Also, the
//! seekoff() method is only supported as far as necessary for
//! tellp() and tellg() to work on the connected stream.  It just
//! passes input through to output without any transformation
//! and there's no way of going backwards to re-read previous data.
//! One implication is that clients of istreams using this class
//! of buffer should prefer putback() to unget() if they need to
//! return characters to the stream.
//!
//! See here for the full list of virtual protected members of
//! std::streambuf:
//! http://www.cplusplus.com/reference/streambuf/streambuf/
//!
class CORE_EXPORT CDualThreadStreamBuf : public std::streambuf
{
    public:
        //! By default, the three buffers will initially have this size.  They
        //! may potentially grow if characters are put back into them.
        static const size_t DEFAULT_BUFFER_CAPACITY;

    public:
        //! Constructor initialises buffers
        CDualThreadStreamBuf(size_t bufferCapacity = DEFAULT_BUFFER_CAPACITY);

        //! Set the end-of-file flag
        void signalEndOfFile(void);

        //! Get the end-of-file flag
        bool endOfFile(void) const;

        //! Set the fatal error flag
        void signalFatalError(void);

        //! Get the fatal error flag
        bool hasFatalError(void) const;

    protected:
        //! Get an estimate of the number of characters still to read after an
        //! underflow.  In the case of this class we return the amount of data
        //! in the intermediate buffer.
        virtual std::streamsize showmanyc(void);

        //! Switch the buffers immediately.  Effectively this flushes data
        //! through with lower latency but also less efficiently.
        virtual int sync(void);

        //! Get up to n characters from the read buffer and store them in the
        //! array pointed to by s.
        virtual std::streamsize xsgetn(char *s, std::streamsize n);

        //! Try to obtain more data for the write buffer.  This is done by
        //! swapping it with the intermediate buffer.  This may block if no data
        //! is available to read in the intermediate buffer.
        virtual int underflow(void);

        //! Put character back in the case of backup underflow.
        virtual int pbackfail(int c = traits_type::eof());

        //! Write up to n characters from the array pointed to by s into the
        //! write buffer.
        virtual std::streamsize xsputn(const char *s, std::streamsize n);

        //! Try to obtain more space in the write buffer.  This is done by
        //! swapping it with the intermediate buffer.  This may block if no data
        //! is available to read in the intermediate buffer.
        virtual int overflow(int c = traits_type::eof());

        //! In a random access stream this would seek to the specified position.
        //! This class does not support such seeking, but implements this method
        //! allowing a zero byte seek in order to allow tellg() and tellp() to
        //! work on the connected stream.
        virtual std::streampos seekoff(std::streamoff off,
                                       std::ios_base::seekdir way,
                                       std::ios_base::openmode which = std::ios_base::in | std::ios_base::out);

    private:
        //! Swap the intermediate buffer with the write buffer.  Will block if
        //! the intermediate buffer is not empty.  NB: m_IntermediateBufferMutex
        //! MUST be locked when this method is called.
        bool swapWriteBuffer(void);

        //! Swap the intermediate buffer with the read buffer.  Will block if
        //! the intermediate buffer is empty.  NB: m_IntermediateBufferMutex
        //! MUST be locked when this method is called.
        bool swapReadBuffer(void);

    private:
        //! Used to manage the two buffers.
        typedef boost::scoped_array<char> TScopedCharArray;

        //! Buffer that put functions will write to.
        TScopedCharArray m_WriteBuffer;

        //! Capacity of the write buffer.
        size_t           m_WriteBufferCapacity;

        //! Buffer that get functions will read from.
        TScopedCharArray m_ReadBuffer;

        //! Capacity of the read buffer.
        size_t           m_ReadBufferCapacity;

        //! Buffer that get functions will read from.
        TScopedCharArray m_IntermediateBuffer;

        //! Capacity of the read buffer.
        size_t           m_IntermediateBufferCapacity;

        //! End of data held in the intermediate buffer.  If this points at the
        //! beginning of the intermediate buffer, the implication is that the
        //! buffer is empty.
        char             *m_IntermediateBufferEnd;

        //! Number of bytes that have been swapped from the read buffer to the
        //! intermediate buffer over the lifetime of this object.  Enables
        //! tellg() to work on an associated istream.
        size_t           m_ReadBytesSwapped;

        //! Number of bytes that have been swapped from the write buffer to the
        //! intermediate buffer over the lifetime of this object.  Enables
        //! tellp() to work on an associated ostream.
        size_t           m_WriteBytesSwapped;

        //! A lock to protect swapping of the buffers and manage blocking when
        CMutex           m_IntermediateBufferMutex;

        //! A condition to wait on ing of the buffers and manage blocking when
        CCondition       m_IntermediateBufferCondition;

        //! Flag to indicate end-of-file.  When this is set, the reader will
        //! receive end-of-file notification once all the buffers are empty.
        //! The writer will not be allowed to add any more data.
        volatile bool    m_Eof;

        //! A call to signalFatalError() chucks away all currently buffered data
        //! and prevents future data being added.
        volatile bool    m_FatalError;
};


}
}

#endif // INCLUDED_ml_core_CDualThreadStreamBuf_h

