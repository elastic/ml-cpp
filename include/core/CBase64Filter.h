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
#ifndef INCLUDED_ml_core_CBase64Filter_h
#define INCLUDED_ml_core_CBase64Filter_h

#include <core/CLogger.h>
#include <core/ImportExport.h>

#include <boost/iostreams/concepts.hpp>
#include <boost/iostreams/operations.hpp>

#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/binary_from_base64.hpp>
#include <boost/archive/iterators/transform_width.hpp>

#include <boost/circular_buffer.hpp>
#include <boost/circular_buffer/base.hpp>

#include <string>

#include <stdint.h>

namespace ml {

namespace core {
//! \brief
//! Convert a stream of bytes into Base64.
//!
//! DESCRIPTION:\n
//! Class to convert bytes into Base64 using the boost iostreams
//! filter interface.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Uses a boost circular_buffer to cache chunks of up to 4096
//! bytes from the incoming stream, as Base64 can only be
//! encoded in multiples of 3.
//! This class is templated on the stream output type, which must
//! be compatible with boost::iostreams::write.
//! Stream data is copied from the incoming stream, processed
//! locally, then copied into the output stream; this could be
//! improved, at the expense of simplicity, if it proves to be
//! overly slow.
//! Based on: https://github.com/UeSyu/base64_filter
//!
class CORE_EXPORT CBase64Encoder {
public:
    typedef boost::circular_buffer<uint8_t> TUInt8Buf;
    typedef TUInt8Buf::iterator TUInt8BufItr;
    typedef TUInt8Buf::const_iterator TUInt8BufCItr;

    typedef char char_type;

    //! Tell boost::iostreams what this filter is capable of
    struct category : public boost::iostreams::output,
                      public boost::iostreams::filter_tag,
                      public boost::iostreams::multichar_tag,
                      public boost::iostreams::closable_tag {};

public:
    //! Constructor
    CBase64Encoder();

    //! Destructor
    virtual ~CBase64Encoder();

    //! Interface method for handling stream data: n bytes are available from s,
    //! and output is written to snk.
    //! Note that up to n bytes should be read if possible, but we don't report
    //! here how many bytes were actually written to the stream, only how many
    //! we actually consumed from s.
    template<typename SINK>
    std::streamsize write(SINK& snk, const char_type* s, std::streamsize n) {
        // copy into the buffer while there is data to read and space in the buffer
        std::streamsize done = 0;
        while (done < n) {
            std::streamsize toCopy =
                std::min(std::streamsize(n - done), std::streamsize(m_Buffer.capacity() - m_Buffer.size()));
            m_Buffer.insert(m_Buffer.end(), s + done, s + done + toCopy);
            done += toCopy;
            this->Encode(snk, false);
        }
        LOG_TRACE("Base64 write " << n);
        return n;
    }

    //! Interface method for terminating this filter class - flush
    //! any remaining bytes and pad the output if necessary.
    template<typename SINK>
    void close(SINK& snk) {
        this->Encode(snk, true);
    }

private:
    //! Do the actual work of encoding the data - take a chunck of buffered data and write
    //! the converted output into the stream snk
    template<typename SINK>
    void Encode(SINK& snk, bool isFinal) {
        typedef boost::archive::iterators::transform_width<TUInt8BufCItr, 6, 8> TUInt8BufCItrTransformItr;
        typedef boost::archive::iterators::base64_from_binary<TUInt8BufCItrTransformItr> TBase64Text;

        TUInt8BufItr endItr = m_Buffer.end();
        // Base64 turns 3 bytes into 4 characters - unless this is the final part
        // of the string, we don't encode non-multiples of 3
        if (isFinal == false) {
            for (std::size_t i = (m_Buffer.size() % 3); i != 0; i--) {
                --endItr;
            }
        }

        // Do the conversion
        std::string e(TBase64Text(m_Buffer.begin()), TBase64Text(endItr));

        // Remove the encoded bytes from the buffer
        m_Buffer.erase(m_Buffer.begin(), endItr);

        // Pad the final string if necessary
        if (isFinal && !e.empty()) {
            std::size_t paddingCount = 4 - e.length() % 4;
            for (std::size_t i = 0; i < paddingCount; i++) {
                e += '=';
            }
        }
        LOG_TRACE("Encoded: " << e);
        boost::iostreams::write(snk, e.c_str(), e.length());
    }

private:
    //! Buffer the incoming stream data so that we can handle non-multiples of 3
    TUInt8Buf m_Buffer;
};

//! \brief
//! Convert a stream of Base64 characters to bytes
//!
//! DESCRIPTION:\n
//! Class to convert Base64 characters into bytes using the boost iostreams
//! filter interface.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Uses a boost circular_buffer to cache chunks of up to 4096
//! bytes from the incoming stream, and another buffer to cache
//! the converted bytes.
//! This class is templated on the stream output type, which must
//! be compatible with boost::iostreams::write.
//! Stream data is copied from the incoming stream, processed
//! locally, then copied into the output stream; this could be
//! improved, at the expense of simplicity, if it proves to be
//! overly slow.
//! Based on: https://github.com/UeSyu/base64_filter
//!
class CORE_EXPORT CBase64Decoder {
public:
    typedef boost::circular_buffer<uint8_t> TUInt8Buf;
    typedef TUInt8Buf::iterator TUInt8BufItr;
    typedef TUInt8Buf::const_iterator TUInt8BufCItr;
    typedef TUInt8Buf::const_reverse_iterator TUInt8BufCRItr;
    typedef char char_type;

    //! Tell boost::iostreams what this filter is capable of
    struct category : public boost::iostreams::input,
                      public boost::iostreams::filter_tag,
                      public boost::iostreams::multichar_tag,
                      public boost::iostreams::closable_tag {};

public:
    //! Constructor
    CBase64Decoder();

    //! Destructor
    virtual ~CBase64Decoder();

    //! Interface method: read as many bytes as we need from src, and
    //! put up to n output bytes into s
    //! The input bytes are buffered, decoded, and the decoded bytes
    //! written to s. Note that we return the number of bytes written to
    //! s, not the number of input bytes copied from src
    template<typename SOURCE>
    std::streamsize read(SOURCE& src, char_type* s, std::streamsize n) {
        // copy into the buffer while there is data to read and space in the buffer
        std::streamsize done = 0;
        char buf[4096];
        while (done < n) {
            std::streamsize toCopy = std::min(std::streamsize(m_BufferOut.size()), std::streamsize(n - done));
            LOG_TRACE("Trying to copy " << toCopy << " bytes into stream, max " << n << ", available "
                                        << m_BufferOut.size());
            for (std::streamsize i = 0; i < toCopy; i++) {
                s[done++] = m_BufferOut.front();
                m_BufferOut.pop_front();
            }
            LOG_TRACE("Eos: " << m_Eos << ", In: " << m_BufferIn.empty() << ", Out: " << m_BufferOut.empty());
            if (done == n) {
                break;
            }
            if ((done > 0) && m_BufferIn.empty() && m_BufferOut.empty() && m_Eos) {
                LOG_TRACE("Base64 READ " << done << ", from n " << n << ", left " << m_BufferOut.size());
                return done;
            }

            // grab some data if we need it
            if ((m_BufferIn.size() < 4) && (m_Eos == false)) {
                std::streamsize readBytes = boost::iostreams::read(src, buf, 4096);
                LOG_TRACE("Read " << readBytes << " from input stream");
                if (readBytes == -1) {
                    LOG_TRACE("Got EOS from underlying store");
                    m_Eos = true;
                } else {
                    for (std::streamsize i = 0; i < readBytes; i++) {
                        // Only copy Base64 characters - JSON punctuation is ignored
                        // The dechunker parses JSON and should give us only base64 strings,
                        // but we don't want to try and decode anything which might cause
                        // the decoder to choke
                        switch (buf[i]) {
                        case ']':
                        case '[':
                        case ',':
                        case '"':
                        case '{':
                        case '}':
                        case '\\':
                        case ' ':
                        case ':':
                            break;

                        default:
                            m_BufferIn.push_back(static_cast<uint8_t>(buf[i]));
                            break;
                        }
                    }
                }
            }
            this->Decode(m_Eos);
            if (m_Eos && m_BufferOut.empty() && m_BufferIn.empty() && (done == 0)) {
                LOG_TRACE("Returning -1 from read");
                return -1;
            }
        }
        LOG_TRACE("Base64 READ " << done << ", from n " << n << ", left " << m_BufferOut.size());
        return done;
    }

    //! Interface method - unused
    template<typename SOURCE>
    void close(SOURCE& /*src*/) {}

private:
    //! Perform the conversion from Base64 to raw bytes
    void Decode(bool isFinal) {
        // Base64 turns 4 characters into 3 bytes
        typedef boost::archive::iterators::binary_from_base64<TUInt8BufCItr> TUInt8BufCItrBinaryBase64Itr;
        typedef boost::archive::iterators::transform_width<TUInt8BufCItrBinaryBase64Itr, 8, 6, uint8_t> TBase64Binary;

        std::size_t inBytes = m_BufferIn.size();
        if (inBytes == 0) {
            return;
        }

        TUInt8BufItr endItr = m_BufferIn.end();
        std::size_t paddingBytes = 0;
        // Only try and decode multiples of 4 characters, unless this is the last
        // data in the stream
        if (isFinal == false) {
            if (inBytes < 4) {
                return;
            }

            for (std::size_t i = 0; i < inBytes % 4; i++) {
                LOG_TRACE("Ignoring end bytes of " << inBytes);
                --endItr;
            }
        } else {
            // We can only work with 4 or more bytes, so with fewer there is something
            // wrong, and there can't be a sensible outcome
            if (inBytes < 4) {
                LOG_ERROR("Invalid size of stream for decoding: " << inBytes);
                m_BufferIn.clear();
                return;
            }
        }

        // Check for padding characters
        {
            TUInt8BufCRItr i = m_BufferIn.rbegin();
            while ((i != m_BufferIn.rend()) && (*i == '=')) {
                ++i;
                paddingBytes++;
            }
        }
        LOG_TRACE("About to decode: " << std::string(m_BufferIn.begin(), endItr));

        m_BufferOut.insert(m_BufferOut.end(), TBase64Binary(m_BufferIn.begin()), TBase64Binary(endItr));

        // Remove padding bytes off the back of the stream
        m_BufferOut.erase_end(paddingBytes);

        // Remove the encoded bytes from the buffer
        m_BufferIn.erase(m_BufferIn.begin(), endItr);
    }

private:
    //! The input buffer
    TUInt8Buf m_BufferIn;

    //! The output buffer
    TUInt8Buf m_BufferOut;

    //! Have we read all the available data from the downstream stream
    bool m_Eos;
};

} // core

} // ml

#endif // INCLUDED_ml_core_CBase64Filter_h
