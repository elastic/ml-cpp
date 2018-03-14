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
#ifndef INCLUDED_ml_api_CLengthEncodedInputParser_h
#define INCLUDED_ml_api_CLengthEncodedInputParser_h


#include <api/CInputParser.h>
#include <api/ImportExport.h>

#include <boost/scoped_array.hpp>

#include <iosfwd>
#include <string>

#include <stdint.h>


namespace ml {
namespace api {

//! \brief
//! Parse the length encoded input data
//!
//! DESCRIPTION:\n
//! Parse an input format structured as follows.
//!
//! At a very high level the format is similar to CSV in that it's:
//! Field names record
//! Data record
//! Data record
//! .
//! .
//! .
//! Data record
//!
//! There must be the same number of fields in each data record as
//! in the field names record, and they must be in the same order
//! (like CSV).
//!
//! Each record has the following format:
//! Number of fields (32 bit integer in network byte order)
//! Data length (32 bit integer in network byte order)
//! Data text (UTF-8 character sequence; not zero terminated)
//! Data length (32 bit integer in network byte order)
//! Data text (UTF-8 character sequence; not zero terminated)
//! .
//! .
//! .
//! Value length (32 bit integer in network byte order)
//! Value text (UTF-8 character sequence; not zero terminated)
//!
//! In other words, each record consists of a number of fields
//! followed by a sequence of length/value pairs corresponding to
//! each of the fields.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The data format is designed to be a simple drop-in replacement
//! for CSV, but order-of-magnitude more efficient to decode.  The
//! reason it's so much more efficient to decode is that there is
//! no need to examine every character to determine whether it's a
//! comma, quote or newline.  Instead the length is read and then
//! data transfer boils down to a memcpy(), which is extremely fast.
//!
//! It is assumed that there will never be more than 2 billion fields
//! in a record, and never more than 2 billion bytes in a field name
//! or value.  This means it is not necessary to worry about whether
//! the 32 bit integers are signed or unsigned: assuming
//! twos-complement format signed and unsigned representations will
//! be the same for values less than 2 billion.  This makes
//! interfacing with Java (which doesn't have built-in unsigned
//! types) easier.
//!
class API_EXPORT CLengthEncodedInputParser : public CInputParser {
    public:
        //! Construct with an input stream to be parsed.  Once a stream is
        //! passed to this constructor, no other object should read from it.
        //! For example, if std::cin is passed, no other object should read from
        //! std::cin, otherwise unpredictable and incorrect results will be
        //! generated.
        //!
        //! The stream passed should have been created in binary mode, i.e. with
        //! the std::ios::binary flag as part of its constructor's openmode
        //! argument.  Otherwise, on Windows, a CTRL+Z in the input stream will
        //! be considered as end-of-file.  The exception is when std::cin is the
        //! input stream, in which case this constructor will set the standard
        //! input of the whole process to binary mode (because it's not possible
        //! to do this for an already opened stream and std::cin will be open
        //! before main() runs).
        CLengthEncodedInputParser(std::istream &strmIn);

        //! Read records from the stream. The supplied reader function is called
        //! once per record.  If the supplied reader function returns false,
        //! reading will stop.  This method keeps reading until it reaches the
        //! end of the stream or an error occurs.  If it successfully reaches
        //! the end of the stream it returns true, otherwise it returns false.
        virtual bool readStream(const TReaderFunc &readerFunc);

    private:
        //! Attempt to parse a single length encoded record from the stream into
        //! the strings in the vector provided.  The vector is a template
        //! argument so that it may be a vector of boost::reference_wrappers
        //! of std::strings instead of std::strings.  The first template
        //! argument indicates whether the vector must have the correct size
        //! when the function is called or whether the function is allowed to
        //! resize it.
        template <bool RESIZE_ALLOWED, typename STR_VEC>
        bool parseRecordFromStream(STR_VEC &results);

        //! Parse a 32 bit unsigned integer from the input stream.
        bool parseUInt32FromStream(uint32_t &num);

        //! Parse a string of given length from the input stream.
        bool parseStringFromStream(size_t length,
                                   std::string &str);

        //! Refill the working buffer from the stream
        size_t refillBuffer(void);

    private:
        //! Allocate this much memory for the working buffer
        static const size_t WORK_BUFFER_SIZE;

        //! Reference to the stream we're going to read from
        std::istream        &m_StrmIn;

        typedef boost::scoped_array<char> TScopedCharArray;

        //! The working buffer is also held as a member to avoid constantly
        //! reallocating it.  It is a raw character array rather than a string
        //! to facilitate the use of std::istream::read() to obtain input.
        //! std::istream::read() uses memcpy() to shuffle data around on all
        //! platforms, and is hence an order of magnitude faster than reading
        //! small chunks of data from the stream repeatedly.  The array of
        //! characters is NOT zero terminated, which is something to be aware of
        //! when accessing it.
        TScopedCharArray m_WorkBuffer;
        const char       *m_WorkBufferPtr;
        const char       *m_WorkBufferEnd;
        bool             m_NoMoreRecords;
};


}
}

#endif // INCLUDED_ml_api_CLengthEncodedInputParser_h

