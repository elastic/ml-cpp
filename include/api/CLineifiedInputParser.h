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
#ifndef INCLUDED_ml_api_CLineifiedInputParser_h
#define INCLUDED_ml_api_CLineifiedInputParser_h

#include <api/CInputParser.h>
#include <api/ImportExport.h>

#include <boost/scoped_array.hpp>

#include <iosfwd>
#include <utility>


namespace ml
{
namespace api
{

//! \brief
//! Base class to parse lines containing nested documents from a stream.
//!
//! DESCRIPTION:\n
//! This class can be used as a base for more complex parsers where each
//! line of a stream is guaranteed to represent exactly one document.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Does not use std::getline as it's not portably fast - see
//! http://info.prelert.com/blog/stdgetline-is-the-poor-relation
//!
//! The original use case was to factor out commonality from lineified
//! JSON and XML parsers.
//!
class API_EXPORT CLineifiedInputParser : public CInputParser
{
    public:
        //! Construct with an input stream to be parsed.  Once a stream is
        //! passed to this constructor, no other object should read from it.
        //! For example, if std::cin is passed, no other object should read from
        //! std::cin, otherwise unpredictable and incorrect results will be
        //! generated.
        CLineifiedInputParser(std::istream &strmIn);

    protected:
        //! Line end character
        static const char LINE_END;

        using TCharPSizePr = std::pair<char*, size_t>;

    protected:
        //! Return a pointer to the start of the next line and its length,
        //! reading extra data from the stream if required.  The pair (NULL, 0)
        //! will be returned if no further data is available.  The newline
        //! character at the end of the line is replaced with a zero terminator
        //! byte so that the line can later be parsed in-situ by a library
        //! that expects a zero-terminated string.
        TCharPSizePr parseLine(void);

        //! Reset the work buffer to empty.  This should be called if the stream
        //! that data is being read from might have had its stream buffer
        //! changed.
        void resetBuffer(void);

    private:
        //! Allocate this much memory for the working buffer
        static const size_t WORK_BUFFER_SIZE;

        //! Reference to the stream we're going to read from
        std::istream     &m_StrmIn;

        using TScopedCharArray = boost::scoped_array<char>;

        //! The working buffer is a raw character array rather than a string to
        //! facilitate the use of std::istream::read() to obtain input rather
        //! than std::getline().  std::getline() is efficient in the GNU STL but
        //! sadly not in the Microsoft or Apache STLs, where it copies one
        //! character at a time.  std::istream::read() uses memcpy() to shuffle
        //! data around on all platforms, and is hence an order of magnitude
        //! faster.  (This is the sort of optimisation to be used ONLY after
        //! careful profiling in the rare cases where the reduction in code
        //! clarity yields a large performance benefit.)  The array of
        //! characters is NOT zero terminated, which is something to be aware of
        //! when accessing it.
        TScopedCharArray m_WorkBuffer;
        size_t           m_WorkBufferCapacity;
        char             *m_WorkBufferPtr;
        char             *m_WorkBufferEnd;
};


}
}

#endif // INCLUDED_ml_api_CLineifiedJsonInputParser_h

