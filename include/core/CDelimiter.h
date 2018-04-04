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
#ifndef INCLUDED_ml_core_CDelimiter_h
#define INCLUDED_ml_core_CDelimiter_h

#include <core/CRegex.h>
#include <core/CStringUtils.h>
#include <core/ImportExport.h>

#include <iosfwd>
#include <string>

namespace ml {
namespace core {

//! \brief
//! Encapsulates a delimiter
//!
//! DESCRIPTION:\n
//! Class to delimit strings.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Delimiters are regular expressions.  This is useful in at least the
//! following circumstances:
//! 1) A delimiter can match either carriage return newline or just
//!    newline, so that it works with both Windows and Unix text files.
//! 2) A delimiter can match any number of spaces in a space delimited
//!    text file.
//!
class CORE_EXPORT CDelimiter {
public:
    //! Delimiter used by default constructor
    static const std::string DEFAULT_DELIMITER;

public:
    //! Constructors.
    CDelimiter();
    CDelimiter(const std::string& delimiter);
    CDelimiter(const std::string& delimiter, const std::string& followingRegex, bool orTime = true);

    //! Operators
    bool operator==(const CDelimiter& rhs) const;
    bool operator!=(const CDelimiter& rhs) const;

    //! Check whether the text that followed the primary delimiter was
    //! acceptable
    bool isFollowingTextAcceptable(size_t searchPos, const std::string& str, bool timePassed) const;

    //! Is the delimiter valid?
    bool valid() const;

    //! Accessor for primary delimiter
    std::string delimiter() const;

    //! Tokenise a string
    void tokenise(const std::string& str, CStringUtils::TStrVec& tokens, std::string& remainder) const;

    //! Tokenise a string, stating whether time has passed since the last
    //! attempt
    void tokenise(const std::string& str, bool timePassed, CStringUtils::TStrVec& tokens, std::string& remainder) const;

    //! Tokenise a string, also retrieving an example of the literal
    //! delimiter that was found
    void tokenise(const std::string& str, CStringUtils::TStrVec& tokens, std::string& exampleDelimiter, std::string& remainder) const;

    //! Tokenise a string, also retrieving an example of the literal
    //! delimiter that was found, stating whether time has passed since the
    //! last attempt
    void tokenise(const std::string& str,
                  bool timePassed,
                  CStringUtils::TStrVec& tokens,
                  std::string& exampleDelimiter,
                  std::string& remainder) const;

    //! Set the quote character
    void quote(char quote, char escape = '\\');

    //! Get the quote character
    char quote() const;

private:
    //! Get the position of the next unescaped quote within a string
    size_t getNextQuote(const std::string& str, size_t startPos) const;

private:
    //! The primary delimiter
    CRegex m_Delimiter;
    bool m_Valid;

    //! Only treat the primary delimiter as a delimiter if it's followed by
    //! this regular expression.
    CRegex m_FollowingRegex;
    bool m_HaveFollowingRegex;

    //! After some time has passed, should we waive the following regex?
    bool m_WaiveFollowingRegexAfterTime;

    //! The quote character (or '\0' if there isn't one).
    //! The main delimiter will be ignored if it's inside quotes.
    char m_Quote;

    //! The character used to escape the quote character ('\0' if none).
    char m_Escape;

    friend CORE_EXPORT std::ostream& operator<<(std::ostream& strm, const CDelimiter& delimiter);
};

//! Useful for debugging and CPPUNIT_ASSERT_EQUALS
CORE_EXPORT std::ostream& operator<<(std::ostream& strm, const CDelimiter& delimiter);
}
}

#endif // INCLUDED_ml_core_CDelimiter_h
