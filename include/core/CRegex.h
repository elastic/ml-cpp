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
#ifndef INCLUDED_ml_core_CRegex_h
#define INCLUDED_ml_core_CRegex_h

#include <core/ImportExport.h>

#include <boost/regex.hpp>

#include <string>

namespace ml {
namespace core {

//! \brief
//! Wrapper around boost::regex
//!
//! DESCRIPTION:\n
//! Wrapper around boost::regex
//!
//! IMPLEMENTATION DECISIONS:\n
//! Uses init method to initialise so exceptions can be caught.
//!
class CORE_EXPORT CRegex {
public:
    using TStrVec = std::vector<std::string>;
    using TStrVecItr = TStrVec::iterator;
    using TStrVecCItr = TStrVec::const_iterator;

public:
    CRegex();

    bool init(const std::string&);

    //! Simple match test for a string
    bool matches(const std::string&) const;

    //! Find the position within a string at which this regex first matches
    bool search(size_t startPos, const std::string& str, size_t& position, size_t& length) const;
    bool search(size_t startPos, const std::string& str, size_t& position) const;
    bool search(const std::string& str, size_t& position, size_t& length) const;
    bool search(const std::string& str, size_t& position) const;

    //! Match a string with the regex AND
    //! tokenise a string by sub-expressions (...)
    //! This is based on the 'grouping' syntax in perl regex
    bool tokenise(const std::string&, TStrVec&) const;

    //! Split a string based on a regex
    bool split(const std::string&, TStrVec&) const;

    //! Get the pattern string (not a reference due to boost API)
    std::string str() const;

    //! How much of the regex is literal characters rather than character
    //! classes?
    size_t literalCount() const;

    //! Useful for converting a string literal into a regex that will match
    //! it
    static std::string escapeRegexSpecial(const std::string& literal);

private:
    bool m_Initialised;
    boost::regex m_Regex;
};
}
}

#endif // INCLUDED_ml_core_CRegex_h
