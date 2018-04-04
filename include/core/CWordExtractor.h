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
#ifndef INCLUDED_ml_core_CWordExtractor_h
#define INCLUDED_ml_core_CWordExtractor_h

#include <core/ImportExport.h>

#include <string>

namespace ml {
namespace core {

//! \brief
//! Class to extract "words" from a string.
//!
//! DESCRIPTION:\n
//! Class to extract words from a string.
//!
//! Words are taken to be sub-strings of 1 or more letters, all lower case
//! except possibly the first, preceded by a space, and followed by 0 or 1
//! punctuation characters and then a space (or the end of the string).
//! Additionally, words must be in the dictionary.  (This implies we'll
//! need to get foreign dictionaries if we get customers using the product
//! in languages other than English.)
//!
//! There is also an option to only accept words when a specified number
//! of words occur consecutively in the string.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Words are returned in a single string.  In future it might be worth
//! adding the option to return them in a vector.
//!
class CORE_EXPORT CWordExtractor {
public:
    //! Extract words from a message, and return them in a space separated
    //! string
    static void extractWordsFromMessage(const std::string& message, std::string& messageWords);

    //! Extract words from a message, and return them in a space separated
    //! string BUT only include words that occur in groups of a specified
    //! size
    static void extractWordsFromMessage(size_t minConsecutive, const std::string& message, std::string& messageWords);

private:
    //! Don't allow objects to be instantiated
    CWordExtractor();
    CWordExtractor(const CWordExtractor&);

private:
    //! The ::ispunct() function's definition of punctuation is too
    //! permissive (basically anything that's not a letter, number or
    //! space), so we have our own definition of punctuation
    static const std::string PUNCT_CHARS;
};
}
}

#endif // INCLUDED_ml_core_CWordExtractor_h
