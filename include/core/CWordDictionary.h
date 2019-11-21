/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CWordDictionary_h
#define INCLUDED_ml_core_CWordDictionary_h

#include <core/CFastMutex.h>
#include <core/CNonCopyable.h>
#include <core/ImportExport.h>

#include <boost/unordered_map.hpp>

#include <string>

namespace ml {
namespace core {

//! \brief
//! Class to check whether words are present in a dictionary.
//!
//! DESCRIPTION:\n
//! Class to check whether words are present in a dictionary.
//!
//! Can also be used to check which part of speech each word
//! represents, for example is it a noun, verb, etc.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Currently this only uses an English dictionary, albeit one with
//! US, UK and Canadian spelling variants.
//!
//! The English word list originated from SCOWL:
//! See http://sourceforge.net/projects/wordlist/files/SCOWL/
//!
//! This was then cross referenced with Moby Part-of-Speech to add
//! the part-of-speech codes:
//! See http://icon.shef.ac.uk/Moby/mpos.html
//!
//! All checks are case-insensitive.
//!
//! TODO - extend this to cope with different dictionaries for
//! different languages.
//!
//! Implemented as a Meyers singleton, but with a static pointer
//! too to avoid repeated locking in the instance() method (see
//! Modern C++ Design by Andrei Alexandrescu for details).
//!
class CORE_EXPORT CWordDictionary : private CNonCopyable {
public:
    //! Types of words.
    //! The values used are deliberately powers of two so that in the
    //! future we could potentially bitwise-or them together for words
    //! with multiple uses.
    enum EPartOfSpeech {
        E_NotInDictionary = 0,
        E_UnknownPart = 1,
        E_Noun = 2,
        E_Plural = 4,
        E_Verb = 8,
        E_Adjective = 16,
        E_Adverb = 32,
        E_Conjunction = 64,
        E_Preposition = 128,
        E_Interjection = 256,
        E_Pronoun = 512,
        E_DefiniteArticle = 1024,
        E_IndefiniteArticle = 2048
    };

public:
    //! Functor for weighting all dictionary words by a certain amount
    template<size_t DEFAULT_EXTRA_WEIGHT>
    class CWeightAll {
    public:
        size_t operator()(EPartOfSpeech partOfSpeech) {
            return (partOfSpeech == E_NotInDictionary) ? 0 : DEFAULT_EXTRA_WEIGHT;
        }
    };

    using TWeightAll2 = CWeightAll<2>;

    //! Functor for weighting one type of dictionary word by a certain
    //! amount and all dictionary words by a different amount
    template<EPartOfSpeech SPECIAL_PART1, size_t EXTRA_WEIGHT1, size_t DEFAULT_EXTRA_WEIGHT>
    class CWeightOnePart {
    public:
        size_t operator()(EPartOfSpeech partOfSpeech) {
            if (partOfSpeech == E_NotInDictionary) {
                return 0;
            }
            return (partOfSpeech == SPECIAL_PART1) ? EXTRA_WEIGHT1 : DEFAULT_EXTRA_WEIGHT;
        }
    };

    using TWeightVerbs5Other2 = CWeightOnePart<E_Verb, 5, 2>;

    //! Functor for weighting two types of dictionary word by certain
    //! amounts and all dictionary words by a different amount
    template<EPartOfSpeech SPECIAL_PART1, size_t EXTRA_WEIGHT1, EPartOfSpeech SPECIAL_PART2, size_t EXTRA_WEIGHT2, size_t DEFAULT_EXTRA_WEIGHT>
    class CWeightTwoParts {
    public:
        size_t operator()(EPartOfSpeech partOfSpeech) {
            if (partOfSpeech == E_NotInDictionary) {
                return 0;
            }
            if (partOfSpeech == SPECIAL_PART1) {
                return EXTRA_WEIGHT1;
            }
            return (partOfSpeech == SPECIAL_PART2) ? EXTRA_WEIGHT2 : DEFAULT_EXTRA_WEIGHT;
        }
    };

    // Similar templates with more arguments can be added as required...

public:
    //! Get the singleton instance
    static const CWordDictionary& instance();

    //! Check if a word is in the dictionary.  Don't call this as well as
    //! partOfSpeech().  Instead simply call partOfSpeech(), noting that
    //! it will return E_NotInDictionary in cases where this method will
    //! return false.
    bool isInDictionary(const std::string& str) const;

    //! Check what part of speech a word is primarily used for.  Note that
    //! many words can be used in different parts of speech and this method
    //! only returns what Grady Ward thought was the primary use when he
    //! created Moby.  This method returns E_NotInDictionary for words that
    //! aren't in the dictionary.
    EPartOfSpeech partOfSpeech(const std::string& str) const;

private:
    //! Constructor for a singleton is private
    CWordDictionary();
    ~CWordDictionary();

private:
    class CStrHashIgnoreCase {
    public:
        size_t operator()(const std::string& str) const;
    };

    class CStrEqualIgnoreCase {
    public:
        bool operator()(const std::string& lhs, const std::string& rhs) const;
    };

private:
    //! Name of the file to load that contains the dictionary words.
    static const char* const DICTIONARY_FILE;

    //! The constructor loads a file, and hence may take a while.  This
    //! mutex prevents the singleton object being constructed simultaneously
    //! in different threads.
    static CFastMutex ms_LoadMutex;

    //! This pointer is set after the singleton object has been constructed,
    //! and avoids the need to lock the mutex on subsequent calls of the
    //! instance() method (once the updated value of this variable has made
    //! its way into every thread).
    static volatile CWordDictionary* ms_Instance;

    //! Stores the dictionary words - using a multi-index even though
    //! there's only one index, because of its flexible key extractors.
    //! The key is the string, but hashed and compared ignoring case.
    using TStrUMap =
        boost::unordered_map<std::string, EPartOfSpeech, CStrHashIgnoreCase, CStrEqualIgnoreCase>;
    using TStrUMapCItr = TStrUMap::const_iterator;

    //! Our dictionary of words
    TStrUMap m_DictionaryWords;
};
}
}

#endif // INCLUDED_ml_core_CWordDictionary_h
