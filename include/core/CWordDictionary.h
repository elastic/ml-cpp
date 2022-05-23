/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */
#ifndef INCLUDED_ml_core_CWordDictionary_h
#define INCLUDED_ml_core_CWordDictionary_h

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
class CORE_EXPORT CWordDictionary {
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
        std::size_t operator()(EPartOfSpeech partOfSpeech) {
            return (partOfSpeech == E_NotInDictionary) ? 0 : DEFAULT_EXTRA_WEIGHT;
        }

        void reset() {
            // NO-OP
        }

        std::size_t minMatchingWeight(std::size_t weight) { return weight; }

        std::size_t maxMatchingWeight(std::size_t weight) { return weight; }
    };

    using TWeightAll2 = CWeightAll<2>;

    //! Functor for weighting one type of dictionary word by a certain
    //! amount and all dictionary words by a different amount
    template<EPartOfSpeech SPECIAL_PART1, std::size_t EXTRA_WEIGHT1, std::size_t DEFAULT_EXTRA_WEIGHT>
    class CWeightOnePart {
    public:
        std::size_t operator()(EPartOfSpeech partOfSpeech) {
            if (partOfSpeech == E_NotInDictionary) {
                return 0;
            }
            return (partOfSpeech == SPECIAL_PART1) ? EXTRA_WEIGHT1 : DEFAULT_EXTRA_WEIGHT;
        }

        void reset() {
            // NO-OP
        }

        std::size_t minMatchingWeight(std::size_t weight) { return weight; }

        std::size_t maxMatchingWeight(std::size_t weight) { return weight; }
    };

    using TWeightVerbs5Other2 = CWeightOnePart<E_Verb, 5, 2>;

    //! Functor for weighting one type of dictionary word by a certain
    //! amount and all dictionary words by a different amount, giving a boost to
    //! the weight when 3 or more dictionary words are adjacent one another.
    template<EPartOfSpeech SPECIAL_PART1, std::size_t EXTRA_WEIGHT1, std::size_t DEFAULT_EXTRA_WEIGHT, std::size_t ADJACENT_PARTS_BOOST>
    class CWeightOnePartAdjacentBoost {
    public:
        std::size_t operator()(EPartOfSpeech partOfSpeech) {
            if (partOfSpeech == E_NotInDictionary) {
                m_NumOfAdjacentDictionaryWords = 0;
                return 0;
            }

            std::size_t weight = (partOfSpeech == SPECIAL_PART1) ? EXTRA_WEIGHT1 : DEFAULT_EXTRA_WEIGHT;
            std::size_t boost = (++m_NumOfAdjacentDictionaryWords > 2) ? ADJACENT_PARTS_BOOST
                                                                       : 1;
            weight *= boost;

            return weight;
        }

        void reset() { m_NumOfAdjacentDictionaryWords = 0; }

        std::size_t minMatchingWeight(std::size_t weight) {
            return (weight <= ADJACENT_PARTS_BOOST)
                       ? weight
                       : (1 + (weight - 1) / ADJACENT_PARTS_BOOST);
        }

        std::size_t maxMatchingWeight(std::size_t weight) {
            return (weight <= std::min(EXTRA_WEIGHT1, DEFAULT_EXTRA_WEIGHT) ||
                    weight > std::max(EXTRA_WEIGHT1 + 1, DEFAULT_EXTRA_WEIGHT + 1))
                       ? weight
                       : (1 + (weight - 1) * ADJACENT_PARTS_BOOST);
        }

    private:
        std::size_t m_NumOfAdjacentDictionaryWords = 0;
    };

    using TWeightVerbs5Other2AdjacentBoost6 = CWeightOnePartAdjacentBoost<E_Verb, 5, 2, 6>;

    //! Functor for weighting two types of dictionary word by certain
    //! amounts and all dictionary words by a different amount
    template<EPartOfSpeech SPECIAL_PART1, std::size_t EXTRA_WEIGHT1, EPartOfSpeech SPECIAL_PART2, std::size_t EXTRA_WEIGHT2, std::size_t DEFAULT_EXTRA_WEIGHT>
    class CWeightTwoParts {
    public:
        std::size_t operator()(EPartOfSpeech partOfSpeech) {
            if (partOfSpeech == E_NotInDictionary) {
                return 0;
            }
            if (partOfSpeech == SPECIAL_PART1) {
                return EXTRA_WEIGHT1;
            }
            return (partOfSpeech == SPECIAL_PART2) ? EXTRA_WEIGHT2 : DEFAULT_EXTRA_WEIGHT;
        }

        void reset() {
            // NO-OP
        }

        std::size_t minMatchingWeight(std::size_t weight) { return weight; }

        std::size_t maxMatchingWeight(std::size_t weight) { return weight; }
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

    //! No copying
    CWordDictionary(const CWordDictionary&) = delete;
    CWordDictionary& operator=(const CWordDictionary&) = delete;

private:
    //! Constructor for a singleton is private
    CWordDictionary();
    ~CWordDictionary();

private:
    class CStrHashIgnoreCase {
    public:
        std::size_t operator()(const std::string& str) const;
    };

    class CStrEqualIgnoreCase {
    public:
        bool operator()(const std::string& lhs, const std::string& rhs) const;
    };

private:
    //! Name of the file to load that contains the dictionary words.
    static const char* const DICTIONARY_FILE;

    //! This pointer is set after the singleton object has been constructed,
    //! and avoids the need to lock the magic static initialisation mutex on
    //! subsequent calls of the instance() method (once the updated value of
    //! this variable is visible in every thread).
    static CWordDictionary* ms_Instance;

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
