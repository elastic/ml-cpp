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

#ifndef INCLUDED_ml_core_CCompressedDictionary_h
#define INCLUDED_ml_core_CCompressedDictionary_h

#include <core/CContainerPrinter.h>
#include <core/CHashing.h>
#include <core/CStringUtils.h>

#include <boost/array.hpp>
#include <boost/operators.hpp>
#include <boost/unordered/unordered_map_fwd.hpp>
#include <boost/unordered/unordered_set_fwd.hpp>

#include <map>
#include <set>
#include <string>

#include <stdint.h>


namespace ml
{
namespace core
{

//! \brief A space efficient representation of a dictionary.
//!
//! DESCRIPTION:\n
//! This implements the concept of a compressed dictionary for arbitrary
//! strings. The idea is to come up with a concise representation of the
//! word strings that have a very low probability of collision. This uses
//! N near independent 64 bit hashes of the words; so even if N is 2 and
//! 260 million different string words are added to the dictionary we
//! expect two different string words to map to the same dictionary word
//! with a probability of less than 1.5e-4.
//!
//! IMPLMENTATION:\n
//! This uses murmur hash 2 behind the scenes because its distributions
//! are good enough and it is very fast to compute.
template<std::size_t N>
class CCompressedDictionary
{
    public:
        using TUInt64Array = boost::array<uint64_t, N>;
        using TStrCPtr = const std::string*;

        //! \brief A hash representation of a string in the dictionary
        //! with low probability of collision even for relatively large
        //! dictionaries.
        //!
        //! DESCRIPTION:\n
        //! We use N near independent 64 bit hashes of the strings.
        //! Assuming the distributions are uniform (which is a reasonable
        //! assumption, see below), the probability of a collision in
        //! any of the hashes is:
        //! <pre class="fragment">
        //!   \f$\displaystyle P = 1 - \prod_{i=0}^{n}{\left(1 - \frac{i}{2^{64}}\right)}\f$
        //! </pre>
        //! (See http://en.wikipedia.org/wiki/Birthday_problem.)
        //!
        //! It is relatively straightforward to show that for large
        //! \f$n\f$ this is approximately:
        //! <pre class="fragment">
        //!   \f$\displaystyle P = 1 - \exp(\frac{-n^2}{2^{64}})\f$
        //! </pre>
        //!
        //! So, for example, if \f$n = 2^{28}\f$, which would be a
        //! dictionary containing 268 million words the probability
        //! of a single hash colliding is around 0.004. So for N
        //! independent hashes we get a probability of all hashes
        //! colliding of \f$0.004^N\f$. If N is 2 the probability of
        //! a collision is \f$1.5 \times 10^{-5}\f$.
        class CWord : private boost::equality_comparable1<CWord,
                              boost::less_than_comparable<CWord> >
        {
            public:
                //! See CMemory.
                static bool dynamicSizeAlwaysZero(void) { return true; }

                //! Used for converting to and from a delimited string.
                static const char DELIMITER;

            public:
                CWord(void)
                {
                    std::fill(m_Hash.begin(), m_Hash.end(), 0);
                }
                CWord(const TUInt64Array &hash) : m_Hash(hash) {}

                bool operator==(const CWord &other) const
                {
                    return m_Hash == other.m_Hash;
                }

                bool operator<(const CWord &rhs) const
                {
                    return m_Hash < rhs.m_Hash;
                }

                bool fromDelimited(const std::string &str)
                {
                    // expect N strings, separated by commas
                    std::size_t n = 0;
                    std::size_t pos = 0;
                    std::size_t comma = 0;
                    while ((comma = str.find(DELIMITER, pos)) != std::string::npos && n < N)
                    {
                        CStringUtils::stringToType(str.substr(pos, comma - pos), m_Hash[n++]);
                        pos = comma + 1;
                    }
                    if (n < N)
                    {
                        CStringUtils::stringToType(str.substr(pos, comma - pos), m_Hash[n++]);
                    }
                    return n == N;
                }

                std::string toDelimited(void) const
                {
                    std::string result = CStringUtils::typeToString(m_Hash[0]);
                    for (std::size_t i = 1; i < N; ++i)
                    {
                        result += DELIMITER;
                        result += CStringUtils::typeToString(m_Hash[i]);
                    }
                    return result;
                }

                std::size_t hash(void) const
                {
                    return static_cast<std::size_t>(m_Hash[0]);
                }

                uint64_t hash64(void) const
                {
                    return m_Hash[0];
                }

                std::string print(void) const
                {
                    return CContainerPrinter::print(m_Hash);
                }

            private:
                TUInt64Array m_Hash;
        };

        //! \brief A fast hash of a dictionary word.
        class CHash : public std::unary_function<CWord, uint64_t>
        {
            public:
                inline std::size_t operator()(const CWord &word) const
                {
                    return word.hash();
                }
        };

        //! The type of an ordered set of words.
        using TWordSet = std::set<CWord>;

        //! The type of an unordered set of words.
        using TWordUSet = boost::unordered_set<CWord, CHash>;

        //! A "template typedef" of an ordered map from words to
        //! objects of type T.
        template<typename T>
        class CWordMap
        {
            public:
                using Type = std::map<CWord, T>;
        };

        //! A "template typedef" of an unordered map from words to
        //! objects of type T.
        template<typename T>
        class CWordUMap
        {
            public:
                using Type = boost::unordered_map<CWord, T, CHash>;
        };

    public:
        CCompressedDictionary(void)
        {
            // 472882027 and 982451653 are prime numbers, so repeatedly
            // multiplying them will give many distinct seeds modulo 2 ^ 64.
            // This method of seed generation was chosen in preference
            // to using uniform random numbers as the Boost random number
            // generators can generate different numbers on different
            // platforms, even with the same random number generator seed.
            m_Seeds[0] = 472882027;
            for (std::size_t i = 1u; i < N; ++i)
            {
                m_Seeds[i] = m_Seeds[i - 1] * 982451653ull;
            }
        }

        //! Extract the dictionary word corresponding to \p word.
        CWord word(const std::string &word) const
        {
            TUInt64Array hash;
            for (std::size_t i = 0u; i < N; ++i)
            {
                hash[i] = CHashing::safeMurmurHash64(word.c_str(),
                                                     static_cast<int>(word.size()),
                                                     m_Seeds[i]);
            }
            return CWord(hash);
        }

        //! Extract the dictionary word corresponding to (\p word1, \p word2).
        CWord word(const std::string &word1,
                   const std::string &word2) const
        {
            TStrCPtr words[] = {&word1, &word2};
            return this->word(words);
        }

        //! Extract the dictionary word corresponding to (\p word1, \p word2, \p word3).
        CWord word(const std::string &word1,
                   const std::string &word2,
                   const std::string &word3) const
        {
            TStrCPtr words[] = {&word1, &word2, &word3};
            return this->word(words);
        }

        //! Extract the dictionary word corresponding to (\p word1, \p word2, \p word3, \p word4).
        CWord word(const std::string &word1,
                   const std::string &word2,
                   const std::string &word3,
                   const std::string &word4) const
        {
            TStrCPtr words[] = {&word1, &word2, &word3, &word4};
            return this->word(words);
        }

    private:
        template<std::size_t NUMBER_OF_WORDS>
        CWord word(const TStrCPtr (&words)[NUMBER_OF_WORDS]) const
        {
            TUInt64Array hashes;
            for (std::size_t i = 0u; i < N; ++i)
            {
                uint64_t &hash = hashes[i];
                for (std::size_t wordIndex = 0; wordIndex < NUMBER_OF_WORDS; ++wordIndex)
                {
                    const std::string &word = *words[wordIndex];
                    hash = CHashing::safeMurmurHash64(word.c_str(),
                                                      static_cast<int>(word.size()),
                                                      (wordIndex) == 0 ? m_Seeds[i] : hash);
                }
            }
            return CWord(hashes);
        }

    private:
        TUInt64Array m_Seeds;
};

template<std::size_t N>
const char CCompressedDictionary<N>::CWord::DELIMITER(',');

}
}

#endif // INCLUDED_ml_core_CCompressedDictionary_h
