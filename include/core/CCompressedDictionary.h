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

#ifndef INCLUDED_ml_core_CCompressedDictionary_h
#define INCLUDED_ml_core_CCompressedDictionary_h

#include <core/CContainerPrinter.h>
#include <core/CHashing.h>
#include <core/CStringUtils.h>

#include <boost/operators.hpp>
#include <boost/unordered/unordered_map_fwd.hpp>
#include <boost/unordered/unordered_set_fwd.hpp>

#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <type_traits>

namespace ml {
namespace core {

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
class CCompressedDictionary {
public:
    using TUInt64Array = std::array<std::uint64_t, N>;

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
    class CWord
        : private boost::equality_comparable1<CWord, boost::less_than_comparable<CWord>> {
    public:
        //! See CMemory.
        static bool dynamicSizeAlwaysZero() { return true; }

        //! Used for converting to and from a delimited string.
        static const char DELIMITER;

    public:
        CWord() = default;
        explicit CWord(const TUInt64Array& hash) : m_Hash(hash) {}

        bool operator==(const CWord& other) const {
            return m_Hash == other.m_Hash;
        }

        bool operator<(const CWord& rhs) const { return m_Hash < rhs.m_Hash; }

        bool fromDelimited(const std::string& delimited) {
            // Expect N strings separated by commas.
            std::size_t n{0};
            std::size_t pos{0};
            std::size_t comma{0};
            while ((comma = delimited.find(DELIMITER, pos)) != std::string::npos && n < N) {
                CStringUtils::stringToType(delimited.substr(pos, comma - pos),
                                           m_Hash[n++]);
                pos = comma + 1;
            }
            if (n < N) {
                CStringUtils::stringToType(delimited.substr(pos, comma - pos),
                                           m_Hash[n++]);
            }
            return n == N;
        }

        std::string toDelimited() const {
            std::string result = CStringUtils::typeToString(m_Hash[0]);
            for (std::size_t i = 1; i < N; ++i) {
                result += DELIMITER;
                result += CStringUtils::typeToString(m_Hash[i]);
            }
            return result;
        }

        std::size_t hash() const { return static_cast<std::size_t>(m_Hash[0]); }

        std::uint64_t hash64() const { return m_Hash[0]; }

        std::string print() const { return CContainerPrinter::print(m_Hash); }

    private:
        TUInt64Array m_Hash{};
    };

    //! \brief Translates supported data types to dictionary words.
    //!
    //! DESCRIPTION:\n
    //! We support creating dictionary words for strings, arithmetic types,
    //! enums and vectors of them.
    class CTranslator {
    public:
        explicit CTranslator(const TUInt64Array& seeds) : m_Hashes{seeds} {}

        void add(const std::string& value) {
            for (auto& hash : m_Hashes) {
                hash = CHashing::safeMurmurHash64(
                    value.c_str(), static_cast<int>(value.size()), hash);
            }
        }

        template<typename T>
        void add(const T& value) {
            // We don't support POD types because the initialisation
            // of padding bytes is undefined.
            static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
                          "Unsupported type");
            for (auto& hash : m_Hashes) {
                hash = CHashing::safeMurmurHash64(&value, static_cast<int>(sizeof(T)), hash);
            }
        }

        template<typename T>
        void add(const std::vector<T>& value) {
            static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
                          "Unsupported type");
            for (auto& hash : m_Hashes) {
                hash = CHashing::safeMurmurHash64(
                    value.data(), static_cast<int>(sizeof(T) * value.size()), hash);
            }
        }

        CWord word() const { return CWord{m_Hashes}; }

    private:
        TUInt64Array m_Hashes;
    };

    //! \brief A fast hash of a dictionary word.
    class CHash {
    public:
        inline std::size_t operator()(const CWord& word) const {
            return word.hash();
        }
    };

    //! The type of an ordered set of words.
    using TWordSet = std::set<CWord>;

    //! The type of an unordered set of words.
    using TWordUSet = boost::unordered_set<CWord, CHash>;

    //! A template typedef of an ordered map from words to objects of type T.
    template<typename T>
    using TWordTMap = std::map<CWord, T, CHash>;

    //! A template typedef of an unordered map from words to objects of type T.
    template<typename T>
    using TWordTUMap = boost::unordered_map<CWord, T, CHash>;

public:
    CCompressedDictionary() {
        // 472882027 and 982451653 are prime numbers, so repeatedly
        // multiplying them will give many distinct seeds modulo 2 ^ 64.
        // This method of seed generation was chosen in preference
        // to using uniform random numbers as the Boost random number
        // generators can generate different numbers on different
        // platforms, even with the same random number generator seed.
        m_Seeds[0] = 472882027;
        for (std::size_t i = 1; i < N; ++i) {
            m_Seeds[i] = m_Seeds[i - 1] * 982451653ULL;
        }
    }

    //! Translate \p word to a dictionary word.
    CWord word(const std::string& word) const {
        CTranslator translator{m_Seeds};
        translator.add(word);
        return translator.word();
    }

    //! Translate (\p word1, \p word2) to a dictionary word.
    CWord word(const std::string& word1, const std::string& word2) const {
        CTranslator translator{m_Seeds};
        translator.add(word1);
        translator.add(word2);
        return translator.word();
    }

    //! Translate (\p word1, \p word2, \p word3) to a dictionary word.
    CWord word(const std::string& word1, const std::string& word2, const std::string& word3) const {
        CTranslator translator{m_Seeds};
        translator.add(word1);
        translator.add(word2);
        translator.add(word3);
        return translator.word();
    }

    //! Translate (\p word1, \p word2, \p word3, \p word4) to a dictionary word.
    CWord word(const std::string& word1,
               const std::string& word2,
               const std::string& word3,
               const std::string& word4) const {
        CTranslator translator{m_Seeds};
        translator.add(word1);
        translator.add(word2);
        translator.add(word3);
        translator.add(word4);
        return translator.word();
    }

    //! Get a translator for the dictionary.
    CTranslator translator() const { return CTranslator{m_Seeds}; }

private:
    TUInt64Array m_Seeds;
};

template<std::size_t N>
const char CCompressedDictionary<N>::CWord::DELIMITER(',');
}
}

#endif // INCLUDED_ml_core_CCompressedDictionary_h
