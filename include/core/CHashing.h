/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_core_CHashing_h
#define INCLUDED_ml_core_CHashing_h

#include <core/CFastMutex.h>
#include <core/CNonInstantiatable.h>
#include <core/CStoredStringPtr.h>
#include <core/ImportExport.h>

#include <boost/random/mersenne_twister.hpp>

#include <functional>
#include <string>
#include <vector>

#include <stdint.h>

namespace ml {
namespace core {

//! \brief Hashing functionality.
//!
//! DESCRIPTION:\n
//! This is a place holder for useful hashing functionality. In particular,
//! it implements various universal hashing schemes, some high quality hash
//! functions (near cryptographic, but much faster).
class CORE_EXPORT CHashing : private CNonInstantiatable {
public:
    //! Implements universal hashing on integers.
    //!
    //! See http://en.wikipedia.org/wiki/Universal_hashing for discussion.
    //!
    //! \warning The hashes use the prime 4294967291 so rely on the fact
    //! that integers being hashed are smaller than this. This should be
    //! sufficient for our use cases!
    class CORE_EXPORT CUniversalHash {
    public:
        using TUInt32Vec = std::vector<uint32_t>;

    public:
        //! A member of the universal (2-independent) hash family on
        //! integers (Carter and Wegman):
        //! <pre class="fragment">
        //!   \f$\displaystyle f_{a,b}(x) = ((ax + b)\ \textup{mod}\ p)\ \textup{mod}\ m\f$
        //! </pre>
        //!
        //! \note This is not the fastest implementation of universal
        //! hashing which is the multiply-shift scheme. We can revisit
        //! the implementation if this is a bottleneck in practice.
        class CORE_EXPORT CUInt32Hash {
        public:
            //! See CMemory.
            static bool dynamicSizeAlwaysZero() { return true; }

        public:
            CUInt32Hash();
            CUInt32Hash(uint32_t m, uint32_t a, uint32_t b);

            //! Get the range.
            uint32_t m() const;

            //! Get the multiplier.
            uint32_t a() const;

            //! Get the offset.
            uint32_t b() const;

            //! \note This is implemented inline in contravention to
            //! the coding standards because we definitely don't want
            //! the cost of a function call here.
            uint32_t operator()(uint32_t x) const {
                // Note by construction:
                //   a * x + b < p^2 + p < 2^64
                return static_cast<uint32_t>(
                    ((static_cast<uint64_t>(m_A) * x + static_cast<uint64_t>(m_B)) % BIG_PRIME) %
                    static_cast<uint64_t>(m_M));
            }

            //! Print the hash function for debug.
            std::string print() const;

        private:
            uint32_t m_M, m_A, m_B;
        };

        using TUInt32HashVec = std::vector<CUInt32Hash>;

        //! A lightweight implementation universal (2-independent) on
        //! 32-bit integers. This doesn't further restrict the range
        //! of the hash so avoids storing an extra integer and doing
        //! modulo the range.
        class CORE_EXPORT CUInt32UnrestrictedHash {
        public:
            //! See CMemory.
            static bool dynamicSizeAlwaysZero() { return true; }

        public:
            CUInt32UnrestrictedHash();
            CUInt32UnrestrictedHash(uint32_t a, uint32_t b);

            //! Get the multiplier.
            uint32_t a() const;

            //! Get the offset.
            uint32_t b() const;

            //! \note This is implemented inline in contravention to
            //! the coding standards because we definitely don't want
            //! the cost of a function call here.
            uint32_t operator()(uint32_t x) const {
                // Note by construction:
                //   a * x + b < p^2 + p < 2^64
                return static_cast<uint32_t>(
                    (static_cast<uint64_t>(m_A) * x + static_cast<uint64_t>(m_B)) % BIG_PRIME);
            }

            //! Print the hash function for debug.
            std::string print() const;

        private:
            uint32_t m_A, m_B;
        };

        using TUInt32UnrestrictedHashVec = std::vector<CUInt32UnrestrictedHash>;

        //! A member of the universal (2-independent) hash family on
        //! vectors of integers (Carter and Wegman):
        //! <pre class="fragment">
        //!   \f$\displaystyle f(\{x_i\}) = \sum_i{h_i(x_i)}\ \textup{mod}\ m\f$
        //! </pre>
        //!
        //! Here, \f$h_i(.)\f$ are independent samples from a universal
        //! family of hash functions.
        //!
        //! \note This is not the fastest implementation of universal
        //! hashing which is the multiply-shift scheme. We can revisit
        //! the implementation if this is a bottleneck in practice.
        class CORE_EXPORT CUInt32VecHash {
        public:
            CUInt32VecHash(uint32_t m, const TUInt32Vec& a, uint32_t b);

            //! Get the range.
            uint32_t m() const;

            //! Get the multipliers.
            const TUInt32Vec& a() const;

            //! Get the offset.
            uint32_t b() const;

            //! Overload for case our vector has two elements to
            //! avoid overhead of creating a vector to hash.
            //!
            //! \note This is implemented inline in contravention to
            //! the coding standards because we definitely don't want
            //! the cost of a function call here.
            uint32_t operator()(uint32_t x1, uint32_t x2) const {
                // Note by construction:
                //   (a(1) * x(1)) mod p + a(2) * x(2) + b
                //     < p^2 + 2*p
                //     < 2^64
                uint64_t h = (static_cast<uint64_t>(m_A[0]) * x1) % BIG_PRIME +
                             static_cast<uint64_t>(m_A[1]) * x2;
                return static_cast<uint32_t>(((h + static_cast<uint64_t>(m_B)) % BIG_PRIME) %
                                             static_cast<uint64_t>(m_M));
            }

            //! \note This is implemented inline in contravention to
            //! the coding standards because we definitely don't want
            //! the cost of a function call here.
            uint32_t operator()(const TUInt32Vec& x) const {
                // Note we variously use that:
                //   a(1) * x(1)
                //     < h mod p + a(i) * x(i)
                //     < h mod p + a(n) * x(n) + b
                //     < p^2 + 2*p
                //     < 2^64
                uint64_t h = static_cast<uint64_t>(m_A[0]) * x[0];
                for (std::size_t i = 1u; i < x.size(); ++i) {
                    h = (h % BIG_PRIME + static_cast<uint64_t>(m_A[i]) * x[i]);
                }
                return static_cast<uint32_t>(((h + static_cast<uint64_t>(m_B)) % BIG_PRIME) %
                                             static_cast<uint64_t>(m_M));
            }

            //! Print the hash function for debug.
            std::string print() const;

        private:
            uint32_t m_M;
            TUInt32Vec m_A;
            uint32_t m_B;
        };

        using TUInt32VecHashVec = std::vector<CUInt32VecHash>;

        //! Converts hash function objects to a string.
        class CORE_EXPORT CToString {
        public:
            CToString(const char delimiter);

            std::string operator()(const CUInt32UnrestrictedHash& hash) const;
            std::string operator()(const CUInt32Hash& hash) const;

        private:
            char m_Delimiter;
        };

        //! Initializes hash function objects from a string.
        class CORE_EXPORT CFromString {
        public:
            CFromString(const char delimiter);

            bool operator()(const std::string& token, CUInt32UnrestrictedHash& hash) const;
            bool operator()(const std::string& token, CUInt32Hash& hash) const;

        private:
            char m_Delimiter;
            mutable std::string m_Token;
        };

    public:
        //! We choose a prime just a smaller than \f$2^{32}\f$. Note
        //! that if unsigned integer multiplication overflows the result
        //! is returned modulo \f$2^{64}\f$ for 64 bit integers however:
        //! <pre class="fragment">
        //!   \f$(ax + b)\textup{ mod }p \neq ((ax + b)\textup{ mod }2^{64})\textup{ mod }p\f$
        //! </pre>
        //!
        //! So in order to guaranty that the hash functions belong
        //! to a universal family for all possible universes we need
        //! to avoid overflow (or explicitly handle it). We choose
        //! to almost always guaranty it by choosing:\n
        //! <pre class="fragment">
        //!   \f$p = 4294967291\f$
        //! </pre>
        //!
        //! which is the largest prime less than \f$2^{32}\f$. See:
        //! http://www.prime-numbers.org/prime-number-4294965000-4294970000.htm
        //! and by make use of the fact that we can take the mod at any
        //! time because if two integers are equal modulo \f$p\f$ they
        //! are identical (in \f$Z/p\f$) so can be interchanged in any
        //! statement which is true for one or the other. For most
        //! applications we'll be mapping unique values (client ids, etc)
        //! to the first n integers so 4294967291 will be plenty big enough
        //! for our universes.
        static const uint64_t BIG_PRIME;

        //! Generate k independent samples of the 32 bit integer universal
        //! hash functions:
        //! <pre class="fragment">
        //!   \f$\displaystyle h_{a,b}\ :\ U \rightarrow [m-1]\f$
        //! </pre>
        //!
        //! \param k The number of hash functions.
        //! \param m The range of the hash functions.
        //! \param result Filled in with the sampled hash functions.
        static void generateHashes(std::size_t k, uint32_t m, TUInt32HashVec& result);

        //! Generate k independent samples of the 32 bit integer universal
        //! hash functions:
        //! <pre class="fragment">
        //!   \f$\displaystyle h_{a,b}\ :\ U \rightarrow [2^{32}]\f$
        //! </pre>
        //!
        //! \param k The number of hash functions.
        //! \param result Filled in with the sampled hash functions.
        static void generateHashes(std::size_t k, TUInt32UnrestrictedHashVec& result);

        //! Generate k independent samples of the 32 bit integer vector
        //! universal hash functions:
        //! <pre class="fragment">
        //!   \f$\displaystyle h\ :\ U \rightarrow [m-1]\f$
        //! </pre>
        //!
        //! \param k The number of hash functions.
        //! \param n The size of vectors to hash.
        //! \param m The range of the hash functions.
        //! \param result Filled in with the sampled hash functions.
        static void generateHashes(std::size_t k, std::size_t n, uint32_t m, TUInt32VecHashVec& result);

    private:
        //! Our random number generator for sampling hash function.
        static boost::random::mt11213b ms_Generator;

        //! Used by generateHashes to protect non thread safe calls
        //! to the random number generator.
        static CFastMutex ms_Mutex;
    };

    //! MurmurHash2: fast 32-bit hash.
    //!
    //! This is very close to Austin Appleby's optimized implementation
    //! of the MurmurHash2 function (which is now in the public domain).
    //! This is neither endian neutral nor alignment safe. If you are
    //! going to use this version you must either be confident or check
    //! that the address of \p key can be safely read. For addressable
    //! values this will be the case, but might for example fail if you
    //! pass in a pointer to the middle of a string. Note that hashing
    //! whole strings will be fine. If you need to check alignment, on
    //! a 32-bit platform it amounts to checking that:
    //! \code
    //!   reinterpret_cast<std::size_t>(key) & 0x3 == 0
    //! \endcode
    //!
    //! Furthermore, you should not serialize the hashed values because
    //! they will be different on machines with different endian
    //! conventions. If you aren't sure that you can safely use this
    //! version then use safeMurmurHash32.
    static uint32_t murmurHash32(const void* key, int length, uint32_t seed);

    //! MurmurHash2: safe 32-bit hash.
    //!
    //! This is very close to Austin Appleby's neutral implementation
    //! of MurmurHash2 (which is now in the public domain). This is
    //! both alignment safe and endian neutral. I have factored this
    //! out from our fastest MurmurHash2 implementation because I
    //! don't want the result of hashing to depend on the address of
    //! the object which it would if we tried to mix the two approaches
    //! and check alignment.
    static uint32_t safeMurmurHash32(const void* key, int length, uint32_t seed);

    //! MurmurHash2: fast 64-bit hash.
    //!
    //! This is adapted from Austin Appleby's optimized implementation
    //! of the 32 bit MurmurHash2 function (which is now in the public
    //! domain). This is neither endian neutral nor alignment safe. If
    //! you are going to use this version you must either be confident
    //! or check that the address of \p key can be safely read. For
    //! addressable values this will be the case, but might for example
    //! fail if you pass in a pointer to the middle of a string. Note
    //! that hashing whole strings will be fine. If you need to check
    //! alignment, on a 64-bit platform it amounts to checking that:
    //! \code
    //!   reinterpret_cast<std::size_t>(key) & 0x7 == 0
    //! \endcode
    //!
    //! Furthermore, you should not serialize the hashed values because
    //! they will be different on machines with different endian
    //! conventions. If you aren't sure that you can safely use this
    //! version then use safeMurmurHash64.
    static uint64_t murmurHash64(const void* key, int length, uint64_t seed);

    //! MurmurHash2: safe 64-bit hash.
    //!
    //! This is adapted from Austin Appleby's neutral implementation
    //! of the 32-bit MurmurHash2 (which is now in the public domain).
    //! This is both alignment safe and endian neutral. I have factored
    //! this out from our fastest MurmurHash2 implementation because I
    //! don't want the result of hashing to depend on the address of
    //! the object, which it would if we tried to mix the two approaches
    //! and check alignment.
    static uint64_t safeMurmurHash64(const void* key, int length, uint64_t seed);

    //! Wrapper for murmur hash to use with basic types.
    //!
    //! \warning This is slower than boost::hash for the types I tested
    //! std::size_t, int, uint64_t, but does have better distributions.
    template<typename T>
    class CMurmurHash2BT : public std::unary_function<T, std::size_t> {
    public:
        //! See CMemory.
        static bool dynamicSizeAlwaysZero() { return true; }

    public:
        CMurmurHash2BT(std::size_t seed = 0x5bd1e995) : m_Seed(seed) {}

        std::size_t operator()(const T& key) const;

    private:
        std::size_t m_Seed;
    };

    //! Wrapper for murmur hash to use with std::string.
    //!
    //! \note This is significantly faster than boost::hash<std::string>
    //! and has better distributions.
    class CORE_EXPORT CMurmurHash2String
        : public std::unary_function<std::string, std::size_t> {
    public:
        //! See CMemory.
        static bool dynamicSizeAlwaysZero() { return true; }
        using TStrCRef = std::reference_wrapper<const std::string>;

    public:
        CMurmurHash2String(std::size_t seed = 0x5bd1e995) : m_Seed(seed) {}

        std::size_t operator()(const std::string& key) const;
        std::size_t operator()(TStrCRef key) const {
            return this->operator()(key.get());
        }
        std::size_t operator()(const CStoredStringPtr& key) const {
            if (key) {
                return this->operator()(*key);
            }
            return m_Seed;
        }

    private:
        std::size_t m_Seed;
    };

    //! Wrapper for murmur hash to use with std::string in cases where a 64
    //! bit hash is required rather than a machine word size hash.  An
    //! example would be where the hash value somehow affects data that is
    //! visible outside the program, such as state persisted to a data
    //! store.  This is also immune to endianness issues.
    class CORE_EXPORT CSafeMurmurHash2String64
        : public std::unary_function<std::string, uint64_t> {
    public:
        //! See CMemory.
        static bool dynamicSizeAlwaysZero() { return true; }
        using TStrCRef = std::reference_wrapper<const std::string>;

    public:
        CSafeMurmurHash2String64(uint64_t seed = 0x5bd1e995) : m_Seed(seed) {}

        uint64_t operator()(const std::string& key) const;
        std::size_t operator()(TStrCRef key) const {
            return this->operator()(key.get());
        }
        std::size_t operator()(const CStoredStringPtr& key) const {
            if (key) {
                return this->operator()(*key);
            }
            return m_Seed;
        }

    private:
        uint64_t m_Seed;
    };

    //! 32 bit hash combine modeled on boost::hash_combine.
    static uint32_t hashCombine(uint32_t seed, uint32_t h);

    //! 64 bit hash combine modeled on boost::hash_combine.
    static uint64_t hashCombine(uint64_t seed, uint64_t h);
};

namespace hash_detail {

//! Selects MurmurHash2 32-bit implementation by default.
template<std::size_t>
struct SMurmurHashForArchitecture {
    static std::size_t hash(const void* key, int length, std::size_t seed) {
        return static_cast<std::size_t>(
            CHashing::murmurHash32(key, length, static_cast<uint32_t>(seed)));
    }
};

//! Selects MurmurHash2 64-bit implementation if we are on a 64-bit platform.
//!
//! If we are on 64-bit platforms the 64-bit implementation is faster.
template<>
struct SMurmurHashForArchitecture<8> {
    static std::size_t hash(const void* key, int length, std::size_t seed) {
        return static_cast<std::size_t>(CHashing::murmurHash64(key, length, seed));
    }
};
}

template<typename T>
inline std::size_t CHashing::CMurmurHash2BT<T>::operator()(const T& key) const {
    return hash_detail::SMurmurHashForArchitecture<sizeof(std::size_t)>::hash(
        &key, static_cast<int>(sizeof(key)), m_Seed);
}

inline std::size_t CHashing::CMurmurHash2String::operator()(const std::string& key) const {
    return hash_detail::SMurmurHashForArchitecture<sizeof(std::size_t)>::hash(
        key.data(), static_cast<int>(key.size()), m_Seed);
}

inline uint64_t CHashing::CSafeMurmurHash2String64::operator()(const std::string& key) const {
    return CHashing::safeMurmurHash64(key.data(), static_cast<int>(key.size()), m_Seed);
}
}
}

#endif // INCLUDED_ml_core_CHashing_h
