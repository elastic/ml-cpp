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

#ifndef INCLUDED_ml_maths_CPRNG_h
#define INCLUDED_ml_maths_CPRNG_h

#include <core/CNonInstantiatable.h>

#include <maths/ImportExport.h>

#include <cstddef>
#include <string>

#include <stdint.h>

namespace ml {
namespace maths {

//! \brief A collection of pseudo random number generators.
//!
//! DESCRIPTION:\n
//! The xoroshiro128+ and xorshift1024* are our two pseudo-random
//! number generators of choice, with the tradeoff being between
//! state size and cycle length and equidistribution.
//!
//! SplitMix is intended mainly for seeding the other two generators.
//!
//! IMPLEMENTATION:\n
//! The generators all implement the contract for a boost pseudo-
//! random number generator, so they can be used freely with the
//! boost::random library.
class MATHS_EXPORT CPRNG : private core::CNonInstantiatable {
private:
    //! Fills [\p begin, \p end) with consecutive random numbers
    //! generated by \p rng.
    template <typename PRNG, typename ITR> static void generate(PRNG &rng, ITR begin, ITR end) {
        for (/**/; begin != end; ++begin) {
            *begin = rng();
        }
    }

public:
    //! \brief The split mix pseudo-random number generator.
    //!
    //! DESCRIPTION:\n
    //! A \f$2^{64}\f$ period pseudo-random number generator based
    //! on Java's splittable random number generator.
    class MATHS_EXPORT CSplitMix64 {
    public:
        typedef uint64_t result_type;

    public:
        CSplitMix64(void);
        CSplitMix64(uint64_t seed);

        //! Compare for equality.
        bool operator==(CSplitMix64 other) const;
        //! Not equal.
        bool operator!=(CSplitMix64 other) const { return !this->operator==(other); }

        void seed(void);
        void seed(uint64_t seed);

        //! The minimum value returnable by operator().
        static uint64_t min(void);
        //! The maximum value returnable by operator().
        static uint64_t max(void);

        //! Generate the next random number.
        uint64_t operator()(void);

        //! Fill the sequence [\p begin, \p end) with the next
        //! \p end - \p begin random numbers.
        template <typename ITR> void generate(ITR begin, ITR end) {
            CPRNG::generate(*this, begin, end);
        }

        //! Discard the next \p n random numbers.
        void discard(uint64_t n);

        //! Persist to a string.
        std::string toString(void) const;
        //! Restore from a string.
        bool fromString(const std::string &state);

    private:
        static const uint64_t A;
        static const uint64_t B;
        static const uint64_t C;

    private:
        //! The state.
        uint64_t m_X;
    };

    //! \brief The xoroshiro128+ pseudo-random number generator.
    //!
    //! DESCRIPTION:\n
    //! A v.fast \f$2^{128}-1\f$ period pseudo-random number
    //! generator with v.good empirical statistical properties.
    //!
    //! The lowest bit is an LFSR so use a sign test to extract
    //! a random Boolean value.
    class MATHS_EXPORT CXorOShiro128Plus {
    public:
        typedef uint64_t result_type;

    public:
        CXorOShiro128Plus(void);
        CXorOShiro128Plus(uint64_t seed);
        template <typename ITR> CXorOShiro128Plus(ITR begin, ITR end) { this->seed(begin, end); }

        //! Compare for equality.
        bool operator==(const CXorOShiro128Plus &other) const;
        //! Not equal.
        bool operator!=(const CXorOShiro128Plus &other) const { return !this->operator==(other); }

        //! Set to the default seeded generator.
        //!
        //! As per recommendations we use CSplitMix64 for seeding.
        void seed(void);
        //! Set to a seeded generator.
        //!
        //! As per recommendations we use CSplitMix64 for seeding.
        void seed(uint64_t seed);
        //! Seed from [\p begin, \p end) which should have two 64 bit
        //! seeds.
        template <typename ITR> void seed(ITR begin, ITR end) {
            std::size_t i = 0u;
            for (/**/; i < 2 && begin != end; ++i, ++begin) {
                m_X[i] = *begin;
            }
            if (i < 2) {
                CSplitMix64 seeds;
                seeds.generate(&m_X[i], &m_X[2]);
            }
        }

        //! The minimum value returnable by operator().
        static uint64_t min(void);
        //! The maximum value returnable by operator().
        static uint64_t max(void);

        //! Generate the next random number.
        uint64_t operator()(void);

        //! Fill the sequence [\p begin, \p end) with the next
        //! \p end - \p begin random numbers.
        template <typename ITR> void generate(ITR begin, ITR end) {
            CPRNG::generate(*this, begin, end);
        }

        //! Discard the next \p n random numbers.
        void discard(uint64_t n);

        //! This is equivalent to \f$2^{64}\f$ calls to next();
        //! it can be used to generate \f$2^{64}\f$ non-overlapping
        //! subsequences of length \f$2^{64}\f$ for parallel
        //! computations.
        void jump(void);

        //! Persist to a string.
        std::string toString(void) const;
        //! Restore from a string.
        bool fromString(const std::string &state);

    private:
        static const uint64_t JUMP[2];

    private:
        //! The state.
        uint64_t m_X[2];
    };

    //! \brief The xorshift1024* pseudo-random number generator.
    //!
    //! DESCRIPTION:\n
    //! A \f$2^{1024}-1\f$ period pseudo-random number generator
    //! with v.good empirical statistical properties.
    //!
    //! Note that the three lowest bits of this generator are LSFRs,
    //! and thus they are slightly less random than the other bits.
    //! Use a sign test to extract a random Boolean value.
    //!
    //! \sa https://en.wikipedia.org/wiki/Xorshift#cite_note-vigna2-9.
    class MATHS_EXPORT CXorShift1024Mult {
    public:
        typedef uint64_t result_type;

    public:
        CXorShift1024Mult(void);
        CXorShift1024Mult(uint64_t seed);
        template <typename ITR> CXorShift1024Mult(ITR begin, ITR end) : m_P(0) {
            this->seed(begin, end);
        }

        //! Compare for equality.
        bool operator==(const CXorShift1024Mult &other) const;
        //! Not equal.
        bool operator!=(const CXorShift1024Mult &other) const { return !this->operator==(other); }

        //! Set to the default seeded generator.
        //!
        //! As per recommendations we use CSplitMix64 for seeding.
        void seed(void);
        //! Set to a seeded generator.
        //!
        //! As per recommendations we use CSplitMix64 for seeding.
        void seed(uint64_t seed);
        //! Seed from [\p begin, \p end) which should have sixteen
        //! 64 bit seeds.
        template <typename ITR> void seed(ITR begin, ITR end) {
            std::size_t i = 0u;
            for (/**/; i < 16 && begin != end; ++i, ++begin) {
                m_X[i] = *begin;
            }
            if (i < 16) {
                CSplitMix64 seeds;
                seeds.generate(&m_X[i], &m_X[16]);
            }
        }

        //! The minimum value returnable by operator().
        static uint64_t min(void);
        //! The maximum value returnable by operator().
        static uint64_t max(void);

        //! Generate the next random number.
        uint64_t operator()(void);

        //! Fill the sequence [\p begin, \p end) with the next
        //! \p end - \p begin random numbers.
        template <typename ITR> void generate(ITR begin, ITR end) {
            CPRNG::generate(*this, begin, end);
        }

        //! Discard the next \p n random numbers.
        void discard(uint64_t n);

        //! This is equivalent to \f$2^{512}\f$ calls to next();
        //! it can be used to generate \f$2^{512}\f$ non-overlapping
        //! subsequences of length \f$2^{512}\f$ for parallel
        //! computations.
        void jump(void);

        //! Persist to a string.
        std::string toString(void) const;
        //! Restore from a string.
        bool fromString(std::string state);

    private:
        static const uint64_t A;
        static const uint64_t JUMP[16];

    private:
        //! The state.
        uint64_t m_X[16];
        //! The current pair.
        int m_P;
    };
};
}
}

#endif// INCLUDED_ml_maths_CPRNG_h
