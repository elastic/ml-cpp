/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_core_CPackedBitVector_h
#define INCLUDED_ml_core_CPackedBitVector_h

#include <core/CMemoryUsage.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/ImportExport.h>

#include <boost/operators.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ml {
namespace core {

//! \brief A compact representation of binary vector.
//!
//! DESCRIPTION:\n
//! This implements a run length encoding of a binary vector (string of 0's and
//! 1's). It has a container like interface which somewhat emulates std::bitset.
//! It supports computing bitwise boolean operations of two vectors and iterating
//! over the indices of the 1 bits. It also has a vector like interface for
//! computing the inner product and related norms (using the conventional definition
//! as the count of the 1 bits in the bitwise and of the two vectors). The count
//! of one bits for the or and exclusive or of two vectors can be computed with
//! the same line scan so this is also supported by supplying the predicate to the
//! inner product.
//!
//! IMPLEMENTATION:\n
//! The space optimal vector depends on the average run length. In particular, it
//! is optimum to use around log2(E[run length]) bits to encode each run. We expect
//! relative short runs in our target applications so stick with std::uint8_t to
//! encode the run length.
//!
//! Because there are only two values we need only store the value of the first bit
//! in the vector and can deduce all other values by the number of runs in between.
//! In practice we store one extra bit, the vector parity to allow us to extend the
//! vector efficiently.
// clang-format off
class CORE_EXPORT CPackedBitVector : private boost::equality_comparable<CPackedBitVector,
                                             boost::partially_ordered<CPackedBitVector,
                                             boost::bitwise<CPackedBitVector>>> {
    // clang-format on
public:
    using TBoolVec = std::vector<bool>;
    using TUInt8Vec = std::vector<std::uint8_t>;

    //! Operations which can be performed in the inner product.
    enum EOperation { E_AND, E_OR, E_XOR };

    //! \brief A forward iterator over the indices of the one bits in bit vector.
    class CORE_EXPORT COneBitIndexConstIterator
        : public std::iterator<std::input_iterator_tag, std::size_t, std::ptrdiff_t> {
    public:
        using TUInt8VecCItr = TUInt8Vec::const_iterator;

    public:
        COneBitIndexConstIterator() = default;
        COneBitIndexConstIterator(bool first, TUInt8VecCItr runLengthsItr, TUInt8VecCItr endRunLengthsItr);
        COneBitIndexConstIterator(std::size_t size, TUInt8VecCItr endRunLengthsItr);

        std::size_t operator*() const { return m_Current; }

        bool operator==(const COneBitIndexConstIterator& rhs) const {
            return m_Current == rhs.m_Current && m_RunLengthsItr == rhs.m_RunLengthsItr;
        }
        bool operator!=(const COneBitIndexConstIterator& rhs) const {
            return !(*this == rhs);
        }

        COneBitIndexConstIterator& operator++() {
            ++m_Current;
            if (m_Current == m_EndOfCurrentRun) {
                this->skipRun();
            }
            return *this;
        }

        COneBitIndexConstIterator operator++(int) {
            COneBitIndexConstIterator result(*this);
            this->operator++();
            return result;
        }

    private:
        void skipRun();
        std::size_t advanceToEndOfRun();

    private:
        std::size_t m_Current = 0;
        std::size_t m_EndOfCurrentRun = 0;
        TUInt8VecCItr m_RunLengthsItr;
        TUInt8VecCItr m_EndRunLengthsItr;
    };

public:
    //! The maximum permitted run length. Longer runs are encoded by stringing
    //! together a number of maximum length runs.
    static const std::uint8_t MAX_RUN_LENGTH;

public:
    CPackedBitVector();
    explicit CPackedBitVector(bool bit);
    CPackedBitVector(std::size_t dimension, bool bit);
    CPackedBitVector(const TBoolVec& bits);

    //! Contract the vector by popping a component from the start.
    void contract();

    //! Extend the vector to dimension adding the component \p n bits
    //! with value \p bit to the end.
    void extend(bool bit, std::size_t n = 1);

    //! \name Persistence
    //@{
    //! Create from delimited values.
    bool fromDelimited(const std::string& str);

    //! Persist state to delimited values.
    std::string toDelimited() const;
    //@}

    //! \name Container Semantics
    //@{
    //! Wraps dimension.
    std::size_t size() const { return this->dimension(); }

    //! Wraps operator().
    bool operator[](std::size_t i) const { return this->operator()(i); }

    //! Check if two collections are identically equal.
    bool operator==(const CPackedBitVector& other) const;

    //! Lexicographical total ordering.
    bool operator<(const CPackedBitVector& rhs) const;

    //! Get the complement, i.e. the collection whose bits are flipped.
    CPackedBitVector operator~() const;

    //! Update to the bitwise and of this and \p other.
    const CPackedBitVector& operator&=(const CPackedBitVector& other);

    //! Update to the bitwise or of this and \p other.
    const CPackedBitVector& operator|=(const CPackedBitVector& other);

    //! Update to the bitwise xor of this and \p other.
    const CPackedBitVector& operator^=(const CPackedBitVector& other);

    //! Get an iterator over the one bits of the collection.
    COneBitIndexConstIterator beginOneBits() const;

    //! Get an iterator for the end of the collection of one bits.
    COneBitIndexConstIterator endOneBits() const;
    //@}

    //! \name Vector Semantics
    //@{}
    //! Get the dimension.
    std::size_t dimension() const;

    //! Get the i'th component (no bounds checking).
    bool operator()(std::size_t i) const;

    //! Inner product.
    double inner(const CPackedBitVector& covector, EOperation op = E_AND) const;

    //! Euclidean norm.
    double euclidean() const { return std::sqrt(this->inner(*this)); }

    //! Manhattan norm.
    double manhattan() const {
        return this->size() == 0 ? 0 : this->inner(*this);
    }
    //@}

    //! Convert to a bit vector.
    TBoolVec toBitVector() const;

    //! Get a checksum of this vector's components.
    std::uint64_t checksum() const;

    //! Debug the memory used by this object.
    void debugMemoryUsage(CMemoryUsage::TMemoryUsagePtr mem) const;

    //! Get the memory used by this object.
    std::size_t memoryUsage() const;

private:
    void bitwise(EOperation op, const CPackedBitVector& other);
    template<typename RUN_ACTION>
    bool lineScan(const CPackedBitVector& covector, RUN_ACTION action) const;
    static void appendRun(std::size_t run, TUInt8Vec& runLengths);
    static void extendRun(std::size_t run, TUInt8Vec& runLengths);
    template<typename T>
    static T bit(EOperation op, T lhs, T rhs);

private:
    // Note that the bools are 1 byte aligned so the following three variables will
    // be packed into the 64 bits.

    //! The dimension of the vector.
    std::uint32_t m_Dimension;

    //! The value of the first component in the vector.
    bool m_First;

    //! The parity of the vector: true indicates that there are an even number runs
    //! and false that there are an odd. Together with m_First this determines the
    //! value of the last component.
    bool m_Parity;

    //! The length of each run. Note that if the length of a run exceeds 255 then
    //! this is encoded in multiple run lengths.
    TUInt8Vec m_RunLengths;
};

//! Output for debug.
CORE_EXPORT
std::ostream& operator<<(std::ostream& o, const CPackedBitVector& v);
}
}

#endif // INCLUDED_ml_core_CPackedBitVector_h
