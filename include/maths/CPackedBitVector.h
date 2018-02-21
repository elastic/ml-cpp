/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CPackedBitVector_h
#define INCLUDED_ml_maths_CPackedBitVector_h

#include <core/CMemoryUsage.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>

#include <maths/ImportExport.h>

#include <boost/operators.hpp>

#include <cstddef>
#include <string>
#include <vector>

#include <math.h>
#include <stdint.h>


namespace ml
{
namespace maths
{

//! \brief A compact representation of binary vector.
//!
//! DESCRIPTION:\n
//! This implements a run length encoding of a binary vector (string
//! of 0's and 1's). It supports efficient inner products using a line
//! scan algorithm. The same algorithm supports bitwise or and exclusive
//! or, which are selected by supplying an option predicate.
//!
//! IMPLEMENTATION:\n
//! The space optimal vector depends on the average run length. In
//! particular, it is optimum to use around log2(E[run length]) bits
//! to encode each run. We expect relative short runs in our target
//! applications so stick with uint8_t to encode the run length.
//!
//! Because there are only two values we need only store the value of
//! the first bit in the vector and can deduce all other values by the
//! number of runs in between. In practice we store one extra bit, the
//! vector parity to allow us to extend the vector efficiently.
class MATHS_EXPORT CPackedBitVector : private boost::equality_comparable< CPackedBitVector,
                                              boost::partially_ordered< CPackedBitVector > >
{
    public:
        typedef std::vector<bool> TBoolVec;

        //! Operations which can be performed in the inner product.
        enum EOperation
        {
            E_AND,
            E_OR,
            E_XOR
        };

    public:
        CPackedBitVector(void);
        explicit CPackedBitVector(bool bit);
        CPackedBitVector(std::size_t dimension, bool bit);
        CPackedBitVector(const TBoolVec &bits);

        //! Contract the vector by popping a component from the start.
        void contract(void);

        //! Extend the vector to dimension adding the component \p bit.
        void extend(bool bit);

        //! \name Persistence
        //@{
        //! Create from delimited values.
        bool fromDelimited(const std::string &str);

        //! Persist state to delimited values.
        std::string toDelimited(void) const;
        //@}

        //! Get the dimension.
        std::size_t dimension(void) const;

        //! Get the i'th component (no bounds checking).
        bool operator()(std::size_t i) const;

        //! Check if two vectors are identically equal.
        bool operator==(const CPackedBitVector &other) const;

        //! Lexicographical total ordering.
        bool operator<(const CPackedBitVector &rhs) const;

        //! Get the complement vector, i.e. the vector whose bits are negated.
        CPackedBitVector complement(void) const;

        //! Inner product.
        double inner(const CPackedBitVector &covector,
                     EOperation op = E_AND) const;

        //! Euclidean norm.
        double euclidean(void) const
        {
            return ::sqrt(this->inner(*this));
        }

        //! Manhattan norm.
        double manhattan(void) const
        {
            return this->inner(*this);
        }

        //! Convert to a bit vector.
        TBoolVec toBitVector(void) const;

        //! Get a checksum of this vector's components.
        uint64_t checksum(void) const;

        //! Debug the memory used by this object.
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this object.
        std::size_t memoryUsage(void) const;

    private:
        typedef std::vector<uint8_t> TUInt8Vec;

    private:
        //! The maximum permitted run length. Longer runs are encoded
        //! by stringing together a number of maximum length runs.
        static const uint8_t MAX_RUN_LENGTH;

    private:
        // Note that the bools are 1 byte aligned so the following
        // three variables will be packed into the 64 bits.

        //! The dimension of the vector.
        uint32_t m_Dimension;

        //! The value of the first component in the vector.
        bool m_First;

        //! The parity of the vector: true indicates that there are an
        //! even number runs and false that there are an odd. Together
        //! with m_First this determines the value of the last component.
        bool m_Parity;

        //! The length of each run. Note that if the length of a run
        //! exceeds 255 then this is encoded in multiple run lengths.
        TUInt8Vec m_RunLengths;
};

//! Output for debug.
MATHS_EXPORT
std::ostream &operator<<(std::ostream &o, const CPackedBitVector &v);

}
}

#endif // INCLUDED_ml_maths_CPackedBitVector_h
