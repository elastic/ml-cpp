/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CBjkstUniqueValues_h
#define INCLUDED_ml_maths_CBjkstUniqueValues_h

#include <core/CHashing.h>
#include <core/CMemory.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>

#include <maths/ImportExport.h>

#include <boost/variant.hpp>

#include <cstddef>
#include <utility>
#include <vector>

#include <stdint.h>

namespace ml {
namespace maths {

//! \brief The BJSKT algorithm for estimating the number of unique values
//! in a collection.
//!
//! DESCRIPTION:\n
//! This implements (more or less) the BJKST algorithm for estimating
//! the number of unique events in a large collection in much reduced
//! space. The implementation isn't the most compact and trades some
//! space for improved performance and ease of implementation.
//!
//! It will estimate the number of unique items in a population with
//! the a space and accuracy tradeoff. In particular,
//! <pre class="fragment">
//!   \f$\displaystyle P\left(\left|\frac{e}{n} - 1\right| > \epsilon\right) < \delta\f$
//! </pre>
//!
//! where, \f$e\f$ is the estimate of the number of unique items, \f$n\f$
//! is the true number of unique items and \f$\epsilon\f$ and \f$\delta\f$
//! are constants such that the algorithm requires space
//! <pre class="fragment">
//!   \f$\displaystyle O\left(\left|\frac{1}{\epsilon^2}\log(\epsilon)log(\delta)\right|\right)\f$
//! </pre>
//!
//! IMPLEMENTATION DECISIONS:\n
//! This implementation is reasonably space efficient but doesn't pack
//! the hash map into a bit array for simplicity and better runtime
//! constants. Also, since it uses an array of uint8_t to store the
//! hash map I've chosen to restrict the map hash function to 16 bits
//! rather than the minimum value needed to get reasonable collision
//! probability. This should be plenty big enough for all sizes of
//! hash map we'd use in practice.
//!
//! The maximum total space used in bytes is:
//! <pre class="fragment">
//!   \f$\displaystyle (44 + 3m)k\f$
//! </pre>
//!
//! Here, \f$k\f$ is the \p numberHashes parameter and \f$m\f$ is the
//! \p maxSize parameter supplied to the constructor. Roughly speaking
//! \f$m = 1/\epsilon^2\f$ and \f$k = -\log(\delta)\f$.
//!
//! Note that the hash map lookup constants are good but the complexity
//! is bad \f$O(m)\f$ so the \p maxSize parameter supplied to the
//! constructor should be less than a few hundred.
class MATHS_EXPORT CBjkstUniqueValues {
public:
    using TUInt32HashVec = core::CHashing::CUniversalHash::TUInt32UnrestrictedHashVec;

public:
    //! Get the count of trailing zeros in value.
    static uint8_t trailingZeros(uint32_t value);

public:
    //! \param numberHashes The number of independent hashes.
    //! \param maxSize The maximum size of the hash sets.
    CBjkstUniqueValues(std::size_t numberHashes, std::size_t maxSize);

    //! Create by traversing a state document.
    CBjkstUniqueValues(core::CStateRestoreTraverser& traverser);

    //! Efficiently swap the contents of two sketches.
    void swap(CBjkstUniqueValues& other);

private:
    //! Create by traversing a state document.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

public:
    //! Convert to a node tree.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Add a new value.
    void add(uint32_t value);

    //! Remove a value.
    void remove(uint32_t value);

    //! Get an estimate of the number of unique values added.
    uint32_t number() const;

    //! Get a checksum for the sketch.
    uint64_t checksum(uint64_t seed = 0) const;

    //! Get the memory used by this sketch.
    void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

    //! Get the memory used by this sketch.
    std::size_t memoryUsage() const;

private:
    using TUInt8Vec = std::vector<uint8_t>;
    using TUInt8VecVec = std::vector<TUInt8Vec>;
    using TUInt32Vec = std::vector<uint32_t>;
    using TUInt32VecItr = TUInt32Vec::iterator;
    using TUInt32VecCItr = TUInt32Vec::const_iterator;

    //! Wraps up the sketch data.
    struct MATHS_EXPORT SSketch {
        SSketch();
        SSketch(std::size_t numberHashes);

        //! Efficiently swap the contents of two sketches.
        void swap(SSketch& other);

        //! Create by traversing a state document.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser, std::size_t numberHashes);

        //! Convert to a node tree.
        void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

        //! Add a new value.
        void add(std::size_t maxSize, uint32_t value);

        //! Remove a value.
        void remove(uint32_t value);

        //! Get an estimate of the number of unique values added.
        uint32_t number() const;

        //! The secondary hash function.
        TUInt32HashVec s_G;
        //! The main hash functions.
        TUInt32HashVec s_H;
        //! The trailing zero counts.
        TUInt8Vec s_Z;
        //! The unique hashed values.
        TUInt8VecVec s_B;
    };

    using TUInt32VecOrSketch = boost::variant<TUInt32Vec, SSketch>;

private:
    //! Maybe switch to sketching the distinct value set.
    void sketch();

private:
    //! The maximum size of the sketch set before compression.
    std::size_t m_MaxSize;
    //! The number of distinct hashes to use in the sketch.
    std::size_t m_NumberHashes;
    //! The distinct count sketch.
    TUInt32VecOrSketch m_Sketch;
};
}
}

#endif // INCLUDED_ml_maths_CBjkstUniqueValues_h
