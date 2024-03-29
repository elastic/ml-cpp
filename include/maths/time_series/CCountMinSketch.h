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

#ifndef INCLUDED_ml_maths_time_series_CCountMinSketch_h
#define INCLUDED_ml_maths_time_series_CCountMinSketch_h

#include <core/CHashing.h>
#include <core/CMemoryUsage.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>

#include <maths/time_series/ImportExport.h>

#include <maths/common/MathsTypes.h>

#include <variant>

namespace ml {
namespace maths {
namespace time_series {

//! \brief Implements Count-Min Sketch approximate counting of
//! categories.
//!
//! DESCRIPTION:\n
//! This implements approximate counts of distinct categories
//! categories in fixed space.
//!
//! For more details see:
//! http://dimacs.rutgers.edu/~graham/pubs/papers/cm-full.pdf.
//!
//! \note That the error on the count is proportional to the total
//! count, i.e. for a given epsilon \f$\epsilon\f$ the data structure
//! estimate bounds the error on a category count to less than
//! \f$n_i + epsilon n\f$, where \f$n_i\f$ is the true category
//! count and \f$n\f$ is the count of all values added.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Counts are stored as real numbers so that they can be aged out.
//!
//! This class uses float storage for the counts. This is because
//! it is intended for use in cases where space is at a premium.
//! *DO NOT* use floats unless doing so gives a significant overall
//! space improvement to the *program* footprint. Note also that the
//! interface to this class is double precision. If floats are used
//! they should be used for storage only and transparent to the rest
//! of the code base.
class MATHS_TIME_SERIES_EXPORT CCountMinSketch {
public:
    CCountMinSketch(std::size_t rows, std::size_t columns);

    //! Create by traversing a state document.
    explicit CCountMinSketch(core::CStateRestoreTraverser& traverser);

    //! Efficient swap the contents of two sketches.
    void swap(CCountMinSketch& sketch) noexcept;

private:
    //! Create by traversing a state document.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

public:
    //! Convert to a node tree.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Get the number of rows.
    std::size_t rows() const;

    //! Get the number of columns.
    std::size_t columns() const;

    //! Get the \f$\delta\f$ probability.
    double delta() const;

    //! Get the \f$P(1-\delta)\f$ error.
    double oneMinusDeltaError() const;

    //! Add a count of \p count for category \p category.
    //!
    //! \note \p count can be negative in which case the count is
    //! removed from the sketch.
    void add(std::uint32_t category, double count);

    //! Remove \p category from the sketch altogether.
    //!
    //! \note That one can decrement the counts by calling add with
    //! a negative count. However, if we have not sketched the counts
    //! this removes the map entry for \p category.
    void removeFromMap(std::uint32_t category);

    //! Age the counts forwards \p time.
    void age(double alpha);

    //! Get the total count of all categories.
    double totalCount() const;

    //! Get the count of category \p category.
    double count(std::uint32_t category) const;

    //! Get the fraction of category \p category.
    double fraction(std::uint32_t category) const;

    //! Check if the counts are sketched.
    bool sketched() const;

    //! Get a checksum for the sketch.
    std::uint64_t checksum(std::uint64_t seed = 0) const;

    //! Get the memory used by this sketch.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const;

    //! Get the memory used by this sketch.
    std::size_t memoryUsage() const;

private:
    using TUInt32HashVec = core::CHashing::CUniversalHash::TUInt32UnrestrictedHashVec;
    using TFloatVec = std::vector<common::CFloatStorage>;
    using TFloatVecVec = std::vector<TFloatVec>;

    //! Wraps up the sketch data.
    struct SSketch {
        SSketch() = default;
        SSketch(std::size_t rows, std::size_t columns);

        //! Create by traversing a state document.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser,
                                    std::size_t rows,
                                    std::size_t columns);

        //! Convert to a node tree.
        void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

        //! The hash functions.
        TUInt32HashVec s_Hashes;

        //! The counts.
        TFloatVecVec s_Counts;
    };

    using TUInt32FloatPr = std::pair<std::uint32_t, common::CFloatStorage>;
    using TUInt32FloatPrVec = std::vector<TUInt32FloatPr>;
    using TUInt32FloatPrVecOrSketch = std::variant<TUInt32FloatPrVec, SSketch>;

    //! Maybe switch to sketching the counts.
    void sketch();

private:
    //! The number of rows.
    std::size_t m_Rows;

    //! The number of columns.
    std::size_t m_Columns;

    //! The total count.
    common::CFloatStorage m_TotalCount;

    //! The sketch.
    TUInt32FloatPrVecOrSketch m_Sketch;
};
}
}
}

#endif // INCLUDED_ml_maths_time_series_CCountMinSketch_h
