/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CCooccurrences_h
#define INCLUDED_ml_maths_CCooccurrences_h

#include <core/CMemoryUsage.h>

#include <maths/CPackedBitVector.h>
#include <maths/ImportExport.h>

#include <boost/unordered_set.hpp>

#include <utility>
#include <vector>

namespace ml
{
namespace core
{
class CMemory;
}
namespace maths
{

//! \brief Computes various measures for the co-occurrence of events,
//! as encoded by an indicator variable, and finds events with significantly
//! evidence that they are co-occuring.
//!
//! DESCRIPTION:\n
class MATHS_EXPORT CCooccurrences
{
    public:
        using TDoubleVec = std::vector<double>;
        using TSizeVec = std::vector<std::size_t>;
        using TSizeSizePr = std::pair<std::size_t, std::size_t>;
        using TSizeSizePrVec = std::vector<TSizeSizePr>;

    public:
        CCooccurrences(std::size_t maximumLength, std::size_t indicatorWidth);

        //! Create from part of a state document.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

        //! Persist state by passing to the supplied inserter.
        void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

        //! Get the top \p n most co-occurring events by significance
        //! with events \p X.
        void topNBySignificance(std::size_t X,
                                std::size_t n,
                                TSizeSizePrVec &top,
                                TDoubleVec &significances) const;

        //! Get the top \p n most co-occurring events by significance.
        void topNBySignificance(std::size_t n,
                                TSizeSizePrVec &top,
                                TDoubleVec &significances) const;

        //! Resize the relevant statistics to accommodate up to \p event
        //! streams.
        void addEventStreams(std::size_t n);

        //! Remove the event streams \p remove.
        void removeEventStreams(const TSizeVec &remove);

        //! Recycle the event streams \p recycle.
        void recycleEventStreams(const TSizeVec &recycle);

        //! Add the value \p x for the variable \p X.
        void add(std::size_t X);

        //! Capture the indicator values of missing events.
        void capture();

        //! Get the checksum of this object.
        uint64_t checksum(uint64_t seed = 0) const;

        //! Debug the memory used by this object.
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this object.
        std::size_t memoryUsage() const;

    private:
        using TSizeUSet = boost::unordered_set<std::size_t>;
        using TPackedBitVectorVec = std::vector<CPackedBitVector>;

    private:
        //! The maximum permitted event sequence length.
        std::size_t m_MaximumLength;

        //! The current length of the event sequences.
        std::size_t m_Length;

        //! The width of the indicator function in "captures".
        std::size_t m_IndicatorWidth;

        //! The current offset from the start of the current indicator.
        std::size_t m_Offset;

        //! The event indicators to add in the next capture.
        TSizeUSet m_CurrentIndicators;

        //! The indicator variables for event streams.
        TPackedBitVectorVec m_Indicators;
};

}
}

#endif // INCLUDED_ml_maths_CCooccurrences_h
