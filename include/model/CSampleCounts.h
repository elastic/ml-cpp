/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CSampleCounts_h
#define INCLUDED_ml_model_CSampleCounts_h

#include <core/CMemory.h>
#include <core/CoreTypes.h>

#include <maths/CBasicStatistics.h>

#include <model/ImportExport.h>

#include <string>
#include <vector>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace model {
class CDataGatherer;

//! \brief Manages setting of sample counts.
//!
//! DESCRIPTION:\n
//! To handle variable data rates we gather statistics on an almost
//! fixed number of measurements (since the distribution of these
//! statistics is a function of the measurement count). The correction
//! we apply to account for the varying measurement count in the
//! likelihood function is approximate and so if the mean bucket
//! bucket count wanders too far from the sample count we reset the
//! the sample count.
class MODEL_EXPORT CSampleCounts {
public:
    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;
    using TSizeVec = std::vector<std::size_t>;

public:
    explicit CSampleCounts(unsigned int sampleCountOverride = 0);

    //! Create a copy that will result in the same persisted state as the
    //! original.  This is effectively a copy constructor that creates a
    //! copy that's only valid for a single purpose.  The boolean flag is
    //! redundant except to create a signature that will not be mistaken for
    //! a general purpose copy constructor.
    CSampleCounts(bool isForPersistence, const CSampleCounts& other);

    CSampleCounts* cloneForPersistence() const;

    //! Persist the sample counts to a state document.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Restore some sample counts from a state document traverser.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Get the sample count identified by \p id.
    unsigned int count(std::size_t id) const;

    //! Get the effective sample count identified by \p id.
    double effectiveSampleCount(std::size_t id) const;

    //! Reset the sample count identified by \p id.
    void resetSampleCount(const CDataGatherer& gatherer, std::size_t id);

    //! Update the effective sample variances to reflect new sample for \p id.
    void updateSampleVariance(std::size_t id);

    //! Update the mean non-zero bucket counts and age the count data.
    void updateMeanNonZeroBucketCount(std::size_t id, double count, double alpha);

    //! Refresh the sample count identified by \p id.
    void refresh(const CDataGatherer& gatherer);

    //! Recycle the sample counts identified by \p idsToRemove.
    void recycle(const TSizeVec& idsToRemove);

    //! Remove all traces of attributes whose identifiers are
    //! greater than or equal to \p lowestIdToRemove.
    void remove(std::size_t lowestIdToRemove);

    //! Resize the sample counts so they can accommodate \p id.
    void resize(std::size_t id);

    //! Get the sample counts checksum.
    uint64_t checksum(const CDataGatherer& gatherer) const;

    //! Debug the memory used by this object.
    void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

    //! Get the memory used by this object.
    std::size_t memoryUsage() const;

    //! Clear the sample counts.
    void clear();

private:
    using TUIntVec = std::vector<unsigned int>;

private:
    //! Get the name of the entity identified by \p id.
    const std::string& name(const CDataGatherer& gatherer, std::size_t id) const;

private:
    //! This overrides the sample counts if non-zero.
    unsigned int m_SampleCountOverride;

    //! The "fixed" number of measurements in a sample.
    TUIntVec m_SampleCounts;

    //! The mean number of measurements per bucket.
    TMeanAccumulatorVec m_MeanNonZeroBucketCounts;

    //! The effective sample variance in the data supplied to the
    //! model. The sample count is reset if the mean bucket count
    //! moves too far from the current value. This is an approximate
    //! estimate of the effective variance, due to the averaging
    //! process, of the samples with which the model has been updated.
    TMeanAccumulatorVec m_EffectiveSampleVariances;
};

} // model
} // ml

#endif // INCLUDED_ml_model_CSampleCounts_h
