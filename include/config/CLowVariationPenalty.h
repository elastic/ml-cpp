/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_config_CLowVariationPenalty_h
#define INCLUDED_ml_config_CLowVariationPenalty_h

#include <config/CPenalty.h>

#include <config/ImportExport.h>

namespace ml {
namespace config {
class CPartitionDataCountStatistics;
class CByAndPartitionDataCountStatistics;
class CByOverAndPartitionDataCountStatistics;

//! \brief A penalty for function values which have very low coefficient of
//! variation and are unlikely to produce anomalies.
//!
//! DESCRIPTION:\n
//! In certain cases, the coefficient of variation of data is very low and anomaly
//! detection will not find anything interesting. Examples include, the case that
//! every category of a distinct count function argument appears in every bucket,
//! say because it labels a monitor, and the case that polled data is analyzed for
//! count anomalies.
class CONFIG_EXPORT CLowVariationPenalty : public CPenalty {
public:
    CLowVariationPenalty(const CAutoconfigurerParams& params);

    //! Create a copy on the heap.
    virtual CLowVariationPenalty* clone() const;

    //! Get the name of this penalty.
    virtual std::string name() const;

private:
    //! Apply a penalty for features with very little variation.
    virtual void penaltyFromMe(CDetectorSpecification& spec) const;

    //! Apply the penalty for count with optionally a partition.
    void penaltiesForCount(const CPartitionDataCountStatistics& stats, CDetectorSpecification& spec) const;

    //! Apply the penalty for count with a by field and optionally a partition.
    void penaltiesForCount(const CByAndPartitionDataCountStatistics& stats, CDetectorSpecification& spec) const;

    //! Apply the penalty for count with a by, over and optionally a partition field.
    void penaltiesForCount(const CByOverAndPartitionDataCountStatistics& stats, CDetectorSpecification& spec) const;

    //! Apply the penalty for distinct count with optionally a partition.
    void penaltyForDistinctCount(const CPartitionDataCountStatistics& stats, CDetectorSpecification& spec) const;

    //! Apply the penalty for distinct count with by and optionally a partition.
    void penaltyForDistinctCount(const CByAndPartitionDataCountStatistics& stats, CDetectorSpecification& spec) const;

    //! Apply the penalty for distinct count with by, over and optionally a partition.
    void penaltyForDistinctCount(const CByOverAndPartitionDataCountStatistics& stats, CDetectorSpecification& spec) const;

    //! Apply the penalty for info content with optionally a partition.
    void penaltyForInfoContent(const CPartitionDataCountStatistics& stats, CDetectorSpecification& spec) const;

    //! Apply the penalty for info content with a by field and optionally a partition.
    void penaltyForInfoContent(const CByAndPartitionDataCountStatistics& stats, CDetectorSpecification& spec) const;

    //! Apply the penalty for info content with a by, over and optionally a partition field.
    void penaltyForInfoContent(const CByOverAndPartitionDataCountStatistics& stats, CDetectorSpecification& spec) const;
};
}
}

#endif // INCLUDED_ml_config_CLowVariationPenalty_h
