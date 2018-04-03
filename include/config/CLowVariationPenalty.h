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
    virtual CLowVariationPenalty* clone(void) const;

    //! Get the name of this penalty.
    virtual std::string name(void) const;

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
