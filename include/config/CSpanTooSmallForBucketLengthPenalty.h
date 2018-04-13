/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_config_CSpanTooSmallForBucketLengthPenalty_h
#define INCLUDED_ml_config_CSpanTooSmallForBucketLengthPenalty_h

#include <config/CPenalty.h>
#include <config/ImportExport.h>

namespace ml {
namespace config {

//! \brief Penalty for the case that the total data range is small w.r.t.
//! the candidate bucket length.
//!
//! DESCRIPTION:\n
//! If we only see a small number of buckets it is difficult to be confident
//! in that choice of bucket length. This penalizes bucket lengths which are
//! large w.r.t. the observed data span.
class CONFIG_EXPORT CSpanTooSmallForBucketLengthPenalty : public CPenalty {
public:
    CSpanTooSmallForBucketLengthPenalty(const CAutoconfigurerParams& params);

    //! Create a copy on the heap.
    virtual CSpanTooSmallForBucketLengthPenalty* clone() const;

    //! Get the name of this penalty.
    virtual std::string name() const;

private:
    //! Compute a penalty for rare detectors.
    virtual void penaltyFromMe(CDetectorSpecification& spec) const;
};
}
}

#endif // INCLUDED_ml_config_CSpanTooSmallForBucketLengthPenalty_h
