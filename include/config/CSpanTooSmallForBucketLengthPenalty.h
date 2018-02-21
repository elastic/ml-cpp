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

#ifndef INCLUDED_ml_config_CSpanTooSmallForBucketLengthPenalty_h
#define INCLUDED_ml_config_CSpanTooSmallForBucketLengthPenalty_h

#include <config/CPenalty.h>
#include <config/ImportExport.h>

namespace ml
{
namespace config
{

//! \brief Penalty for the case that the total data range is small w.r.t.
//! the candidate bucket length.
//!
//! DESCRIPTION:\n
//! If we only see a small number of buckets it is difficult to be confident
//! in that choice of bucket length. This penalizes bucket lengths which are
//! large w.r.t. the observed data span.
class CONFIG_EXPORT CSpanTooSmallForBucketLengthPenalty : public CPenalty
{
    public:
        CSpanTooSmallForBucketLengthPenalty(const CAutoconfigurerParams &params);

        //! Create a copy on the heap.
        virtual CSpanTooSmallForBucketLengthPenalty *clone(void) const;

        //! Get the name of this penalty.
        virtual std::string name(void) const;

    private:
        //! Compute a penalty for rare detectors.
        virtual void penaltyFromMe(CDetectorSpecification &spec) const;
};

}
}

#endif // INCLUDED_ml_config_CSpanTooSmallForBucketLengthPenalty_h
