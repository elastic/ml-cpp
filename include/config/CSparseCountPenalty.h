/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_config_CSparseCountPenalty_h
#define INCLUDED_ml_config_CSparseCountPenalty_h

#include <config/CPenalty.h>
#include <config/ImportExport.h>

namespace ml {
namespace config {

//! \brief Penalty for the case that counting functions are viewed at a
//! bucket length which is too short relative the data rate.
//!
//! DESCRIPTION:\n
//! If the bucket length is small compared to the mean arrival time then
//! buckets will typically either contain zero or one values and so will
//! not properly capture the variation in arrival times. This penalizes
//! bucket lengths which are less than the shortest bucket length which
//! captures the count distribution at longer bucket lengths.
class CONFIG_EXPORT CSparseCountPenalty : public CPenalty {
public:
    CSparseCountPenalty(const CAutoconfigurerParams& params);

    //! Create a copy on the heap.
    virtual CSparseCountPenalty* clone() const;

    //! Get the name of this penalty.
    virtual std::string name() const;

private:
    //! Compute a penalty for rare detectors.
    virtual void penaltyFromMe(CDetectorSpecification& spec) const;
};
}
}

#endif // INCLUDED_ml_config_CSparseCountPenalty_h
