/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_config_CPolledDataPenalty_h
#define INCLUDED_ml_config_CPolledDataPenalty_h

#include <core/CoreTypes.h>

#include <config/CPenalty.h>
#include <config/ImportExport.h>

#include <boost/optional.hpp>

namespace ml {
namespace config {
class CAutoconfigurerParams;
class CDataCountStatistics;

//! \brief A penalty for using detectors which have a bucket length shorter
//! than the polling interval for regular data.
//!
//! DESCRIPTION:\n
//! If the data arrive at very uniform intervals then there is no point in
//! having bucket lengths less than the arrival interval. This tests this
//! condition and applies a decreasing penalty based on the number of intervals
//! for which this behavior has been observed.
class CONFIG_EXPORT CPolledDataPenalty : public CPenalty {
public:
    CPolledDataPenalty(const CAutoconfigurerParams& params);

    //! Create a copy on the heap.
    virtual CPolledDataPenalty* clone() const;

    //! Get the name of this penalty.
    virtual std::string name() const;

private:
    using TOptionalTime = boost::optional<core_t::TTime>;

private:
    //! Compute a penalty for rare detectors.
    virtual void penaltyFromMe(CDetectorSpecification& spec) const;

    //! Get the interval at which the data are polled if there is one.
    TOptionalTime pollingInterval(const CDataCountStatistics& stats) const;
};
}
}

#endif // INCLUDED_ml_config_CPolledDataPenalty_h
