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
    virtual CPolledDataPenalty* clone(void) const;

    //! Get the name of this penalty.
    virtual std::string name(void) const;

private:
    typedef boost::optional<core_t::TTime> TOptionalTime;

private:
    //! Compute a penalty for rare detectors.
    virtual void penaltyFromMe(CDetectorSpecification& spec) const;

    //! Get the interval at which the data are polled if there is one.
    TOptionalTime pollingInterval(const CDataCountStatistics& stats) const;
};
}
}

#endif // INCLUDED_ml_config_CPolledDataPenalty_h
