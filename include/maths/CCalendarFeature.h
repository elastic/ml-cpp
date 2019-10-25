/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDE_ml_maths_CCalendarFeature_h
#define INCLUDE_ml_maths_CCalendarFeature_h

#include <core/CoreTypes.h>

#include <maths/ImportExport.h>

#include <boost/operators.hpp>

#include <array>
#include <iosfwd>

#include <stdint.h>

namespace ml {
namespace maths {

//! \brief A light weight encoding of a calendar feature.
//!
//! IMPLEMENTATION:\n
//! Note that this purposely doesn't use an enum for encoding the feature
//! so that the member size is only 16 bits rather than sizeof(int).
class MATHS_EXPORT CCalendarFeature
    : boost::less_than_comparable<CCalendarFeature, boost::equality_comparable<CCalendarFeature>> {
public:
    //! See core::CMemory.
    static bool dynamicSizeAlwaysZero() { return true; }

    static const uint16_t DAYS_SINCE_START_OF_MONTH = 1;
    static const uint16_t DAYS_BEFORE_END_OF_MONTH = 2;
    static const uint16_t DAY_OF_WEEK_AND_WEEKS_SINCE_START_OF_MONTH = 3;
    static const uint16_t DAY_OF_WEEK_AND_WEEKS_BEFORE_END_OF_MONTH = 4;
    static const uint16_t BEGIN_FEATURES = 1;
    static const uint16_t END_FEATURES = 5;

    using TCalendarFeature4Ary = std::array<CCalendarFeature, 4>;

public:
    CCalendarFeature();
    CCalendarFeature(uint16_t feature, core_t::TTime time);

    //! Get all the features for \p time.
    static TCalendarFeature4Ary features(core_t::TTime time);

    //! Initialize with day of week, month and the month and year.
    void initialize(uint16_t feature, int dayOfWeek, int dayOfMonth, int month, int year);

    //! Initialize from \p value.
    bool fromDelimited(const std::string& value);

    //! Convert to a delimited string.
    std::string toDelimited() const;

    //! Check this and \p other for equality.
    bool operator==(CCalendarFeature rhs) const;

    //! Total ordering of two calendar features.
    bool operator<(CCalendarFeature rhs) const;

    //! \name Time Transforms
    //@{
    //! The offset of \p time w.r.t. the start of the current month's
    //! feature window.
    core_t::TTime offset(core_t::TTime time) const;

    //! Check if \p time is in this feature's window.
    bool inWindow(core_t::TTime time) const;
    //@}

    //! Get this feature's window.
    core_t::TTime window() const;

    //! Get a checksum for this object.
    uint64_t checksum(uint64_t seed = 0) const;

    //! Get a debug description of the feature.
    std::string print() const;

private:
    //! An invalid feature value.
    static const uint16_t INVALID;

private:
    //! The feature.
    uint16_t m_Feature;
    //! The feature value.
    uint16_t m_Value;
};

MATHS_EXPORT
std::ostream& operator<<(std::ostream& strm, const CCalendarFeature& feature);
}
}

#endif // INCLUDE_ml_maths_CCalendarFeature_h
