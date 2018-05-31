/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CSeasonalTime_h
#define INCLUDED_ml_maths_CSeasonalTime_h

#include <core/CoreTypes.h>

#include <maths/ImportExport.h>

#include <memory>
#include <string>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {

//! \brief Provides times for seasonal components of a time series
//! decomposition.
class MATHS_EXPORT CSeasonalTime {
public:
    using TTimeTimePr = std::pair<core_t::TTime, core_t::TTime>;

public:
    CSeasonalTime();
    CSeasonalTime(core_t::TTime period, double precedence);
    virtual ~CSeasonalTime() = default;

    //! A total order on seasonal times.
    bool operator<(const CSeasonalTime& rhs) const;

    //! Get a copy of this time.
    //!
    //! \warning The caller owns the result.
    virtual CSeasonalTime* clone() const = 0;

    //! Initialize from a string created by persist.
    virtual bool fromString(const std::string& value) = 0;

    //! Convert to a string.
    virtual std::string toString() const = 0;

    //! \name Time Transforms
    //@{
    //! Extract the time of \p time in the current period.
    double periodic(core_t::TTime time) const;

    //! Extract the time of \p time in the current regression.
    double regression(core_t::TTime time) const;

    //! Get the interval between in regression time units
    //! between \p start and \p end.
    double regressionInterval(core_t::TTime start, core_t::TTime end) const;

    //! Get the start of the repeat containing \p time.
    core_t::TTime startOfWindowRepeat(core_t::TTime time) const;

    //! Get the start of the window containing \p time.
    core_t::TTime startOfWindow(core_t::TTime time) const;

    //! Check if \p time is in the window.
    bool inWindow(core_t::TTime time) const;
    //@}

    //! \name Period
    //@{
    //! Get the period.
    core_t::TTime period() const;

    //! Set the period.
    void period(core_t::TTime period);
    //@}

    //! \name Regression
    //@{
    //! Get the origin of the time coordinates.
    core_t::TTime regressionOrigin() const;

    //! Set the origin of the time coordinates.
    void regressionOrigin(core_t::TTime origin);
    //@}

    //! \name Window
    //@{
    //! Get the repeat interval for the window pattern.
    virtual core_t::TTime windowRepeat() const = 0;

    //! Get the start of the window pattern.
    virtual core_t::TTime windowRepeatStart() const = 0;

    //! Get the start of the window.
    virtual core_t::TTime windowStart() const = 0;

    //! Get the end of the window.
    virtual core_t::TTime windowEnd() const = 0;

    //! Get the window.
    TTimeTimePr window() const;

    //! Get the window length.
    core_t::TTime windowLength() const;

    //! Check if this applies to a time window.
    bool windowed() const;

    //! Get the fraction of time which overlaps the window.
    double fractionInWindow() const;
    //@}

    //! Check whether this time's seasonal component time excludes
    //! modeling \p other's.
    bool excludes(const CSeasonalTime& other) const;

    //! True if this has a weekend and false otherwise.
    virtual bool hasWeekend() const = 0;

    //! Get a checksum for this object.
    virtual uint64_t checksum(uint64_t seed = 0) const = 0;

private:
    //! Get the start of the repeat interval beginning at
    //! \p offset including \p time.
    //!
    //! For diurnal time this is the start of the week containing
    //! \p time for other times it is the start of the period
    //! containing \p time.
    core_t::TTime startOfWindowRepeat(core_t::TTime offset, core_t::TTime time) const;

    //! Get the scale to apply when computing the regression time.
    virtual core_t::TTime regressionTimeScale() const = 0;

private:
    //! The periodic repeat.
    core_t::TTime m_Period;
    //! The origin of the time coordinates used to maintain
    //! a reasonably conditioned Gramian of the design matrix.
    core_t::TTime m_RegressionOrigin;
    //! The precedence of the corresponding component when
    //! deciding which to keep amongst alternatives.
    double m_Precedence;
};

//! \brief Provides times for daily and weekly period seasonal
//! components of a time series decomposition.
class MATHS_EXPORT CDiurnalTime : public CSeasonalTime {
public:
    CDiurnalTime();
    CDiurnalTime(core_t::TTime startOfWeek,
                 core_t::TTime windowStart,
                 core_t::TTime windowEnd,
                 core_t::TTime period,
                 double precedence = 1.0);

    //! Get a copy of this time.
    CDiurnalTime* clone() const;

    //! Initialize from a string created by persist.
    virtual bool fromString(const std::string& value);

    //! Convert to a string.
    virtual std::string toString() const;

    //! Get the length of a week.
    virtual core_t::TTime windowRepeat() const;

    //! Get the start of the week.
    virtual core_t::TTime windowRepeatStart() const;

    //! Get the start of the window.
    virtual core_t::TTime windowStart() const;

    //! Get the end of the window.
    virtual core_t::TTime windowEnd() const;

    //! True if this has a weekend and false otherwise.
    virtual bool hasWeekend() const;

    //! Get a checksum for this object.
    virtual uint64_t checksum(uint64_t seed = 0) const;

private:
    //! Get the scale to apply when computing the regression time.
    virtual core_t::TTime regressionTimeScale() const;

private:
    //! The start of the week.
    core_t::TTime m_StartOfWeek;
    //! The start of the window.
    core_t::TTime m_WindowStart;
    //! The end of the window.
    core_t::TTime m_WindowEnd;
};

//! \brief Provides times for arbitrary period seasonal components
//! of a time series decomposition.
class MATHS_EXPORT CGeneralPeriodTime : public CSeasonalTime {
public:
    CGeneralPeriodTime() = default;
    CGeneralPeriodTime(core_t::TTime period, double precedence = 1.0);

    //! Get a copy of this time.
    CGeneralPeriodTime* clone() const;

    //! Initialize from a string created by persist.
    virtual bool fromString(const std::string& value);

    //! Convert to a string.
    virtual std::string toString() const;

    //! Return the period.
    virtual core_t::TTime windowRepeat() const;

    //! Returns zero.
    virtual core_t::TTime windowRepeatStart() const;

    //! Returns zero.
    virtual core_t::TTime windowStart() const;

    //! Returns the period.
    virtual core_t::TTime windowEnd() const;

    //! Returns false.
    virtual bool hasWeekend() const;

    //! Get a checksum for this object.
    virtual uint64_t checksum(uint64_t seed = 0) const;

private:
    //! Get the scale to apply when computing the regression time.
    virtual core_t::TTime regressionTimeScale() const;
};

//! \brief Convert CSeasonalTime sub-classes to/from text representations.
//!
//! DESCRIPTION:\n
//! Encapsulate the conversion of arbitrary CSeasonalTime sub-classes to/from
//! textual state. In particular, the field name associated with each type of
//! CSeasonalTime is then in one file.
class MATHS_EXPORT CSeasonalTimeStateSerializer {
public:
    //! Shared pointer to the CTimeSeriesDecompositionInterface abstract
    //! base class.
    using TSeasonalTimePtr = std::shared_ptr<CSeasonalTime>;

public:
    //! Construct the appropriate CSeasonalTime sub-class from its state
    //! document representation. Sets \p result to NULL on failure.
    static bool acceptRestoreTraverser(TSeasonalTimePtr& result,
                                       core::CStateRestoreTraverser& traverser);

    //! Persist state by passing information to \p inserter.
    static void acceptPersistInserter(const CSeasonalTime& time,
                                      core::CStatePersistInserter& inserter);
};
}
}

#endif // INCLUDED_ml_maths_CSeasonalTime_h
