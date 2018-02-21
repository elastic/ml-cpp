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

#ifndef INCLUDED_ml_maths_CSeasonalTime_h
#define INCLUDED_ml_maths_CSeasonalTime_h

#include <core/CoreTypes.h>

#include <maths/ImportExport.h>

#include <boost/shared_ptr.hpp>

#include <string>

namespace ml
{
namespace core
{
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths
{

//! \brief Provides times for seasonal components of a time series
//! decomposition.
class MATHS_EXPORT CSeasonalTime
{
    public:
        typedef std::pair<core_t::TTime, core_t::TTime> TTimeTimePr;

    public:
        CSeasonalTime(void);
        CSeasonalTime(core_t::TTime period);
        virtual ~CSeasonalTime(void);

        //! Get a copy of this time.
        //!
        //! \warning The caller owns the result.
        virtual CSeasonalTime *clone(void) const = 0;

        //! Initialize from a string created by persist.
        virtual bool fromString(const std::string &value) = 0;

        //! Convert to a string.
        virtual std::string toString(void) const = 0;

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
        core_t::TTime period(void) const;

        //! Set the period.
        void period(core_t::TTime period);
        //@}

        //! \name Regression
        //@{
        //! Get the origin of the time coordinates.
        core_t::TTime regressionOrigin(void) const;

        //! Set the origin of the time coordinates.
        void regressionOrigin(core_t::TTime origin);
        //@}

        //! \name Window
        //@{
        //! Get the repeat interval for the window pattern.
        virtual core_t::TTime windowRepeat(void) const = 0;

        //! Get the start of the window pattern.
        virtual core_t::TTime windowRepeatStart(void) const = 0;

        //! Get the start of the window.
        virtual core_t::TTime windowStart(void) const = 0;

        //! Get the end of the window.
        virtual core_t::TTime windowEnd(void) const = 0;

        //! Get the window.
        TTimeTimePr window(void) const;

        //! Get the window length.
        core_t::TTime windowLength(void) const;

        //! Check if this applies to a time window.
        bool windowed(void) const;

        //! Get the fraction of time which overlaps the window.
        double fractionInWindow(void) const;
        //@}

        //! Get the decay rate scaled from \p fromPeriod period to
        //! \p toPeriod period.
        //!
        //! \param[in] decayRate The decay rate for \p fromPeriod.
        //! \param[in] fromPeriod The period of \p decayRate.
        //! \param[in] toPeriod The desired period decay rate.
        static double scaleDecayRate(double decayRate,
                                     core_t::TTime fromPeriod,
                                     core_t::TTime toPeriod);

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
        virtual core_t::TTime regressionTimeScale(void) const = 0;

    private:
        //! The periodic repeat.
        core_t::TTime m_Period;
        //! The origin of the time coordinates used to maintain
        //! a reasonably conditioned Gramian of the design matrix.
        core_t::TTime m_RegressionOrigin;
};

//! \brief Provides times for daily and weekly period seasonal
//! components of a time series decomposition.
class MATHS_EXPORT CDiurnalTime : public CSeasonalTime
{
    public:
        CDiurnalTime(void);
        CDiurnalTime(core_t::TTime startOfWeek,
                     core_t::TTime windowStart,
                     core_t::TTime windowEnd,
                     core_t::TTime period);

        //! Get a copy of this time.
        CDiurnalTime *clone(void) const;

        //! Initialize from a string created by persist.
        virtual bool fromString(const std::string &value);

        //! Convert to a string.
        virtual std::string toString(void) const;

        //! Get the length of a week.
        virtual core_t::TTime windowRepeat(void) const;

        //! Get the start of the week.
        virtual core_t::TTime windowRepeatStart(void) const;

        //! Get the start of the window.
        virtual core_t::TTime windowStart(void) const;

        //! Get the end of the window.
        virtual core_t::TTime windowEnd(void) const;

        //! Get a checksum for this object.
        virtual uint64_t checksum(uint64_t seed = 0) const;

    private:
        //! Get the scale to apply when computing the regression time.
        virtual core_t::TTime regressionTimeScale(void) const;

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
class MATHS_EXPORT CGeneralPeriodTime : public CSeasonalTime
{
    public:
        CGeneralPeriodTime(void);
        CGeneralPeriodTime(core_t::TTime period);

        //! Get a copy of this time.
        CGeneralPeriodTime *clone(void) const;

        //! Initialize from a string created by persist.
        virtual bool fromString(const std::string &value);

        //! Convert to a string.
        virtual std::string toString(void) const;

        //! Return the period.
        virtual core_t::TTime windowRepeat(void) const;

        //! Returns zero.
        virtual core_t::TTime windowRepeatStart(void) const;

        //! Returns zero.
        virtual core_t::TTime windowStart(void) const;

        //! Returns the period.
        virtual core_t::TTime windowEnd(void) const;

        //! Get a checksum for this object.
        virtual uint64_t checksum(uint64_t seed = 0) const;

    private:
        //! Get the scale to apply when computing the regression time.
        virtual core_t::TTime regressionTimeScale(void) const;
};

//! \brief Convert CSeasonalTime sub-classes to/from text representations.
//!
//! DESCRIPTION:\n
//! Encapsulate the conversion of arbitrary CSeasonalTime sub-classes to/from
//! textual state. In particular, the field name associated with each type of
//! CSeasonalTime is then in one file.
class MATHS_EXPORT CSeasonalTimeStateSerializer
{
    public:
        //! Shared pointer to the CTimeSeriesDecompositionInterface abstract
        //! base class.
        typedef boost::shared_ptr<CSeasonalTime> TSeasonalTimePtr;

    public:
        //! Construct the appropriate CSeasonalTime sub-class from its state
        //! document representation. Sets \p result to NULL on failure.
        static bool acceptRestoreTraverser(TSeasonalTimePtr &result,
                                           core::CStateRestoreTraverser &traverser);

        //! Persist state by passing information to \p inserter.
        static void acceptPersistInserter(const CSeasonalTime &time,
                                          core::CStatePersistInserter &inserter);

};

}
}

#endif // INCLUDED_ml_maths_CSeasonalTime_h
