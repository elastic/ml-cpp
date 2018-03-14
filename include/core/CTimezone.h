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
#ifndef INCLUDED_ml_core_CTimezone_h
#define INCLUDED_ml_core_CTimezone_h

#include <core/CFastMutex.h>
#include <core/CNonCopyable.h>
#include <core/CoreTypes.h>
#include <core/ImportExport.h>

#include <boost/date_time/local_time/local_time.hpp>

#include <string>

#include <time.h>


namespace ml {
namespace core {


//! \brief
//! Portability wrapper to set the current time zone
//!
//! DESCRIPTION:\n
//! Portability wrapper to set the current time zone.  The Windows C runtime
//! library is not good at handling timezones other than the one that
//! Windows itself is set to, hence the need to wrap timezone related C
//! library calls.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Implemented as a Meyers singleton.
//!
//! On Unix, we use the standard timezone definitions in /usr/share/zoneinfo.
//! This is all handled automatically by the C runtime library.
//!
//! On Windows, calls that need timezone information, such as strftime(),
//! strptime() and localtime() are wrapped and instead use Boost to
//! translate UTC to local times.  This means that on Windows we're reliant
//! on a CSV file that comes with Boost to contain the correct timezone
//! information for every timezone that a customer might use.  (The Visual
//! C++ runtime library's ability to switch timezones is unbelivably poor -
//! daylight saving start/end dates are hardcoded to those in use in the USA
//! between 1986 and 2007 if you switch away from the operating system
//! timezone!)
//!
//! The reason we don't use Boost on all platforms is that Boost's knowledge
//! of daylight saving rules is inferior to that of Unix.  Boost only has one
//! set of rules per timezone, whereas Unix knows that, for example, the
//! daylight saving rules in the USA changed in 2007, so knows that if we're
//! parsing a file from 2006 then different rules apply.
//!
//! As a result, our implementation is imperfect for parsing historical data
//! on Windows.  Problems resulting from this are likely to be rare though.
//!
class CORE_EXPORT CTimezone : private CNonCopyable {
    public:
        //! Get the singleton instance
        static CTimezone &instance(void);

        //! Get the name of the current timezone.  This will be a POSIX name,
        //! e.g. Europe/London or America/New_York, or, if the timezone has not
        //! been changed from the system default, an empty string.
        const std::string &timezoneName(void) const;

        //! Set the name of the timezone used by the C library functions (or our
        //! replacements for them)
        //! Example input: America/New_York
        //!                Europe/London
        //! A blank string will cause the timezone of the machine
        //! we're running on to be used (which the C library will determine
        //! in an OS dependent manner).
        bool timezoneName(const std::string &name);

        //! Convenience wrapper around the setter for timezone name
        static bool setTimezone(const std::string &timezone);

        //! Abbreviation for standard time in the current timezone
        std::string stdAbbrev(void) const;

        //! Abbreviation for daylight saving time in the current timezone
        std::string dstAbbrev(void) const;

        //! Normalise a local time structure and also return the corresponding
        //! epoch time (i.e. seconds since midnight on 1/1/1970 UTC).  This
        //! is a replacement for mktime() that switches to using Boost on
        //! Windows when the program's timezone is different to the operating
        //! system's timezone.
        core_t::TTime localToUtc(struct tm &localTime) const;

        //! Convert a UTC time to local time in the current timezone.
        bool utcToLocal(core_t::TTime utcTime, struct tm &localTime) const;

        //! Get the date fields.
        bool dateFields(core_t::TTime utcTime,
                        int &daysSinceSunday,
                        int &dayOfMonth,
                        int &daysSinceJanuary1st,
                        int &monthsSinceJanuary,
                        int &yearsSince1900,
                        int &secondsSinceMidnight) const;

    private:
        //! Constructor for a singleton is private.
        CTimezone(void);
        ~CTimezone(void);

    private:
        //! Since there is one timezone for the whole program, access to it is
        //! protected by this mutex
        mutable CFastMutex               m_Mutex;

        //! Name of the current timezone in use within this program, or blank to
        //! use the current operating system settings
        std::string                      m_Name;

#ifdef Windows
        //! Boost timezone database
        boost::local_time::tz_database   m_TimezoneDb;

        //! Boost timezone database
        boost::local_time::time_zone_ptr m_Timezone;
#endif
};


}
}

#endif // INCLUDED_ml_core_CTimezone_h

