/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CTimeUtils_h
#define INCLUDED_ml_core_CTimeUtils_h

#include <core/CNonInstantiatable.h>
#include <core/CoreTypes.h>
#include <core/ImportExport.h>

#include <boost/unordered_set.hpp>

#include <cstdint>
#include <string>

namespace ml {
namespace core {

//! \brief
//! A holder of time utility methods.
//!
//! DESCRIPTION:\n
//! A holder of time utility methods.  All methods are static; an object of
//! this class should never be constructed.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Class consoidates time methods used throughout the Ml
//! codebase.
//!
class CORE_EXPORT CTimeUtils : private CNonInstantiatable {
public:
    //! Maximum tolerable clock discrepancy between machines at the same
    //! customer site
    static const core_t::TTime MAX_CLOCK_DISCREPANCY;

    using TTimeBoolPr = std::pair<core_t::TTime, bool>;

public:
    //! Current time in seconds since the epoch
    static core_t::TTime now();

    //! Current time in milliseconds since the epoch
    static std::int64_t nowMs();

    //! Date and time to string according to http://www.w3.org/TR/NOTE-datetime
    //! E.g. 1997-07-16T19:20:30+01:00
    static std::string toIso8601(core_t::TTime t);

    //! Date and time to string according to local convention
    static std::string toLocalString(core_t::TTime t);

    //! Time only to string
    //! E.g. 19:20:30
    static std::string toTimeString(core_t::TTime t);

    //! Converts an epoch seconds timestamp to epoch millis
    static int64_t toEpochMs(core_t::TTime t);
    //! strptime interface
    //! NOTE: the time returned here is a UTC value
    static bool strptime(const std::string& format,
                         const std::string& dateTime,
                         core_t::TTime& preTime);

    //! Same strptime interface as above, but doesn't print any error messages
    static bool strptimeSilent(const std::string& format,
                               const std::string& dateTime,
                               core_t::TTime& preTime);

    //! Is a given word a day of the week name, month name, or timezone
    //! abbreviation in the current locale?  Input should be trimmed of
    //! whitespace before calling this function.
    static bool isDateWord(const std::string& word);

    //! Formats the given duration as human-readable string.
    static std::string durationToString(core_t::TTime duration);

    //! Convert a string representation of a time duration (in ES format e.g. "1000ms") to a whole number
    //! of seconds. Returns a default value if any error occurs, however the assumption is that the input string
    //! has already been validated by ES.
    static TTimeBoolPr timeDurationStringToSeconds(const std::string& timeDurationString,
                                                   core_t::TTime defaultValue);

private:
    //! Factor out common code from the three string conversion methods
    static void toStringCommon(core_t::TTime t, const std::string& format, std::string& result);

private:
    //! Class to cache date words so that we don't have to repeatedly use
    //! strptime() to check for them
    class CDateWordCache {
    public:
        //! Get the singleton instance
        static const CDateWordCache& instance();

        //! Check if a word is a date word
        bool isDateWord(const std::string& word) const;

    private:
        //! Constructor for a singleton is private
        CDateWordCache();
        ~CDateWordCache();

    private:
        //! This pointer is set after the singleton object has been constructed,
        //! and avoids the need to lock the magic static initialisation mutex on
        //! subsequent calls of the instance() method (once the updated value of
        //! this variable is visible in every thread).
        static CDateWordCache* ms_Instance;

        using TStrUSet = boost::unordered_set<std::string>;

        //! Our cache of date words
        TStrUSet m_DateWords;
    };
};
}
}

#endif // INCLUDED_ml_core_CTimeUtils_h
