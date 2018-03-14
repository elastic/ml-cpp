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
#ifndef INCLUDED_ml_core_CTimeUtils_h
#define INCLUDED_ml_core_CTimeUtils_h

#include <core/CFastMutex.h>
#include <core/CNonInstantiatable.h>
#include <core/CoreTypes.h>
#include <core/ImportExport.h>

#include <boost/unordered_set.hpp>

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

    public:
        //! Current time
        static core_t::TTime  now(void);

        //! Date and time to string according to http://www.w3.org/TR/NOTE-datetime
        //! E.g. 1997-07-16T19:20:30+01:00
        static std::string    toIso8601(core_t::TTime t);

        //! Date and time to string according to local convention
        static std::string    toLocalString(core_t::TTime t);

        //! Time only to string
        //! E.g. 19:20:30
        static std::string    toTimeString(core_t::TTime t);

        //! strptime interface
        //! NOTE: the time returned here is a UTC value
        static bool strptime(const std::string &format,
                             const std::string &dateTime,
                             core_t::TTime &preTime);

        //! Same strptime interface as above, but doesn't print any error messages
        static bool strptimeSilent(const std::string &format,
                                   const std::string &dateTime,
                                   core_t::TTime &preTime);

        //! Is a given word a day of the week name, month name, or timezone
        //! abbreviation in the current locale?  Input should be trimmed of
        //! whitespace before calling this function.
        static bool isDateWord(const std::string &word);

    private:
        //! Factor out common code from the three string conversion methods
        static void toStringCommon(core_t::TTime t,
                                   const std::string &format,
                                   std::string &result);

    private:
        //! Class to cache date words so that we don't have to repeatedly use
        //! strptime() to check for them
        class CDateWordCache {
            public:
                //! Get the singleton instance
                static const CDateWordCache &instance(void);

                //! Check if a word is a date word
                bool isDateWord(const std::string &word) const;

            private:
                //! Constructor for a singleton is private
                CDateWordCache(void);
                ~CDateWordCache(void);

            private:
                //! Protect the singleton's initialisation, preventing it from
                //! being constructed simultaneously in different threads.
                static CFastMutex ms_InitMutex;

                //! This pointer is set after the singleton object has been
                //! constructed, and avoids the need to lock the mutex on
                //! subsequent calls of the instance() method (once the updated
                //! value of this variable has made its way into every thread).
                static volatile CDateWordCache *ms_Instance;

                typedef boost::unordered_set<std::string> TStrUSet;

                //! Our cache of date words
                TStrUSet                       m_DateWords;
        };
};


}
}

#endif // INCLUDED_ml_core_CTimeUtils_h

