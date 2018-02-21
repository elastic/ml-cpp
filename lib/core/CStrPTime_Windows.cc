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
#include <core/CStrPTime.h>

#include <core/CStringUtils.h>
#include <core/CTimezone.h>

#include <string>

#include <ctype.h>
#include <string.h>

// We don't have a header for this on Windows, so declare it here
extern "C"
char *strptime(const char *buf, const char *fmt, struct tm *tm);


namespace ml
{
namespace core
{


// Our Windows strptime() implementation supports the %z and %Z time formats.
// However, on Windows struct tm doesn't have any members for GMT offset that
// we could set, so strptime() doesn't make any changes to the output based
// on these flags.
//
// Therefore, this small extension to strptime() allows it to work with %z
// when the %z is the last non-whitespace in the format specifier.
// Realistically (or at least hopefully), the %z will always come at the end
// of a date/time.
//
// Also, since strptime() uses the C runtime globals _tzname[0] and _tzname[1],
// whereas we might want to use a different timezone, we replace %Z in the
// format string with a string obtained from the CTimezone singleton.
char *CStrPTime::strPTime(const char *buf,
                          const char *format,
                          struct tm *tm)
{
    // If any of the inputs are NULL then do nothing
    if (buf == 0 || format == 0 || tm == 0)
    {
        return 0;
    }

    std::string adjFormat(format);

    // Replace %Z first if present
    size_t tznamePos(adjFormat.find("%Z"));
    if (tznamePos != std::string::npos)
    {
        // Find the corresponding place in the buffer
        char *excess(CStrPTime::strPTime(buf,
                                         adjFormat.substr(0, tznamePos).c_str(),
                                         tm));
        if (excess == 0)
        {
            return 0;
        }

        // Skip leading whitespace
        while (::isspace(static_cast<unsigned char>(*excess)))
        {
            ++excess;
        }

        // Only GMT and the standard and daylight saving timezone names for the
        // current timezone are supported, as per the strptime() man page
        std::string possTzName(excess);
        if (possTzName.find("GMT") == 0)
        {
            adjFormat.replace(tznamePos, 2, "GMT");
        }
        else
        {
            CTimezone &tz = CTimezone::instance();
            std::string stdAbbrev(tz.stdAbbrev());
            if (possTzName.find(stdAbbrev) == 0)
            {
                adjFormat.replace(tznamePos, 2, stdAbbrev);
            }
            else
            {
                std::string dstAbbrev(tz.dstAbbrev());
                if (possTzName.find(dstAbbrev) == 0)
                {
                    adjFormat.replace(tznamePos, 2, dstAbbrev);
                }
                else
                {
                    return 0;
                }
            }
        }
    }

    // Check if the format specifier includes a %z
    size_t zPos(adjFormat.find("%z"));
    if (zPos != std::string::npos)
    {
        // If there's anything except whitespace after the
        // %z it's too complicated
        if (adjFormat.find_first_not_of(CStringUtils::WHITESPACE_CHARS, zPos + 2) != std::string::npos)
        {
            return 0;
        }

        adjFormat.erase(zPos);
    }

    char *excess(::strptime(buf, adjFormat.c_str(), tm));

    // We only have more work to do if %z was in the string, and
    // the basic strptime() call worked
    if (excess != 0 && zPos != std::string::npos)
    {
        // Skip leading whitespace
        while (::isspace(static_cast<unsigned char>(*excess)))
        {
            ++excess;
        }

        // We expect something along the lines of +0000 or
        // -0500, i.e. a plus or minus sign followed by 4 digits
        core_t::TTime sign(0);
        if (*excess == '+')
        {
            sign = 1;
        }
        else if (*excess == '-')
        {
            sign = -1;
        }
        else
        {
            return 0;
        }

        ++excess;

        core_t::TTime hour(0);
        if (*excess >= '0' && *excess <= '2')
        {
            hour += (*excess - '0') * 10;
        }
        else
        {
            return 0;
        }

        ++excess;

        if (*excess >= '0' && *excess <= '9')
        {
            hour += (*excess - '0');
        }
        else
        {
            return 0;
        }

        ++excess;

        core_t::TTime minute(0);
        if (*excess >= '0' && *excess <= '5')
        {
            minute += (*excess - '0') * 10;
        }
        else
        {
            return 0;
        }

        ++excess;

        if (*excess >= '0' && *excess <= '9')
        {
            minute += (*excess - '0');
        }
        else
        {
            return 0;
        }

        ++excess;

        // Now we know how many minutes and hours ahead or behind GMT the
        // time we just parsed was explicitly specified to be, so convert
        // the struct tm on the basis that it's GMT and then subtract the
        // explicit adjustment.
        core_t::TTime utcTime(::_mkgmtime(tm));
        utcTime -= sign * minute * 60;
        utcTime -= sign * hour * 60 * 60;

        CTimezone &tz = CTimezone::instance();
        if (tz.utcToLocal(utcTime, *tm) == false)
        {
            return 0;
        }
    }

    return excess;
}


}
}

