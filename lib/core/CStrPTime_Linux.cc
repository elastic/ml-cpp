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

#include <string>

#include <ctype.h>
#include <string.h>

namespace ml {
namespace core {

// On Linux strptime() accepts %z, but doesn't make any changes
// to the output based on it!
//
// Therefore, this small extension to strptime() allows it
// to work with %z when the %z is the last non-whitespace in
// the format specifier.  Realistically (or at least hopefully),
// the %z will always come at the end of a date/time.
//
// Also, strptime() on Linux is supposed to skip over a timezone
// name indicated by the %Z format, but (at least on Fedora 9) it
// doesn't.  So Linux requires special handling for %Z too.
// The way this is done is to replace %Z in the format string
// with one of "GMT", tzname[0] or tzname[1] if they are present
// at the appropriate point in the buffer.
//
// (Interestingly, strptime() works fine on Mac OS X.)
char* CStrPTime::strPTime(const char* buf, const char* format, struct tm* tm) {
    // If any of the inputs are NULL then do nothing
    if (buf == 0 || format == 0 || tm == 0) {
        return 0;
    }

    std::string adjFormat(format);

    // Replace %Z first if present
    size_t tznamePos(adjFormat.find("%Z"));
    if (tznamePos != std::string::npos) {
        // Find the corresponding place in the buffer
        char* excess(CStrPTime::strPTime(buf, adjFormat.substr(0, tznamePos).c_str(), tm));
        if (excess == 0) {
            return 0;
        }

        // Skip leading whitespace
        while (::isspace(static_cast<unsigned char>(*excess))) {
            ++excess;
        }

        // Only GMT and the standard and daylight saving timezone names for the
        // current timezone are supported, as per the strptime() man page
        std::string possTzName(excess);
        if (possTzName.find("GMT") == 0) {
            adjFormat.replace(tznamePos, 2, "GMT");
        } else if (possTzName.find(::tzname[0]) == 0) {
            adjFormat.replace(tznamePos, 2, ::tzname[0]);
        } else if (possTzName.find(::tzname[1]) == 0) {
            adjFormat.replace(tznamePos, 2, ::tzname[1]);
        } else {
            return 0;
        }
    }

    // Check if the format specifier includes a %z
    size_t zPos(adjFormat.find("%z"));
    if (zPos != std::string::npos) {
        // If there's anything except whitespace after the
        // %z it's too complicated
        if (adjFormat.find_first_not_of(CStringUtils::WHITESPACE_CHARS, zPos + 2) != std::string::npos) {
            return 0;
        }

        adjFormat.erase(zPos);
    }

    char* excess(::strptime(buf, adjFormat.c_str(), tm));

    // We only have more work to do if %z was in the string, and
    // the basic strptime() call worked
    if (excess != 0 && zPos != std::string::npos) {
        // Skip leading whitespace
        while (::isspace(static_cast<unsigned char>(*excess))) {
            ++excess;
        }

        // We expect something along the lines of +0000 or
        // -0500, i.e. a plus or minus sign followed by 4 digits
        int sign(0);
        if (*excess == '+') {
            sign = 1;
        } else if (*excess == '-') {
            sign = -1;
        } else {
            return 0;
        }

        ++excess;

        int hour(0);
        if (*excess >= '0' && *excess <= '2') {
            hour += (*excess - '0') * 10;
        } else {
            return 0;
        }

        ++excess;

        if (*excess >= '0' && *excess <= '9') {
            hour += (*excess - '0');
        } else {
            return 0;
        }

        ++excess;

        int minute(0);
        if (*excess >= '0' && *excess <= '5') {
            minute += (*excess - '0') * 10;
        } else {
            return 0;
        }

        ++excess;

        if (*excess >= '0' && *excess <= '9') {
            minute += (*excess - '0');
        } else {
            return 0;
        }

        ++excess;

        // Now we know how many minutes and hours ahead or behind GMT
        // the time we just parsed was explicitly specified to be.
        // Next we need to find out the offset that Linux is currently
        // assuming, and frig the struct tm to account for any difference.
        // For example, suppose the timezone is US Eastern, but the %z just
        // parsed as +0000, i.e. GMT.  Assuming the time is in winter, we'll
        // find that local time is 18000 seconds behind GMT, so we'll need
        // to add 18000 seconds to the values held in the struct tm to
        // compensate.

        // On Linux, the global timezone variable declared in time.h
        // holds the current timezone's offset from GMT in seconds.
        int secondAdj(static_cast<int>(::timezone) % 60);
        int minuteAdj(((static_cast<int>(::timezone) / 60) % 60) + (sign * minute));
        int hourAdj((static_cast<int>(::timezone) / 3600) + (sign * hour));

        // This may leave the hours, minutes and seconds outside their
        // normal range.  However, luckily part of the functionality of
        // mktime() is to normalise them, so we can just call that.
        tm->tm_sec -= secondAdj;
        tm->tm_min -= minuteAdj;
        tm->tm_hour -= hourAdj;

        ::mktime(tm);
    }

    return excess;
}
}
}
