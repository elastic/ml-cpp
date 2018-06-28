/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CStrFTime.h>

#include <core/CTimezone.h>
#include <core/CoreTypes.h>

#include <iomanip>
#include <sstream>
#include <string>

#include <errno.h>
#include <stdlib.h>

namespace ml {
namespace core {

// Work around the fact that Windows strftime() treats %z differently to Unix
size_t CStrFTime::strFTime(char* buf, size_t maxSize, const char* format, struct tm* tm) {
    if (buf == 0 || format == 0 || tm == 0) {
        errno = EINVAL;
        return 0;
    }

    std::string adjFormat(format);

    size_t zPos(adjFormat.find("%z"));
    if (zPos != std::string::npos) {
        // The approach is to replace the %z with a literal
        core_t::TTime localTm(CTimezone::instance().localToUtc(*tm));
        core_t::TTime gmTm(::_mkgmtime(tm));
        core_t::TTime diffSeconds(gmTm - localTm);
        core_t::TTime diffMinutes(diffSeconds / 60);
        core_t::TTime diffHours(diffMinutes / 60);

        std::ostringstream strm;
        strm << ((diffSeconds < 0) ? '-' : '+') << std::setfill('0')
             << std::setw(2) << ::_abs64(diffHours) << std::setfill('0')
             << std::setw(2) << (::_abs64(diffMinutes) % 60);

        adjFormat.replace(zPos, 2, strm.str());
    }

    zPos = adjFormat.find("%Z");
    if (zPos != std::string::npos) {
        CTimezone& tz = CTimezone::instance();

        // +ve means in DST; -ve means unknown
        if (tm->tm_isdst > 0) {
            adjFormat.replace(zPos, 2, tz.dstAbbrev());
        } else {
            adjFormat.replace(zPos, 2, tz.stdAbbrev());
        }
    }

    return ::strftime(buf, maxSize, adjFormat.c_str(), tm);
}
}
}
