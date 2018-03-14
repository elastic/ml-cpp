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

#include <config/CTools.h>

#include <core/CHashing.h>

#include <maths/CTools.h>

#include <boost/range.hpp>

#include <cctype>
#include <cstddef>
#include <math.h>

namespace ml {
namespace config {
namespace {
const core::CHashing::CMurmurHash2String HASHER;
const uint64_t                           LOWER_BITS = 0xffffffff;
const uint64_t                           UPPER_BITS = LOWER_BITS << 32;
}

uint32_t CTools::category32(const std::string &value) {
    return category32(HASHER(value));
}

uint32_t CTools::category32(std::size_t category64) {
    return static_cast<uint32_t>(((category64 & UPPER_BITS) >> 32) ^ (category64 & LOWER_BITS));
}

std::size_t CTools::category64(const std::string &value) {
    return HASHER(value);
}

double CTools::interpolate(double a, double b, double pa, double pb, double x) {
    return maths::CTools::truncate(pa + (pb - pa) * (x - a) / (b - a),
                                   std::min(pa, pb), std::max(pa, pb));
}

double CTools::powInterpolate(double p, double a, double b, double pa, double pb, double x) {
    return maths::CTools::truncate(pa + (pb - pa) * ::pow((x - a) / (b - a), p),
                                   std::min(pa, pb), std::max(pa, pb));
}

double CTools::logInterpolate(double a, double b, double pa, double pb, double x) {
    double la = maths::CTools::fastLog(a);
    double lb = maths::CTools::fastLog(b);
    double lx = maths::CTools::fastLog(x);
    return maths::CTools::truncate(pa + (pb - pa) * (lx - la) / (lb - la),
                                   std::min(pa, pb), std::max(pa, pb));
}

std::string CTools::prettyPrint(double d) {
    char buf[20];
    ::memset(buf, 0, sizeof(buf));

    if (::fabs(d) <= 1e-3) {
        std::sprintf(buf, "%.2e", d);
    } else if (::fabs(d) < 0.1) {
        std::sprintf(buf, "%.3f", d);
    } else if (::fabs(d) < 1.0) {
        std::sprintf(buf, "%.2f", d);
    } else if (::fabs(d) < 1e2) {
        std::sprintf(buf, "%.1f", d);
    } else if (::fabs(d) < 1e5) {
        std::sprintf(buf, "%.0f", d);
    } else if (::fabs(d) < 1e13) {
        std::sprintf(buf, "%.0f", d);
        char *end = std::find(buf, buf + 20, '\0');
        for (char *pos = end;
             pos - buf > 3 && std::isdigit(static_cast<unsigned char>(pos[-4]));
             pos -= 3, ++end) {
            std::copy_backward(pos - 3, end, end + 1);
            pos[-3] = ',';
        }
    } else {
        std::sprintf(buf, "%.2e", d);
    }

    return buf;
}

std::string CTools::prettyPrint(core_t::TTime time) {
    static const char *SUFFIXES[] = { " week", " day", " hr", " min", " sec" };

    std::string   result;
    core_t::TTime intervals[] =
    {
        (time / 604800),
        (time / 86400)  % 7,
        (time / 3600)   % 24,
        (time / 60)     % 60,
        time           % 60
    };
    for (std::size_t i = 0u; i < boost::size(intervals); ++i) {
        if (intervals[i] != 0) {
            result += (result.empty() ? "" : " ")
                      + core::CStringUtils::typeToString(intervals[i])
                      + SUFFIXES[i];
        }
    }
    return result;
}

}
}
