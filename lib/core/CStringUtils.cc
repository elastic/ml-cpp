/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CStringUtils.h>

#include <core/CLogger.h>
#include <core/CStrCaseCmp.h>

#include <boost/multi_array.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace {
//! In order to avoid a failure on read we need to account for the rounding
//! performed by strtod which can result in the rounded value being outside
//! the representable range for values near minimum and maximum double.
double clampToReadable(double x) {
    static const double SMALLEST = -(1.0 - 5e-16) * std::numeric_limits<double>::max();
    static const double LARGEST = (1.0 - 5e-16) * std::numeric_limits<double>::max();
    return (x < SMALLEST ? SMALLEST : x > LARGEST ? LARGEST : x);
}

// To ensure the singleton locale is constructed before multiple threads may
// require it, call locale() during the static initialisation phase of the
// program.  Of course, the locale may already be constructed before this if
// another static object has used it.
const std::locale& DO_NOT_USE_THIS_VARIABLE = ml::core::CStringUtils::locale();
}

namespace ml {
namespace core {

// Initialise static class members
const std::string CStringUtils::WHITESPACE_CHARS(" \t\r\n\v\f");

int CStringUtils::utf8ByteType(char c) {
    unsigned char u = static_cast<unsigned char>(c);

    if ((u & 0x80) == 0) {
        // Single byte character
        return 1;
    }
    if ((u & 0xC0) == 0x80) {
        // Continuation character
        return -1;
    }
    if ((u & 0xE0) == 0xC0) {
        // Start of two byte character
        return 2;
    }
    if ((u & 0xF0) == 0xE0) {
        // Start of three byte character
        return 3;
    }
    if ((u & 0xF8) == 0xF0) {
        // Start of four byte character
        return 4;
    }
    if ((u & 0xFC) == 0xF8) {
        // Start of five byte character
        return 5;
    }

    return 6;
}

std::string CStringUtils::toLower(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), &::tolower);
    return str;
}

std::string CStringUtils::toUpper(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), &::toupper);
    return str;
}

size_t CStringUtils::numMatches(const std::string& str, const std::string& word) {
    size_t count(0);
    std::string::size_type pos(0);

    while (pos != std::string::npos) {
        pos = str.find(word, pos);
        if (pos != std::string::npos) {
            ++count;

            // start next search after this word
            pos += word.length();
        }
    }

    return count;
}

void CStringUtils::trimWhitespace(std::string &str) {
    CStringUtils::trim(WHITESPACE_CHARS, str);
}

void CStringUtils::trim(const std::string& toTrim, std::string& str) {
    if (toTrim.empty() || str.empty()) {
        return;
    }

    std::string::size_type pos = str.find_last_not_of(toTrim);
    if (pos == std::string::npos) {
        // Special case - entire string is being trimmed
        str.clear();
        return;
    }

    str.erase(pos + 1);

    pos = str.find_first_not_of(toTrim);
    if (pos != std::string::npos && pos > 0) {
        str.erase(0, pos);
    }
}

std::string CStringUtils::normaliseWhitespace(const std::string& str) {
    std::string result;
    result.reserve(str.length());

    bool outputSpace(true);
    for (std::string::const_iterator iter = str.begin(); iter != str.end(); ++iter) {
        char current(*iter);
        if (::isspace(static_cast<unsigned char>(current))) {
            if (outputSpace) {
                outputSpace = false;
                result += ' ';
            }
        } else {
            outputSpace = true;
            result += current;
        }
    }

    return result;
}

size_t CStringUtils::replace(const std::string& from, const std::string& to, std::string& str) {
    if (from == to) {
        return 0;
    }

    size_t count(0);
    std::string::size_type pos(0);
    while (pos != std::string::npos) {
        pos = str.find(from, pos);

        if (pos == std::string::npos) {
            return count;
        }

        str.replace(pos, from.size(), to);
        ++count;

        pos += to.size();
    }

    return count;
}

size_t CStringUtils::replaceFirst(const std::string& from,
                                  const std::string& to,
                                  std::string& str) {
    if (from == to) {
        return 0;
    }

    std::string::size_type pos = str.find(from);
    if (pos == std::string::npos) {
        return 0;
    }

    str.replace(pos, from.size(), to);

    return 1;
}

void CStringUtils::escape(char escape, const std::string& toEscape, std::string& str) {
    if (escape == '\0' || toEscape.empty()) {
        return;
    }

    std::string::size_type pos(0);
    while (pos < str.length()) {
        pos = str.find_first_of(toEscape, pos);
        if (pos == std::string::npos) {
            break;
        }

        str.insert(pos, 1, escape);

        // Skip the next 2 characters as we don't want to process the same one
        // twice and we've just inserted one
        pos += 2;
    }
}

void CStringUtils::unEscape(char escape, std::string& str) {
    if (escape == '\0') {
        return;
    }

    std::string::size_type pos(0);
    while (pos < str.length()) {
        pos = str.find(escape, pos);
        if (pos == std::string::npos) {
            break;
        } else if (pos + 1 == str.length()) {
            LOG_WARN(<< "String to be unescaped ends with escape character: " << str);
        }

        str.erase(pos, 1);

        // Skip the next character so that an escaped escape character
        // is converted to a single escape character
        ++pos;
    }
}

std::string CStringUtils::_typeToString(const unsigned long long& i) {
    char buf[4 * sizeof(unsigned long long)];
    ::memset(buf, 0, sizeof(buf));

    ::sprintf(buf, "%llu", i);

    return buf;
}

std::string CStringUtils::_typeToString(const unsigned long& i) {
    char buf[4 * sizeof(unsigned long)];
    ::memset(buf, 0, sizeof(buf));

    ::sprintf(buf, "%lu", i);

    return buf;
}

std::string CStringUtils::_typeToString(const unsigned int& i) {
    char buf[4 * sizeof(unsigned int)];
    ::memset(buf, 0, sizeof(buf));

    ::sprintf(buf, "%u", i);

    return buf;
}

std::string CStringUtils::_typeToString(const unsigned short& i) {
    char buf[4 * sizeof(unsigned short)];
    ::memset(buf, 0, sizeof(buf));

    ::sprintf(buf, "%hu", i);

    return buf;
}

std::string CStringUtils::_typeToString(const long long& i) {
    char buf[4 * sizeof(long long)];
    ::memset(buf, 0, sizeof(buf));

    ::sprintf(buf, "%lld", i);

    return buf;
}

std::string CStringUtils::_typeToString(const long& i) {
    char buf[4 * sizeof(long)];
    ::memset(buf, 0, sizeof(buf));

    ::sprintf(buf, "%ld", i);

    return buf;
}

std::string CStringUtils::_typeToString(const int& i) {
    char buf[4 * sizeof(int)];
    ::memset(buf, 0, sizeof(buf));

    ::sprintf(buf, "%d", i);

    return buf;
}

std::string CStringUtils::_typeToString(const short& i) {
    char buf[4 * sizeof(short)];
    ::memset(buf, 0, sizeof(buf));

    ::sprintf(buf, "%hd", i);

    return buf;
}

std::string CStringUtils::_typeToString(const bool& b) {
    return (b ? "true" : "false");
}

std::string CStringUtils::_typeToString(const double& i) {
    // Note the extra large buffer here, which is because the format string is
    // "%f" rather than "%g", which means we could be printing a 308 digit
    // number without resorting to scientific notation
    char buf[64 * sizeof(double)];
    ::memset(buf, 0, sizeof(buf));

    ::sprintf(buf, "%f", i);

    return buf;
}

std::string CStringUtils::_typeToString(const char* str) {
    return str;
}

std::string CStringUtils::_typeToString(const char& c) {
    return std::string(1, c);
}

// This may seem silly, but it allows generic code to be written
const std::string& CStringUtils::_typeToString(const std::string& str) {
    return str;
}

std::string CStringUtils::typeToStringPretty(double d) {
    // Maximum size =   1  (for sign)
    //                + 7  (for # s.f.)
    //                + 1  (for decimal point)
    //                + 5  (for w.c. e+308)
    //                + 1  (for terminating character)
    //              =  16

    char buf[16];
    ::memset(buf, 0, sizeof(buf));
    ::sprintf(buf, "%.7g", d);
    return buf;
}

std::string CStringUtils::typeToStringPrecise(double d, CIEEE754::EPrecision precision) {
    // Just use a large enough buffer to hold maximum precision.
    char buf[4 * sizeof(double)];
    ::memset(buf, 0, sizeof(buf));

    // To retain the correct number of significant figures this uses
    // scientific notation format when d < 1. Note that the number
    // specifier in e format is number of points after the decimal
    // place so number of significant figures + 1. Note also that
    // when printing to limited precision we must correctly round
    // the value before printing because printing just truncates;
    // for example,
    //   sprintf(buf, "%.6e", 0.49999998)
    // gives 4.999999e-1 rather than the correctly rounded value 0.5.

    int ret = 0;
    switch (precision) {
    case CIEEE754::E_HalfPrecision:
        ret = std::fabs(d) < 1.0 && d != 0.0
                  ? ::sprintf(buf, "%.2e",
                              clampToReadable(CIEEE754::round(d, CIEEE754::E_HalfPrecision)))
                  : ::sprintf(buf, "%.3g",
                              clampToReadable(CIEEE754::round(d, CIEEE754::E_HalfPrecision)));
        break;

    case CIEEE754::E_SinglePrecision:
        ret = std::fabs(d) < 1.0 && d != 0.0
                  ? ::sprintf(buf, "%.6e",
                              clampToReadable(CIEEE754::round(d, CIEEE754::E_SinglePrecision)))
                  : ::sprintf(buf, "%.7g",
                              clampToReadable(CIEEE754::round(d, CIEEE754::E_SinglePrecision)));
        break;

    case CIEEE754::E_DoublePrecision:
        ret = std::fabs(d) < 1.0 && d != 0.0
                  ? ::sprintf(buf, "%.14e", clampToReadable(d))
                  : ::sprintf(buf, "%.15g", clampToReadable(d));
        break;
    }

    // Workaround for Visual C++ 2010 misformatting - replace
    // 123.45e010 with 123.45e10 and 123.45e-010 with 123.45e-10.
    // Also it is inefficient to output trailing zeros, i.e.
    // 1.23456000000000e-11 so we strip these off in the following.
    if (ret > 2) {
        // Look for an 'e'
        char* ptr(static_cast<char*>(::memchr(buf, 'e', ret - 1)));
        if (ptr != nullptr) {
            bool edit = false;
            bool minus = false;

            // Strip off any trailing zeros and a trailing point.
            char* bwd = ptr;
            for (;;) {
                --bwd;
                if (*bwd == '0' || *bwd == '.') {
                    edit = true;
                } else {
                    break;
                }
            }

            // Strip off any leading zeros in the exponent.
            char* fwd = ptr;
            for (;;) {
                ++fwd;
                if (*fwd == '-') {
                    minus = true;
                } else if (*fwd == '+' || *fwd == '0') {
                    edit = true;
                } else {
                    break;
                }
            }

            if (edit) {
                std::string adjResult;
                adjResult.reserve(ret - 1);
                // mantissa
                adjResult.assign(buf, bwd + 1);
                if (::isdigit(static_cast<unsigned char>(*fwd))) {
                    adjResult.append(minus ? "e-" : "e");
                    // exponent
                    adjResult.append(fwd);
                }
                return adjResult;
            }
        }
    }

    return buf;
}

bool CStringUtils::_stringToType(bool silent, const std::string& str, unsigned long long& i) {
    if (str.empty()) {
        if (!silent) {
            LOG_ERROR(<< "Unable to convert empty string to unsigned long long");
        }
        return false;
    }

    char* endPtr(nullptr);
    errno = 0;
    unsigned long long ret(::strtoull(str.c_str(), &endPtr, 0));

    if (ret == 0 && errno == EINVAL) {
        if (!silent) {
            LOG_ERROR(<< "Unable to convert string '" << str
                      << "'"
                         " to unsigned long long: "
                      << ::strerror(errno));
        }
        return false;
    }

    if (ret == ULLONG_MAX && errno == ERANGE) // note ULLONG_MAX used for compatability with strtoull
    {
        if (!silent) {
            LOG_ERROR(<< "Unable to convert string '" << str
                      << "'"
                         " to unsigned long long: "
                      << ::strerror(errno));
        }
        return false;
    }

    if (endPtr != nullptr && *endPtr != '\0') {
        if (!silent) {
            LOG_ERROR(<< "Unable to convert string '" << str
                      << "'"
                         " to unsigned long long: first invalid character "
                      << endPtr);
        }
        return false;
    }

    i = ret;

    return true;
}

bool CStringUtils::_stringToType(bool silent, const std::string& str, unsigned long& i) {
    if (str.empty()) {
        if (!silent) {
            LOG_ERROR(<< "Unable to convert empty string to unsigned long");
        }
        return false;
    }

    char* endPtr(nullptr);
    errno = 0;
    unsigned long ret(::strtoul(str.c_str(), &endPtr, 0));

    if (ret == 0 && errno == EINVAL) {
        if (!silent) {
            LOG_ERROR(<< "Unable to convert string '" << str
                      << "'"
                         " to unsigned long: "
                      << ::strerror(errno));
        }
        return false;
    }

    if (ret == ULONG_MAX && errno == ERANGE) // note ULONG_MAX used for compatability with strtoul
    {
        if (!silent) {
            LOG_ERROR(<< "Unable to convert string '" << str
                      << "'"
                         " to unsigned long: "
                      << ::strerror(errno));
        }
        return false;
    }

    if (endPtr != nullptr && *endPtr != '\0') {
        if (!silent) {
            LOG_ERROR(<< "Unable to convert string '" << str
                      << "'"
                         " to unsigned long: first invalid character "
                      << endPtr);
        }
        return false;
    }

    i = ret;

    return true;
}

bool CStringUtils::_stringToType(bool silent, const std::string& str, unsigned int& i) {
    // First try to convert to unsigned long.
    // If that works check the range for unsigned int.
    unsigned long ret(0);
    if (CStringUtils::_stringToType(silent, str, ret) == false) {
        return false;
    }

    // Now check if the result is in range for unsigned int
    if (ret > std::numeric_limits<unsigned int>::max()) {
        if (!silent) {
            LOG_ERROR(<< "Unable to convert string '" << str
                      << "'"
                         " to unsigned int - out of range");
        }
        return false;
    }

    i = static_cast<unsigned int>(ret);

    return true;
}

bool CStringUtils::_stringToType(bool silent, const std::string& str, unsigned short& i) {
    // First try to convert to unsigned long.
    // If that works check the range for unsigned short.
    unsigned long ret(0);
    if (CStringUtils::_stringToType(silent, str, ret) == false) {
        return false;
    }

    // Now check if the result is in range for unsigned short
    if (ret > std::numeric_limits<unsigned short>::max()) {
        if (!silent) {
            LOG_ERROR(<< "Unable to convert string '" << str
                      << "'"
                         " to unsigned short - out of range");
        }
        return false;
    }

    i = static_cast<unsigned short>(ret);

    return true;
}

bool CStringUtils::_stringToType(bool silent, const std::string& str, long long& i) {
    if (str.empty()) {
        if (!silent) {
            LOG_ERROR(<< "Unable to convert empty string to long long");
        }
        return false;
    }

    char* endPtr(nullptr);
    errno = 0;
    long long ret(::strtoll(str.c_str(), &endPtr, 0));

    if (ret == 0 && errno == EINVAL) {
        if (!silent) {
            LOG_ERROR(<< "Unable to convert string '" << str
                      << "'"
                         " to long long: "
                      << ::strerror(errno));
        }
        return false;
    }

    if ((ret == LLONG_MIN || ret == LLONG_MAX) && errno == ERANGE) // note LLONG_MAX used for compatability with strtoll
    {
        if (!silent) {
            LOG_ERROR(<< "Unable to convert string '" << str
                      << "'"
                         " to long long: "
                      << ::strerror(errno));
        }
        return false;
    }

    if (endPtr != nullptr && *endPtr != '\0') {
        if (!silent) {
            LOG_ERROR(<< "Unable to convert string '" << str
                      << "'"
                         " to long long: first invalid character "
                      << endPtr);
        }
        return false;
    }

    i = ret;

    return true;
}

bool CStringUtils::_stringToType(bool silent, const std::string& str, long& i) {
    if (str.empty()) {
        if (!silent) {
            LOG_ERROR(<< "Unable to convert empty string to long");
        }
        return false;
    }

    char* endPtr(nullptr);
    errno = 0;
    long ret(::strtol(str.c_str(), &endPtr, 0));

    if (ret == 0 && errno == EINVAL) {
        if (!silent) {
            LOG_ERROR(<< "Unable to convert string '" << str
                      << "'"
                         " to long: "
                      << ::strerror(errno));
        }
        return false;
    }

    if ((ret == LONG_MIN || ret == LONG_MAX) && errno == ERANGE) // note LONG_MAX used for compatability with strtol
    {
        if (!silent) {
            LOG_ERROR(<< "Unable to convert string '" << str
                      << "'"
                         " to long: "
                      << ::strerror(errno));
        }
        return false;
    }

    if (endPtr != nullptr && *endPtr != '\0') {
        if (!silent) {
            LOG_ERROR(<< "Unable to convert string '" << str
                      << "'"
                         " to long: first invalid character "
                      << endPtr);
        }
        return false;
    }

    i = ret;

    return true;
}

bool CStringUtils::_stringToType(bool silent, const std::string& str, int& i) {
    // First try to convert to long.  If that works check the range for int.
    long ret(0);
    if (CStringUtils::_stringToType(silent, str, ret) == false) {
        return false;
    }

    // Now check if the result is in range for int
    if (ret < std::numeric_limits<int>::min() || ret > std::numeric_limits<int>::max()) {
        if (!silent) {
            LOG_ERROR(<< "Unable to convert string '" << str
                      << "'"
                         " to int - out of range");
        }
        return false;
    }

    i = static_cast<int>(ret);

    return true;
}

bool CStringUtils::_stringToType(bool silent, const std::string& str, short& i) {
    // First try to convert to long.  If that works check the range for short.
    long ret(0);
    if (CStringUtils::_stringToType(silent, str, ret) == false) {
        return false;
    }

    // Now check if the result is in range for short
    if (ret < std::numeric_limits<short>::min() ||
        ret > std::numeric_limits<short>::max()) {
        if (!silent) {
            LOG_ERROR(<< "Unable to convert string '" << str
                      << "'"
                         " to short - out of range");
        }
        return false;
    }

    i = static_cast<short>(ret);

    return true;
}

bool CStringUtils::_stringToType(bool silent, const std::string& str, bool& ret) {
    switch (str.length()) {
    case 0:
        if (!silent) {
            LOG_ERROR(<< "Cannot convert empty string to bool");
        }
        return false;
    case 1:
        switch (str[0]) {
        case 'T':
        case 'Y':
        case 't':
        case 'y':
            ret = true;
            return true;
        case 'F':
        case 'N':
        case 'f':
        case 'n':
            ret = false;
            return true;
        }
        break;
    case 2:
        if (CStrCaseCmp::strCaseCmp(str.c_str(), "no") == 0) {
            ret = false;
            return true;
        }
        if (CStrCaseCmp::strCaseCmp(str.c_str(), "on") == 0) {
            ret = true;
            return true;
        }
        break;
    case 3:
        if (CStrCaseCmp::strCaseCmp(str.c_str(), "yes") == 0) {
            ret = true;
            return true;
        }
        if (CStrCaseCmp::strCaseCmp(str.c_str(), "off") == 0) {
            ret = false;
            return true;
        }
        break;
    case 4:
        if (CStrCaseCmp::strCaseCmp(str.c_str(), "true") == 0) {
            ret = true;
            return true;
        }
        break;
    case 5:
        if (CStrCaseCmp::strCaseCmp(str.c_str(), "false") == 0) {
            ret = false;
            return true;
        }
        break;
    }

    long l(0);
    if (CStringUtils::_stringToType(silent, str, l) == false) {
        if (!silent) {
            LOG_ERROR(<< "Cannot convert " << str << " to bool");
        }
        return false;
    }

    ret = (l != 0);

    return true;
}

bool CStringUtils::_stringToType(bool silent, const std::string& str, double& d) {
    if (str.empty()) {
        if (!silent) {
            LOG_ERROR(<< "Unable to convert empty string to double");
        }
        return false;
    }

    char* endPtr(nullptr);
    errno = 0;
    double ret(::strtod(str.c_str(), &endPtr));

    if (ret == 0 && errno == EINVAL) {
        if (!silent) {
            LOG_ERROR(<< "Unable to convert string '" << str
                      << "'"
                         " to double: "
                      << ::strerror(errno));
        }
        return false;
    }

    if ((ret == HUGE_VAL || ret == -HUGE_VAL) && errno == ERANGE) {
        if (!silent) {
            LOG_ERROR(<< "Unable to convert string '" << str
                      << "'"
                         " to double: "
                      << ::strerror(errno));
        }
        return false;
    }

    if (endPtr != nullptr && *endPtr != '\0') {
        if (!silent) {
            LOG_ERROR(<< "Unable to convert string '" << str
                      << "'"
                         " to double: first invalid character "
                      << endPtr);
        }
        return false;
    }

    d = ret;

    return true;
}

bool CStringUtils::_stringToType(bool silent, const std::string& str, char& c) {
    if (str.length() != 1) {
        if (!silent) {
            LOG_ERROR(<< "Unable to convert string '" << str
                      << "' to char: " << (str.empty() ? "too short" : "too long"));
        }
        return false;
    }

    c = str[0];

    return true;
}

// This may seem silly, but it allows generic code to be written
bool CStringUtils::_stringToType(bool /* silent */, const std::string& str, std::string& outStr) {
    outStr = str;

    return true;
}

void CStringUtils::tokenise(const std::string& delim,
                            const std::string& str,
                            TStrVec& tokens,
                            std::string& remainder) {
    std::string::size_type pos(0);

    for (;;) {
        std::string::size_type pos2(str.find(delim, pos));
        if (pos2 == std::string::npos) {
            remainder.assign(str, pos, str.size() - pos);
            break;
        } else {
            tokens.push_back(str.substr(pos, pos2 - pos));
            pos = pos2 + delim.size();
        }
    }
}

std::string CStringUtils::longestCommonSubstr(const std::string& str1,
                                              const std::string& str2) {
    std::string common;
    if (str1.empty() || str2.empty()) {
        return common;
    }

    size_t firstLen(str1.length());
    size_t secondLen(str2.length());

    // Set up the matrix
    using T2DSizeArray = boost::multi_array<size_t, 2>;
    T2DSizeArray matrix(boost::extents[firstLen][secondLen]);

    size_t maxLen(0);
    size_t lastSubstrBegin(0);

    for (size_t i = 0; i < firstLen; ++i) {
        for (size_t j = 0; j < secondLen; ++j) {
            if (str1[i] != str2[j]) {
                matrix[i][j] = 0;
            } else {
                if (i == 0 || j == 0) {
                    matrix[i][j] = 1;
                } else {
                    matrix[i][j] = 1 + matrix[i - 1][j - 1];
                }

                if (matrix[i][j] > maxLen) {
                    maxLen = matrix[i][j];
                    size_t thisSubstrBegin(i - maxLen + 1);

                    if (lastSubstrBegin == thisSubstrBegin) {
                        // We're continuing the current longest common substring
                        common += str1[i];
                    } else {
                        // We're starting a new longest common substring
                        common.assign(str1, thisSubstrBegin, maxLen);
                        lastSubstrBegin = thisSubstrBegin;
                    }
                }
            }
        }
    }

    return common;
}

std::string CStringUtils::longestCommonSubsequence(const std::string& str1,
                                                   const std::string& str2) {
    std::string common;
    if (str1.empty() || str2.empty()) {
        return common;
    }

    size_t firstLen(str1.length());
    size_t secondLen(str2.length());

    // Set up the matrix - dimensions are one bigger than the string lengths
    using T2DSizeArray = boost::multi_array<size_t, 2>;
    T2DSizeArray matrix(boost::extents[firstLen + 1][secondLen + 1]);

    // Initialise the top row and left column of the matrix to zero
    for (size_t i = 0; i <= firstLen; ++i) {
        matrix[i][0] = 0;
    }

    for (size_t j = 0; j <= secondLen; ++j) {
        matrix[0][j] = 0;
    }

    // Fill in the rest of the matrix
    for (size_t i = 1; i <= firstLen; ++i) {
        for (size_t j = 1; j <= secondLen; ++j) {
            if (str1[i - 1] == str2[j - 1]) {
                matrix[i][j] = matrix[i - 1][j - 1] + 1;
            } else {
                matrix[i][j] = std::max(matrix[i][j - 1], matrix[i - 1][j]);
            }
        }
    }

    // Find the length of the longest common subsequence from the bottom right
    // corner of the matrix - if this length is zero, we don't need to backtrack
    // to find the actual characters
    size_t seqLen(matrix[firstLen][secondLen]);
    if (seqLen > 0) {
        // Create a string of NULLs to be overwritten (in reverse order) by the
        // actual characters
        common.resize(seqLen);
        size_t resPos(seqLen - 1);

        // Now backtrack through the matrix to find the common sequence
        size_t i(firstLen);
        size_t j(secondLen);
        while (i > 0 && j > 0) {
            if (str1[i - 1] == str2[j - 1]) {
                common[resPos] = str1[i - 1];

                // If we've got all the characters we need we can stop early
                if (resPos == 0) {
                    break;
                }

                --i;
                --j;
                --resPos;
            } else {
                if (matrix[i][j - 1] >= matrix[i - 1][j]) {
                    --j;
                } else {
                    --i;
                }
            }
        }
    }

    return common;
}

std::string CStringUtils::wideToNarrow(const std::wstring& wideStr) {
    // Annoyingly, the STL character transformations only work on
    // character arrays, and not std::string objects
    std::string narrowStr(wideStr.length(), '\0');

    // Note: this won't always work for non-ASCII data, and it can't
    // cope with UTF8 either, so we should replace it with a proper
    // string conversion library, e.g. ICU
    using TWCharTCType = std::ctype<wchar_t>;
    std::use_facet<TWCharTCType>(CStringUtils::locale())
        .narrow(wideStr.data(), wideStr.data() + wideStr.length(), '?', &narrowStr[0]);
    return narrowStr;
}

std::wstring CStringUtils::narrowToWide(const std::string& narrowStr) {
    // Annoyingly, the STL character transformations only work on
    // character arrays, and not std::string objects
    std::wstring wideStr(narrowStr.length(), L'\0');

    // Note: this won't always work for non-ASCII data, and it can't
    // cope with UTF8 either, so we should replace it with a proper
    // string conversion library, e.g. ICU
    using TWCharTCType = std::ctype<wchar_t>;
    std::use_facet<TWCharTCType>(CStringUtils::locale())
        .widen(narrowStr.data(), narrowStr.data() + narrowStr.length(), &wideStr[0]);
    return wideStr;
}

const std::locale& CStringUtils::locale() {
    static std::locale loc;
    return loc;
}
}
}
