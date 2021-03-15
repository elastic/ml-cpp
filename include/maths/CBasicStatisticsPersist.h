/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CBasicStatisticsPersist_h
#define INCLUDED_ml_maths_CBasicStatisticsPersist_h

#include <core/CHashing.h>
#include <core/CLogger.h>
#include <core/CStringUtils.h>

#include <maths/CBasicStatistics.h>
#include <maths/CDoublePrecisionStorage.h>
#include <maths/CLinearAlgebraPersist.h>

#include <cstddef>

namespace ml {
namespace maths {
namespace basic_statistics_detail {
//! Function to do conversion from string.
template<typename T>
bool stringToType(const std::string& str, T& value) {
    return core::CStringUtils::stringToType(str, value);
}
//! Function to do conversion from string to float storage.
inline bool stringToType(const std::string& str, CFloatStorage& value) {
    return value.fromString(str);
}
//! Function to do conversion from string to double storage.
inline bool stringToType(const std::string& str, CDoublePrecisionStorage& value) {
    double d;
    if (core::CStringUtils::stringToType<double>(str, d) == false) {
        return false;
    }
    value = d;
    return true;
}
//! Function to do conversion from string to a vector.
template<typename T, std::size_t N>
bool stringToType(const std::string& str, CVectorNx1<T, N>& value) {
    return value.fromDelimited(str);
}
//! Function to do conversion from string to a symmetric matrix.
template<typename T, std::size_t N>
bool stringToType(const std::string& str, CSymmetricMatrixNxN<T, N>& value) {
    return value.fromDelimited(str);
}
//! Function to do conversion from string to a vector.
template<typename T>
bool stringToType(const std::string& str, CVector<T>& value) {
    return value.fromDelimited(str);
}
//! Function to do conversion from string to a symmetric matrix.
template<typename T>
bool stringToType(const std::string& str, CSymmetricMatrix<T>& value) {
    return value.fromDelimited(str);
}

//! Function to do conversion to a string.
template<typename T>
inline std::string typeToString(const T& value) {
    return core::CStringUtils::typeToStringPrecise(value, core::CIEEE754::E_SinglePrecision);
}
//! Function to do conversion to a string from float storage.
inline std::string typeToString(const CFloatStorage& value) {
    return value.toString();
}
//! Function to do conversion to a string from double storage.
inline std::string typeToString(const CDoublePrecisionStorage& value) {
    return core::CStringUtils::typeToStringPrecise(double(value), core::CIEEE754::E_DoublePrecision);
}
//! Function to do conversion to a string from a vector.
template<typename T, std::size_t N>
inline std::string typeToString(const CVectorNx1<T, N>& value) {
    return value.toDelimited();
}
//! Function to do conversion to a string from a symmetric matrix.
template<typename T, std::size_t N>
inline std::string typeToString(const CSymmetricMatrixNxN<T, N>& value) {
    return value.toDelimited();
}
//! Function to do conversion to a string from a vector.
template<typename T>
inline std::string typeToString(const CVector<T>& value) {
    return value.toDelimited();
}
//! Function to do conversion to a string from a symmetric matrix.
template<typename T>
inline std::string typeToString(const CSymmetricMatrix<T>& value) {
    return value.toDelimited();
}
}

template<typename T, unsigned int ORDER>
bool CBasicStatistics::SSampleCentralMoments<T, ORDER>::fromDelimited(const std::string& str) {
    if (str.empty()) {
        LOG_ERROR(<< "Empty accumulator representation");
        return false;
    }

    // Reuse this same string to avoid as many memory allocations as
    // possible.  The reservation is 15 because:
    // a) For string implementations using the short string
    //    optimisation we don't want to cause an unnecessary
    //    allocation
    // b) The count comes first and it's likely to be shorter than
    //    the subsequent moments
    std::string token;
    token.reserve(15);
    std::size_t delimPos{str.find(INTERNAL_DELIMITER, 0)};
    if (delimPos == std::string::npos) {
        token.assign(str, 0, str.length());
    } else {
        token.assign(str, 0, delimPos);
    }

    if (!basic_statistics_detail::stringToType(token, s_Count)) {
        LOG_ERROR(<< "Invalid count : element " << token << " in " << str);
        return false;
    }

    std::size_t lastDelimPos{delimPos};
    std::size_t index{0};
    while (lastDelimPos != std::string::npos && index < ORDER) {
        delimPos = str.find(INTERNAL_DELIMITER, lastDelimPos + 1);
        if (delimPos == std::string::npos) {
            token.assign(str, lastDelimPos + 1, str.length() - lastDelimPos);
        } else {
            token.assign(str, lastDelimPos + 1, delimPos - lastDelimPos - 1);
        }

        if (!basic_statistics_detail::stringToType(token, s_Moments[index++])) {
            LOG_ERROR(<< "Invalid moment " << index << " : element " << token
                      << " in " << str);
            return false;
        }

        lastDelimPos = delimPos;
    }

    return true;
}

template<typename T, unsigned int ORDER>
std::string CBasicStatistics::SSampleCentralMoments<T, ORDER>::toDelimited() const {
    std::string result(basic_statistics_detail::typeToString(s_Count));
    for (std::size_t index = 0; index < ORDER; ++index) {
        result += INTERNAL_DELIMITER;
        result += basic_statistics_detail::typeToString(s_Moments[index]);
    }

    return result;
}

template<typename T, unsigned int ORDER>
uint64_t CBasicStatistics::SSampleCentralMoments<T, ORDER>::checksum() const {
    std::ostringstream raw;
    raw << basic_statistics_detail::typeToString(s_Count);
    for (std::size_t i = 0; i < ORDER; ++i) {
        raw << ' ';
        raw << basic_statistics_detail::typeToString(s_Moments[i]);
    }
    core::CHashing::CSafeMurmurHash2String64 hasher;
    return hasher(raw.str());
}

template<typename POINT>
bool CBasicStatistics::SSampleCovariances<POINT>::fromDelimited(std::string str) {
    std::size_t dimension{0u};
    std::size_t pos{str.find_first_of(CLinearAlgebra::DELIMITER)};
    if (!core::CStringUtils::stringToType(str.substr(0, pos), dimension)) {
        LOG_ERROR(<< "Failed to extract dimension from " << str.substr(0, pos));
        return false;
    }
    str = str.substr(pos + 1);

    std::size_t count{0u};
    for (std::size_t i = 0; i < dimension; ++i) {
        count = str.find_first_of(CLinearAlgebra::DELIMITER, count + 1);
    }
    if (!s_Count.fromDelimited(str.substr(0, count))) {
        LOG_ERROR(<< "Failed to extract counts from " << str.substr(0, count));
        return false;
    }

    str = str.substr(count + 1);
    std::size_t means{0u};
    for (std::size_t i = 0; i < dimension; ++i) {
        means = str.find_first_of(CLinearAlgebra::DELIMITER, means + 1);
    }
    if (!s_Mean.fromDelimited(str.substr(0, means))) {
        LOG_ERROR(<< "Failed to extract means from " << str.substr(0, means));
        return false;
    }

    str = str.substr(means + 1);
    if (!s_Covariances.fromDelimited(str)) {
        LOG_ERROR(<< "Failed to extract covariances from " << str);
        return false;
    }

    return true;
}

template<typename POINT>
std::string CBasicStatistics::SSampleCovariances<POINT>::toDelimited() const {
    return core::CStringUtils::typeToString(s_Count.dimension()) +
           CLinearAlgebra::DELIMITER + s_Count.toDelimited() +
           CLinearAlgebra::DELIMITER + s_Mean.toDelimited() +
           CLinearAlgebra::DELIMITER + s_Covariances.toDelimited();
}

template<typename POINT>
uint64_t CBasicStatistics::SSampleCovariances<POINT>::checksum() const {
    std::ostringstream raw;
    raw << basic_statistics_detail::typeToString(s_Count);
    raw << ' ';
    raw << basic_statistics_detail::typeToString(s_Mean);
    raw << ' ';
    raw << basic_statistics_detail::typeToString(s_Covariances);
    core::CHashing::CSafeMurmurHash2String64 hasher;
    return hasher(raw.str());
}

template<typename T, typename CONTAINER, typename LESS>
bool CBasicStatistics::COrderStatisticsImpl<T, CONTAINER, LESS>::fromDelimited(const std::string& value) {
    return this->fromDelimited(value, [](const std::string& value_, T& result) {
        return basic_statistics_detail::stringToType(value_, result);
    });
}

template<typename T, typename CONTAINER, typename LESS>
bool CBasicStatistics::COrderStatisticsImpl<T, CONTAINER, LESS>::fromDelimited(
    const std::string& value,
    const TFromString& fromString) {
    this->clear();

    if (value.empty()) {
        return true;
    }

    T statistic;

    std::size_t delimPos{value.find(INTERNAL_DELIMITER)};
    if (delimPos == std::string::npos) {
        if (fromString(value, statistic) == false) {
            LOG_ERROR(<< "Invalid statistic in '" << value << "'");
            return false;
        }
        m_Statistics[--m_UnusedCount] = statistic;
        return true;
    }

    m_UnusedCount = m_Statistics.size();

    std::string token;
    token.reserve(15);
    token.assign(value, 0, delimPos);
    if (fromString(token, statistic) == false) {
        LOG_ERROR(<< "Invalid statistic '" << token << "' in '" << value << "'");
        return false;
    }
    m_Statistics[--m_UnusedCount] = statistic;

    while (delimPos < value.size()) {
        if (m_UnusedCount == 0) {
            LOG_ERROR(<< "Too many statistics in '" << value
                      << "' - expected at most " << m_Statistics.size());
            return false;
        }
        std::size_t nextDelimPos{
            std::min(value.find(INTERNAL_DELIMITER, delimPos + 1), value.size())};
        token.assign(value, delimPos + 1, nextDelimPos - delimPos - 1);
        if (fromString(token, statistic) == false) {
            LOG_ERROR(<< "Invalid statistic '" << token << "' in '" << value << "'");
            return false;
        }
        m_Statistics[--m_UnusedCount] = statistic;
        delimPos = nextDelimPos;
    }

    return true;
}

template<typename T, typename CONTAINER, typename LESS>
std::string CBasicStatistics::COrderStatisticsImpl<T, CONTAINER, LESS>::toDelimited() const {
    return this->toDelimited([](const T& value_) {
        return basic_statistics_detail::typeToString(value_);
    });
}

template<typename T, typename CONTAINER, typename LESS>
std::string
CBasicStatistics::COrderStatisticsImpl<T, CONTAINER, LESS>::toDelimited(const TToString& toString) const {
    if (this->count() == 0) {
        return std::string{};
    }
    std::size_t i{m_Statistics.size()};
    std::string result{toString(m_Statistics[i - 1])};
    for (--i; i > m_UnusedCount; --i) {
        result += INTERNAL_DELIMITER;
        result += toString(m_Statistics[i - 1]);
    }
    return result;
}

template<typename T, typename CONTAINER, typename LESS>
uint64_t CBasicStatistics::COrderStatisticsImpl<T, CONTAINER, LESS>::checksum(uint64_t seed) const {
    if (this->count() == 0) {
        return seed;
    }
    std::vector<T> sorted(this->begin(), this->end());
    std::sort(sorted.begin(), sorted.end(), m_Less);
    std::ostringstream raw;
    raw << basic_statistics_detail::typeToString(sorted[0]);
    for (std::size_t i = 1; i < sorted.size(); ++i) {
        raw << ' ';
        raw << basic_statistics_detail::typeToString(sorted[i]);
    }
    core::CHashing::CSafeMurmurHash2String64 hasher(seed);
    return hasher(raw.str());
}
}
}

#endif // INCLUDED_ml_maths_CBasicStatisticsPersist_h
