/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <core/CHashing.h>

#include <core/CFastMutex.h>
#include <core/CLogger.h>
#include <core/CScopedFastLock.h>
#include <core/CStringUtils.h>

#include <boost/config.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include <algorithm>
#include <iterator>

namespace ml {
namespace core {

namespace {
using TUniform32 = boost::random::uniform_int_distribution<std::uint32_t>;
CFastMutex generatorMutex;
boost::random::mt11213b generator;
}

const std::uint64_t CHashing::CUniversalHash::BIG_PRIME = 4294967291ULL;

CHashing::CUniversalHash::CUInt32Hash::CUInt32Hash()
    : m_M(1000), m_A(1), m_B(0) {
}

CHashing::CUniversalHash::CUInt32Hash::CUInt32Hash(std::uint32_t m, std::uint32_t a, std::uint32_t b)
    : m_M(m), m_A(a), m_B(b) {
}

std::uint32_t CHashing::CUniversalHash::CUInt32Hash::m() const {
    return m_M;
}

std::uint32_t CHashing::CUniversalHash::CUInt32Hash::a() const {
    return m_A;
}

std::uint32_t CHashing::CUniversalHash::CUInt32Hash::b() const {
    return m_B;
}

std::string CHashing::CUniversalHash::CUInt32Hash::print() const {
    std::ostringstream result;
    result << "\"((" << m_A << " * x + " << m_B << ") mod " << BIG_PRIME
           << ") mod " << m_M << "\"";
    return result.str();
}

CHashing::CUniversalHash::CUInt32UnrestrictedHash::CUInt32UnrestrictedHash()
    : m_A(1), m_B(0) {
}

CHashing::CUniversalHash::CUInt32UnrestrictedHash::CUInt32UnrestrictedHash(std::uint32_t a,
                                                                           std::uint32_t b)
    : m_A(a), m_B(b) {
}

std::uint32_t CHashing::CUniversalHash::CUInt32UnrestrictedHash::a() const {
    return m_A;
}

std::uint32_t CHashing::CUniversalHash::CUInt32UnrestrictedHash::b() const {
    return m_B;
}

std::string CHashing::CUniversalHash::CUInt32UnrestrictedHash::print() const {
    std::ostringstream result;
    result << "\"(" << m_A << " * x + " << m_B << ") mod " << BIG_PRIME << "\"";
    return result.str();
}

CHashing::CUniversalHash::CUInt32VecHash::CUInt32VecHash(std::uint32_t m,
                                                         const TUInt32Vec& a,
                                                         std::uint32_t b)
    : m_M(m), m_A(a), m_B(b) {
}

std::uint32_t CHashing::CUniversalHash::CUInt32VecHash::m() const {
    return m_M;
}

const CHashing::CUniversalHash::TUInt32Vec&
CHashing::CUniversalHash::CUInt32VecHash::a() const {
    return m_A;
}

std::uint32_t CHashing::CUniversalHash::CUInt32VecHash::b() const {
    return m_B;
}

std::string CHashing::CUniversalHash::CUInt32VecHash::print() const {
    std::ostringstream result;
    result << "\"((" << m_A[0] << "* x0";
    for (std::size_t i = 1; i < m_A.size(); ++i) {
        result << " + " << m_A[i] << "* x" << i;
    }
    result << ") mod " << BIG_PRIME << ") mod " << m_M << "\"";
    return result.str();
}

CHashing::CUniversalHash::CToString::CToString(const char delimiter)
    : m_Delimiter(delimiter) {
}

std::string CHashing::CUniversalHash::CToString::
operator()(const CUInt32UnrestrictedHash& hash) const {
    return CStringUtils::typeToString(hash.a()) + m_Delimiter +
           CStringUtils::typeToString(hash.b());
}

std::string CHashing::CUniversalHash::CToString::operator()(const CUInt32Hash& hash) const {
    return CStringUtils::typeToString(hash.m()) + m_Delimiter +
           CStringUtils::typeToString(hash.a()) + m_Delimiter +
           CStringUtils::typeToString(hash.b());
}

CHashing::CUniversalHash::CFromString::CFromString(const char delimiter)
    : m_Delimiter(delimiter) {
}

bool CHashing::CUniversalHash::CFromString::
operator()(const std::string& token, CUInt32UnrestrictedHash& hash) const {
    std::size_t delimPos = token.find(m_Delimiter);
    if (delimPos == std::string::npos) {
        LOG_ERROR(<< "Invalid hash state " << token);
        return false;
    }

    std::uint32_t a;
    std::uint32_t b;
    m_Token.assign(token, 0, delimPos);
    if (CStringUtils::stringToType(m_Token, a) == false) {
        LOG_ERROR(<< "Invalid multiplier in " << m_Token);
        return false;
    }
    m_Token.assign(token, delimPos + 1, token.length() - delimPos);
    if (CStringUtils::stringToType(m_Token, b) == false) {
        LOG_ERROR(<< "Invalid offset in " << m_Token);
        return false;
    }
    hash = CUInt32UnrestrictedHash(a, b);

    return true;
}
bool CHashing::CUniversalHash::CFromString::operator()(const std::string& token,
                                                       CUInt32Hash& hash) const {
    std::size_t firstDelimPos = token.find(m_Delimiter);
    if (firstDelimPos == std::string::npos) {
        LOG_ERROR(<< "Invalid hash state " << token);
        return false;
    }
    std::size_t secondDelimPos = token.find(m_Delimiter, firstDelimPos + 1);
    if (secondDelimPos == std::string::npos) {
        LOG_ERROR(<< "Invalid hash state " << token);
        return false;
    }

    std::uint32_t m;
    std::uint32_t a;
    std::uint32_t b;
    m_Token.assign(token, 0, firstDelimPos);
    if (CStringUtils::stringToType(m_Token, m) == false) {
        LOG_ERROR(<< "Invalid range in " << m_Token);
        return false;
    }
    m_Token.assign(token, firstDelimPos + 1, secondDelimPos - firstDelimPos - 1);
    if (CStringUtils::stringToType(m_Token, a) == false) {
        LOG_ERROR(<< "Invalid offset in " << m_Token);
        return false;
    }
    m_Token.assign(token, secondDelimPos + 1, token.length() - secondDelimPos);
    if (CStringUtils::stringToType(m_Token, b) == false) {
        LOG_ERROR(<< "Invalid multiplier in " << m_Token);
        return false;
    }
    hash = CUInt32Hash(m, a, b);

    return true;
}

void CHashing::CUniversalHash::generateHashes(std::size_t k, std::uint32_t m, TUInt32HashVec& result) {
    TUInt32Vec a;
    TUInt32Vec b;
    a.reserve(k);
    b.reserve(k);

    {
        CScopedFastLock scopedLock(generatorMutex);

        TUniform32 uniform1(1, static_cast<std::uint32_t>(BIG_PRIME - 1));
        std::generate_n(std::back_inserter(a), k,
                        [uniform1] { return uniform1(generator); });

        TUniform32 uniform0(0, static_cast<std::uint32_t>(BIG_PRIME - 1));
        std::generate_n(std::back_inserter(b), k,
                        [uniform0] { return uniform0(generator); });
    }
    for (unsigned int& ai : a) {
        if (ai == 0) {
            LOG_ERROR(<< "Expected a in [1," << BIG_PRIME << ")");
            ai = 1;
        }
    }

    result.reserve(k);
    for (std::size_t i = 0; i < k; ++i) {
        result.push_back(CUInt32Hash(m, a[i], b[i]));
    }
}

void CHashing::CUniversalHash::generateHashes(std::size_t k,
                                              TUInt32UnrestrictedHashVec& result) {
    TUInt32Vec a;
    TUInt32Vec b;
    a.reserve(k);
    b.reserve(k);

    {
        CScopedFastLock scopedLock(generatorMutex);

        TUniform32 uniform1(1, static_cast<std::uint32_t>(BIG_PRIME - 1));
        std::generate_n(std::back_inserter(a), k,
                        [uniform1] { return uniform1(generator); });

        TUniform32 uniform0(0, static_cast<std::uint32_t>(BIG_PRIME - 1));
        std::generate_n(std::back_inserter(b), k,
                        [uniform0] { return uniform0(generator); });
    }
    for (unsigned int& ai : a) {
        if (ai == 0) {
            LOG_ERROR(<< "Expected a in [1," << BIG_PRIME << ")");
            ai = 1;
        }
    }

    result.reserve(k);
    for (std::size_t i = 0; i < k; ++i) {
        result.push_back(CUInt32UnrestrictedHash(a[i], b[i]));
    }
}

void CHashing::CUniversalHash::generateHashes(std::size_t k,
                                              std::size_t n,
                                              std::uint32_t m,
                                              TUInt32VecHashVec& result) {
    using TUInt32VecVec = std::vector<TUInt32Vec>;

    TUInt32VecVec a;
    TUInt32Vec b;
    a.reserve(k);
    b.reserve(k);

    {
        CScopedFastLock scopedLock(generatorMutex);

        for (std::size_t i = 0; i < k; ++i) {
            a.push_back(TUInt32Vec());
            a.back().reserve(n);
            TUniform32 uniform1(1, static_cast<std::uint32_t>(BIG_PRIME - 1));
            std::generate_n(std::back_inserter(a.back()), n,
                            [uniform1] { return uniform1(generator); });
            for (unsigned int& aj : a.back()) {
                if (aj == 0) {
                    LOG_ERROR(<< "Expected a in [1," << BIG_PRIME << ")");
                    aj = 1;
                }
            }
        }

        TUniform32 uniform0(0, static_cast<std::uint32_t>(BIG_PRIME - 1));
        std::generate_n(std::back_inserter(b), k,
                        [uniform0] { return uniform0(generator); });
    }

    result.reserve(k);
    for (std::size_t i = 0; i < k; ++i) {
        result.push_back(CUInt32VecHash(m, a[i], b[i]));
    }
}

std::uint32_t CHashing::murmurHash32(const void* key, int length, std::uint32_t seed) {
    constexpr std::uint32_t m = 0x5bd1e995;
    constexpr int r = 24;

    std::uint32_t h = seed ^ length;

    // Note, remainder = length % 4
    const int remainder = length & 0x3;
    const auto* data = static_cast<const std::uint32_t*>(key);
    // Note, shift = (length - remainder) / 4
    const std::uint32_t* end = data + ((length - remainder) >> 2);

    while (data != end) {
        std::uint32_t k = *reinterpret_cast<const std::uint32_t*>(data);

        k *= m;
        k ^= k >> r;
        k *= m;

        h *= m;
        h ^= k;

        ++data;
    }

    const auto* remainingData = reinterpret_cast<const unsigned char*>(end);

    switch (remainder) {
    case 3:
        h ^= remainingData[2] << 16;
        BOOST_FALLTHROUGH;
    case 2:
        h ^= remainingData[1] << 8;
        BOOST_FALLTHROUGH;
    case 1:
        h ^= remainingData[0];
        h *= m;
        BOOST_FALLTHROUGH;
    default:
        break;
    }

    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;

    return h;
}

std::uint32_t CHashing::safeMurmurHash32(const void* key, int length, std::uint32_t seed) {
    constexpr std::uint32_t m = 0x5bd1e995;
    constexpr int r = 24;

    std::uint32_t h = seed ^ length;

    const auto* data = static_cast<const unsigned char*>(key);

    // Endian and alignment neutral implementation of the main loop.
    while (length >= 4) {
        std::uint32_t k;

        k = data[0];
        k |= data[1] << 8;
        k |= data[2] << 16;
        k |= data[3] << 24;

        k *= m;
        k ^= k >> r;
        k *= m;

        h *= m;
        h ^= k;

        data += 4;
        length -= 4;
    }

    switch (length) {
    case 3:
        h ^= data[2] << 16;
        BOOST_FALLTHROUGH;
    case 2:
        h ^= data[1] << 8;
        BOOST_FALLTHROUGH;
    case 1:
        h ^= data[0];
        h *= m;
        BOOST_FALLTHROUGH;
    default:
        break;
    };

    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;

    return h;
}

std::uint64_t CHashing::murmurHash64(const void* key, int length, std::uint64_t seed) {
    constexpr std::uint64_t m = 0xc6a4a7935bd1e995ULL;
    constexpr int r = 47;

    std::uint64_t h = seed ^ (length * m);

    // Note, remainder = length % 8
    const int remainder = length & 0x7;
    const auto* data = static_cast<const std::uint64_t*>(key);
    // Note, shift = (length - remainder) / 8
    const std::uint64_t* end = data + ((length - remainder) >> 3);

    while (data != end) {
        std::uint64_t k = *data;

        k *= m;
        k ^= k >> r;
        k *= m;

        h ^= k;
        h *= m;

        ++data;
    }

    const auto* remainingData = reinterpret_cast<const unsigned char*>(end);

    switch (remainder) {
    case 7:
        h ^= std::uint64_t(remainingData[6]) << 48;
        BOOST_FALLTHROUGH;
    case 6:
        h ^= std::uint64_t(remainingData[5]) << 40;
        BOOST_FALLTHROUGH;
    case 5:
        h ^= std::uint64_t(remainingData[4]) << 32;
        BOOST_FALLTHROUGH;
    case 4:
        h ^= std::uint64_t(remainingData[3]) << 24;
        BOOST_FALLTHROUGH;
    case 3:
        h ^= std::uint64_t(remainingData[2]) << 16;
        BOOST_FALLTHROUGH;
    case 2:
        h ^= std::uint64_t(remainingData[1]) << 8;
        BOOST_FALLTHROUGH;
    case 1:
        h ^= std::uint64_t(remainingData[0]);
        h *= m;
        BOOST_FALLTHROUGH;
    default:
        break;
    }

    h ^= h >> r;
    h *= m;
    h ^= h >> r;

    return h;
}

std::uint64_t CHashing::safeMurmurHash64(const void* key, int length, std::uint64_t seed) {
    constexpr std::uint64_t m = 0xc6a4a7935bd1e995ULL;
    constexpr int r = 47;

    std::uint64_t h = seed ^ (length * m);

    const auto* data = static_cast<const unsigned char*>(key);

    // Endian and alignment neutral implementation.
    while (length >= 8) {
        std::uint64_t k;

        k = std::uint64_t(data[0]);
        k |= std::uint64_t(data[1]) << 8;
        k |= std::uint64_t(data[2]) << 16;
        k |= std::uint64_t(data[3]) << 24;
        k |= std::uint64_t(data[4]) << 32;
        k |= std::uint64_t(data[5]) << 40;
        k |= std::uint64_t(data[6]) << 48;
        k |= std::uint64_t(data[7]) << 56;

        k *= m;
        k ^= k >> r;
        k *= m;

        h ^= k;
        h *= m;

        data += 8;
        length -= 8;
    }

    switch (length) {
    case 7:
        h ^= std::uint64_t(data[6]) << 48;
        BOOST_FALLTHROUGH;
    case 6:
        h ^= std::uint64_t(data[5]) << 40;
        BOOST_FALLTHROUGH;
    case 5:
        h ^= std::uint64_t(data[4]) << 32;
        BOOST_FALLTHROUGH;
    case 4:
        h ^= std::uint64_t(data[3]) << 24;
        BOOST_FALLTHROUGH;
    case 3:
        h ^= std::uint64_t(data[2]) << 16;
        BOOST_FALLTHROUGH;
    case 2:
        h ^= std::uint64_t(data[1]) << 8;
        BOOST_FALLTHROUGH;
    case 1:
        h ^= std::uint64_t(data[0]);
        h *= m;
        BOOST_FALLTHROUGH;
    default:
        break;
    }

    h ^= h >> r;
    h *= m;
    h ^= h >> r;

    return h;
}

std::uint32_t CHashing::hashCombine(std::uint32_t seed, std::uint32_t h) {
    constexpr std::uint32_t C = 0x9e3779b9;
    seed ^= h + C + (seed << 6) + (seed >> 2);
    return seed;
}

std::uint64_t CHashing::hashCombine(std::uint64_t seed, std::uint64_t h) {
    // As with boost::hash_combine use the binary expansion of an irrational
    // number to generate 64 random independent bits, i.e.
    //   C = 2^64 / "golden ratio" = 2^65 / (1 + 5^(1/2))
    constexpr std::uint64_t C = 0x9e3779b97f4A7c15ULL;
    seed ^= h + C + (seed << 6) + (seed >> 2);
    return seed;
}
}
}
