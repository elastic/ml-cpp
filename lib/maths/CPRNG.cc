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

#include <core/CPersistUtils.h>
#include <core/CStringUtils.h>

#include <maths/CPRNG.h>

#include <boost/numeric/conversion/bounds.hpp>

#include <algorithm>

namespace ml {
namespace maths {

namespace {
namespace detail {

//! Discard a sequence of \p n random numbers.
template<typename PRNG>
inline void discard(uint64_t n, PRNG &rng) {
    for (/**/; n > 0; --n) {
        rng();
    }
}

//! Rotate about the \p k'th bit.
uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

}
}

CPRNG::CSplitMix64::CSplitMix64(void) : m_X(0) {
    this->seed();
}

CPRNG::CSplitMix64::CSplitMix64(uint64_t seed) : m_X(0) {
    this->seed(seed);
}

bool CPRNG::CSplitMix64::operator==(CSplitMix64 other) const
{
    return m_X == other.m_X;
}

void CPRNG::CSplitMix64::seed(void) {
    m_X = 0;
}

void CPRNG::CSplitMix64::seed(uint64_t seed) {
    m_X = seed;
}

uint64_t CPRNG::CSplitMix64::min(void) {
    return 0;
}

uint64_t CPRNG::CSplitMix64::max(void) {
    return boost::numeric::bounds<uint64_t>::highest();
}

uint64_t CPRNG::CSplitMix64::operator()(void) {
    uint64_t x = (m_X += A);
    x = (x ^ (x >> 30)) * B;
    x = (x ^ (x >> 27)) * C;
    return x ^ (x >> 31);
}

void CPRNG::CSplitMix64::discard(uint64_t n) {
    detail::discard(n, *this);
}

std::string CPRNG::CSplitMix64::toString(void) const
{
    return core::CStringUtils::typeToString(m_X);
}

bool CPRNG::CSplitMix64::fromString(const std::string &state) {
    return core::CStringUtils::stringToType(state, m_X);
}

const uint64_t CPRNG::CSplitMix64::A(0x9E3779B97F4A7C15);
const uint64_t CPRNG::CSplitMix64::B(0xBF58476D1CE4E5B9);
const uint64_t CPRNG::CSplitMix64::C(0x94D049BB133111EB);


CPRNG::CXorOShiro128Plus::CXorOShiro128Plus(void) {
    this->seed();
}

CPRNG::CXorOShiro128Plus::CXorOShiro128Plus(uint64_t seed) {
    this->seed(seed);
}

bool CPRNG::CXorOShiro128Plus::operator==(const CXorOShiro128Plus &other) const
{
    return std::equal(&m_X[0], &m_X[2], &other.m_X[0]);
}

void CPRNG::CXorOShiro128Plus::seed(void) {
    this->seed(0);
}

void CPRNG::CXorOShiro128Plus::seed(uint64_t seed) {
    CSplitMix64 seeds(seed);
    seeds.generate(&m_X[0], &m_X[2]);
}

uint64_t CPRNG::CXorOShiro128Plus::min(void) {
    return 0;
}

uint64_t CPRNG::CXorOShiro128Plus::max(void) {
    return boost::numeric::bounds<uint64_t>::highest();
}

uint64_t CPRNG::CXorOShiro128Plus::operator()(void) {
    uint64_t x0 = m_X[0];
    uint64_t x1 = m_X[1];
    uint64_t result = x0 + x1;
    x1 ^= x0;
    m_X[0] = detail::rotl(x0, 55) ^ x1 ^ (x1 << 14);
    m_X[1] = detail::rotl(x1, 36);
    return result;
}

void CPRNG::CXorOShiro128Plus::discard(uint64_t n) {
    detail::discard(n, *this);
}

void CPRNG::CXorOShiro128Plus::jump(void) {
    uint64_t x[2] = { 0 };
    for(std::size_t i = 0; i < 2; ++i) {
        for(unsigned int b = 0; b < 64; ++b) {
            if (JUMP[i] & 1ULL << b) {
                x[0] ^= m_X[0];
                x[1] ^= m_X[1];
            }
            this->operator()();
        }
    }

    m_X[0] = x[0];
    m_X[1] = x[1];
}

std::string CPRNG::CXorOShiro128Plus::toString(void) const
{
    const uint64_t *begin = &m_X[0];
    const uint64_t *end   = &m_X[2];
    return core::CPersistUtils::toString(begin, end);
}

bool CPRNG::CXorOShiro128Plus::fromString(const std::string &state) {
    return core::CPersistUtils::fromString(state, &m_X[0], &m_X[2]);
}

const uint64_t CPRNG::CXorOShiro128Plus::JUMP[] = { 0xbeac0467eba5facb, 0xd86b048b86aa9922 };


CPRNG::CXorShift1024Mult::CXorShift1024Mult(void) : m_P(0) {
    this->seed();
}

CPRNG::CXorShift1024Mult::CXorShift1024Mult(uint64_t seed) : m_P(0) {
    this->seed(seed);
}

bool CPRNG::CXorShift1024Mult::operator==(const CXorShift1024Mult &other) const
{
    return m_P == other.m_P && std::equal(&m_X[0], &m_X[16], &other.m_X[0]);
}

void CPRNG::CXorShift1024Mult::seed(void) {
    this->seed(0);
}

void CPRNG::CXorShift1024Mult::seed(uint64_t seed) {
    CSplitMix64 seeds(seed);
    seeds.generate(&m_X[0], &m_X[16]);
}

uint64_t CPRNG::CXorShift1024Mult::min(void) {
    return 0;
}

uint64_t CPRNG::CXorShift1024Mult::max(void) {
    return boost::numeric::bounds<uint64_t>::highest();
}

uint64_t CPRNG::CXorShift1024Mult::operator()(void) {
    uint64_t x0 = m_X[m_P];
    m_P = (m_P + 1) & 15;
    uint64_t x1 = m_X[m_P];
    x1 ^= x1 << 31;
    m_X[m_P] = x1 ^ x0 ^ (x1 >> 11) ^ (x0 >> 30);
    return m_X[m_P] * A;
}

void CPRNG::CXorShift1024Mult::discard(uint64_t n) {
    detail::discard(n, *this);
}

void CPRNG::CXorShift1024Mult::jump(void) {
    uint64_t t[16] = { 0 };

    for (std::size_t i = 0; i < 16; ++i) {
        for (unsigned int b = 0; b < 64; ++b) {
            if (JUMP[i] & 1ULL << b) {
                for (int j = 0; j < 16; ++j) {
                    t[j] ^= m_X[(j + m_P) & 15];
                }
            }
            this->operator()();
        }
    }

    for (int j = 0; j < 16; j++) {
        m_X[(j + m_P) & 15] = t[j];
    }
}

std::string CPRNG::CXorShift1024Mult::toString(void) const
{
    const uint64_t *begin = &m_X[0];
    const uint64_t *end   = &m_X[16];
    return core::CPersistUtils::toString(begin, end)
           + core::CPersistUtils::PAIR_DELIMITER
           + core::CStringUtils::typeToString(m_P);
}

bool CPRNG::CXorShift1024Mult::fromString(std::string state) {
    std::size_t delimPos = state.find(core::CPersistUtils::PAIR_DELIMITER);
    if (delimPos == std::string::npos) {
        return false;
    }
    std::string p;
    p.assign(state, delimPos + 1, state.length() - delimPos);
    if (!core::CStringUtils::stringToType(p, m_P)) {
        return false;
    }
    state.resize(delimPos);
    return core::CPersistUtils::fromString(state, &m_X[0], &m_X[16]);
}

const uint64_t CPRNG::CXorShift1024Mult::A(1181783497276652981);
const uint64_t CPRNG::CXorShift1024Mult::JUMP[16] =
{
    0x84242f96eca9c41d, 0xa3c65b8776f96855, 0x5b34a39f070b5837, 0x4489affce4f31a1e,
    0x2ffeeb0a48316f40, 0xdc2d9891fe68c022, 0x3659132bb12fea70, 0xaac17d8efa43cab8,
    0xc4cb815590989b13, 0x5ee975283d71c93b, 0x691548c86c1bd540, 0x7910c41d10a1e6a5,
    0x0b5fc64563b3e2a8, 0x047f7684e9fc949d, 0xb99181f2d8f685ca, 0x284600e3f30e38c3
};

}
}
