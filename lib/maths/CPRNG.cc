/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CPRNG.h>

#include <core/CPersistUtils.h>
#include <core/CStringUtils.h>

#include <maths/CChecksum.h>

#include <algorithm>

namespace ml {
namespace maths {

namespace {
namespace detail {

//! Discard a sequence of \p n random numbers.
template<typename PRNG>
inline void discard(std::uint64_t n, PRNG& rng) {
    for (/**/; n > 0; --n) {
        rng();
    }
}

//! Rotate about the \p k'th bit.
std::uint64_t rotl(const std::uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}
}
}

CPRNG::CSplitMix64::CSplitMix64() : m_X(0) {
    this->seed();
}

CPRNG::CSplitMix64::CSplitMix64(result_type seed) : m_X(0) {
    this->seed(seed);
}

bool CPRNG::CSplitMix64::operator==(CSplitMix64 other) const {
    return m_X == other.m_X;
}

void CPRNG::CSplitMix64::seed() {
    m_X = 0;
}

void CPRNG::CSplitMix64::seed(result_type seed) {
    m_X = seed;
}

CPRNG::CSplitMix64::result_type CPRNG::CSplitMix64::operator()() {
    result_type x = (m_X += A);
    x = (x ^ (x >> 30)) * B;
    x = (x ^ (x >> 27)) * C;
    return x ^ (x >> 31);
}

void CPRNG::CSplitMix64::discard(result_type n) {
    detail::discard(n, *this);
}

std::string CPRNG::CSplitMix64::toString() const {
    return core::CStringUtils::typeToString(m_X);
}

bool CPRNG::CSplitMix64::fromString(const std::string& state) {
    return core::CStringUtils::stringToType(state, m_X);
}

std::uint64_t CPRNG::CSplitMix64::checksum(std::uint64_t seed) const {
    return CChecksum::calculate(seed, m_X);
}

const CPRNG::CSplitMix64::result_type CPRNG::CSplitMix64::A(0x9E3779B97F4A7C15);
const CPRNG::CSplitMix64::result_type CPRNG::CSplitMix64::B(0xBF58476D1CE4E5B9);
const CPRNG::CSplitMix64::result_type CPRNG::CSplitMix64::C(0x94D049BB133111EB);

CPRNG::CXorOShiro128Plus::CXorOShiro128Plus() {
    this->seed();
}

CPRNG::CXorOShiro128Plus::CXorOShiro128Plus(result_type seed) {
    this->seed(seed);
}

bool CPRNG::CXorOShiro128Plus::operator==(const CXorOShiro128Plus& other) const {
    return std::equal(&m_X[0], &m_X[2], &other.m_X[0]);
}

void CPRNG::CXorOShiro128Plus::seed() {
    this->seed(0);
}

void CPRNG::CXorOShiro128Plus::seed(result_type seed) {
    CSplitMix64 seeds(seed);
    seeds.generate(&m_X[0], &m_X[2]);
}

CPRNG::CXorOShiro128Plus::result_type CPRNG::CXorOShiro128Plus::operator()() {
    result_type x0 = m_X[0];
    result_type x1 = m_X[1];
    result_type result = x0 + x1;
    x1 ^= x0;
    m_X[0] = detail::rotl(x0, 55) ^ x1 ^ (x1 << 14);
    m_X[1] = detail::rotl(x1, 36);
    return result;
}

void CPRNG::CXorOShiro128Plus::discard(result_type n) {
    detail::discard(n, *this);
}

void CPRNG::CXorOShiro128Plus::jump() {
    result_type x[2] = {0};
    for (std::size_t i = 0; i < 2; ++i) {
        for (unsigned int b = 0; b < 64; ++b) {
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

std::string CPRNG::CXorOShiro128Plus::toString() const {
    const result_type* begin = &m_X[0];
    const result_type* end = &m_X[2];
    return core::CPersistUtils::toString(begin, end);
}

bool CPRNG::CXorOShiro128Plus::fromString(const std::string& state) {
    return core::CPersistUtils::fromString(state, &m_X[0], &m_X[2]);
}

std::uint64_t CPRNG::CXorOShiro128Plus::checksum(std::uint64_t seed) const {
    return CChecksum::calculate(seed, m_X);
}

const CPRNG::CXorOShiro128Plus::result_type CPRNG::CXorOShiro128Plus::JUMP[] = {
    0xbeac0467eba5facb, 0xd86b048b86aa9922};

CPRNG::CXorShift1024Mult::CXorShift1024Mult() : m_P(0) {
    this->seed();
}

CPRNG::CXorShift1024Mult::CXorShift1024Mult(result_type seed) : m_P(0) {
    this->seed(seed);
}

bool CPRNG::CXorShift1024Mult::operator==(const CXorShift1024Mult& other) const {
    return m_P == other.m_P && std::equal(&m_X[0], &m_X[16], &other.m_X[0]);
}

void CPRNG::CXorShift1024Mult::seed() {
    this->seed(0);
}

void CPRNG::CXorShift1024Mult::seed(result_type seed) {
    CSplitMix64 seeds(seed);
    seeds.generate(&m_X[0], &m_X[16]);
}

CPRNG::CXorShift1024Mult::result_type CPRNG::CXorShift1024Mult::operator()() {
    result_type x0 = m_X[m_P];
    m_P = (m_P + 1) & 15;
    result_type x1 = m_X[m_P];
    x1 ^= x1 << 31;
    m_X[m_P] = x1 ^ x0 ^ (x1 >> 11) ^ (x0 >> 30);
    return m_X[m_P] * A;
}

void CPRNG::CXorShift1024Mult::discard(result_type n) {
    detail::discard(n, *this);
}

void CPRNG::CXorShift1024Mult::jump() {
    result_type t[16] = {0};

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

std::string CPRNG::CXorShift1024Mult::toString() const {
    const result_type* begin = &m_X[0];
    const result_type* end = &m_X[16];
    return core::CPersistUtils::toString(begin, end) +
           core::CPersistUtils::PAIR_DELIMITER + core::CStringUtils::typeToString(m_P);
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

std::uint64_t CPRNG::CXorShift1024Mult::checksum(std::uint64_t seed) const {
    return CChecksum::calculate(seed, m_X);
}

const CPRNG::CXorShift1024Mult::result_type CPRNG::CXorShift1024Mult::A(1181783497276652981);
const CPRNG::CXorShift1024Mult::result_type CPRNG::CXorShift1024Mult::JUMP[16] = {
    0x84242f96eca9c41d, 0xa3c65b8776f96855, 0x5b34a39f070b5837,
    0x4489affce4f31a1e, 0x2ffeeb0a48316f40, 0xdc2d9891fe68c022,
    0x3659132bb12fea70, 0xaac17d8efa43cab8, 0xc4cb815590989b13,
    0x5ee975283d71c93b, 0x691548c86c1bd540, 0x7910c41d10a1e6a5,
    0x0b5fc64563b3e2a8, 0x047f7684e9fc949d, 0xb99181f2d8f685ca,
    0x284600e3f30e38c3};
}
}
