/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CPackedBitVector.h>

#include <core/CHashing.h>
#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CPersistUtils.h>
#include <core/CStringUtils.h>

namespace ml {
namespace core {

CPackedBitVector::CPackedBitVector()
    : m_Dimension(0), m_First(false), m_Parity(true) {
}

CPackedBitVector::CPackedBitVector(bool bit)
    : m_Dimension(1), m_First(bit), m_Parity(true), m_RunLengths(1, 1) {
}

CPackedBitVector::CPackedBitVector(std::size_t dimension, bool bit)
    : m_Dimension(static_cast<uint32_t>(dimension)), m_First(bit), m_Parity(true) {
    if (dimension > 0) {
        std::size_t remainder = static_cast<std::size_t>(MAX_RUN_LENGTH);
        for (/**/; remainder <= dimension;
             remainder += static_cast<std::size_t>(MAX_RUN_LENGTH)) {
            m_RunLengths.push_back(MAX_RUN_LENGTH);
        }
        remainder -= static_cast<std::size_t>(MAX_RUN_LENGTH);
        m_RunLengths.push_back(static_cast<uint8_t>(dimension - remainder));
    }
}

CPackedBitVector::CPackedBitVector(const TBoolVec& bits)
    : m_Dimension(static_cast<uint32_t>(bits.size())),
      m_First(bits.empty() ? false : bits[0]), m_Parity(true) {
    std::size_t length = 1u;
    for (std::size_t i = 1u; i < bits.size(); ++i) {
        if (bits[i] == bits[i - 1]) {
            if (++length == static_cast<std::size_t>(MAX_RUN_LENGTH)) {
                m_RunLengths.push_back(MAX_RUN_LENGTH);
                length -= static_cast<std::size_t>(MAX_RUN_LENGTH);
            }
        } else {
            m_Parity = !m_Parity;
            m_RunLengths.push_back(static_cast<uint8_t>(length));
            length = 1;
        }
    }
    m_RunLengths.push_back(static_cast<uint8_t>(length));
}

void CPackedBitVector::contract() {
    if (m_Dimension == 0) {
        return;
    }

    if (--m_Dimension == 0) {
        m_First = false;
        m_Parity = true;
        m_RunLengths.clear();
        return;
    }

    if (m_RunLengths.front() == MAX_RUN_LENGTH) {
        std::size_t i = 1u;
        for (/**/; m_RunLengths[i] == MAX_RUN_LENGTH && i < m_RunLengths.size(); ++i) {
        }
        if (m_RunLengths[i] == 0) {
            m_RunLengths.erase(m_RunLengths.begin() + i);
            --m_RunLengths[i - 1];
        } else {
            --m_RunLengths[i];
        }
    } else if (--m_RunLengths.front() == 0) {
        m_First = !m_First;
        m_Parity = !m_Parity;
        m_RunLengths.erase(m_RunLengths.begin());
    }
}

void CPackedBitVector::extend(bool bit) {
    ++m_Dimension;

    if (m_Dimension == 1) {
        m_First = bit;
        m_Parity = true;
        m_RunLengths.push_back(1);
    } else if (m_Parity ? (bit != m_First) : (bit == m_First)) {
        m_Parity = !m_Parity;
        m_RunLengths.push_back(1);
    } else if (m_RunLengths.back() + 1 == MAX_RUN_LENGTH) {
        ++m_RunLengths.back();
        m_RunLengths.push_back(0);
    } else {
        ++m_RunLengths.back();
    }
}

bool CPackedBitVector::fromDelimited(const std::string& str) {
    std::size_t last = 0u;
    std::size_t pos = str.find_first_of(CPersistUtils::DELIMITER, last);
    if (pos == std::string::npos ||
        CStringUtils::stringToType(str.substr(last, pos - last), m_Dimension) == false) {
        LOG_ERROR(<< "Invalid packed vector in " << str);
        return false;
    }

    last = pos;
    pos = str.find_first_of(CPersistUtils::DELIMITER, last + 1);
    int first = 0;
    if (pos == std::string::npos ||
        CStringUtils::stringToType(str.substr(last + 1, pos - last - 1), first) == false) {
        LOG_ERROR(<< "Invalid packed vector in " << str);
        return false;
    }
    m_First = (first != 0);

    last = pos;
    pos = str.find_first_of(CPersistUtils::DELIMITER, last + 1);
    int parity = 0;
    if (pos == std::string::npos ||
        CStringUtils::stringToType(str.substr(last + 1, pos - last - 1), parity) == false) {
        LOG_ERROR(<< "Invalid packed vector in " << str);
        return false;
    }
    m_Parity = (parity != 0);

    if (CPersistUtils::fromString(str.substr(pos + 1), m_RunLengths) == false) {
        LOG_ERROR(<< "Invalid packed vector in " << str);
        return false;
    }

    return true;
}

std::string CPackedBitVector::toDelimited() const {
    std::string result;
    result += CStringUtils::typeToString(m_Dimension) + CPersistUtils::DELIMITER;
    result += CStringUtils::typeToString(static_cast<int>(m_First)) + CPersistUtils::DELIMITER;
    result += CStringUtils::typeToString(static_cast<int>(m_Parity)) + CPersistUtils::DELIMITER;
    result += CPersistUtils::toString(m_RunLengths);
    return result;
}

std::size_t CPackedBitVector::dimension() const {
    return m_Dimension;
}

bool CPackedBitVector::operator()(std::size_t i) const {
    bool parity = true;
    for (std::size_t j = 0u, k = static_cast<std::size_t>(m_RunLengths[j]);
         k <= i; k += static_cast<std::size_t>(m_RunLengths[++j])) {
        if (m_RunLengths[j] != MAX_RUN_LENGTH) {
            parity = !parity;
        }
    }
    return parity ? m_First : !m_First;
}

bool CPackedBitVector::operator==(const CPackedBitVector& other) const {
    return m_Dimension == other.m_Dimension && m_First == other.m_First &&
           m_Parity == other.m_Parity && m_RunLengths == other.m_RunLengths;
}

bool CPackedBitVector::operator<(const CPackedBitVector& rhs) const {
#define LESS_OR_GREATER(a, b) if (a < b) { return true; } else if (b < a) { return false; }
    LESS_OR_GREATER(m_Dimension, rhs.m_Dimension)
    LESS_OR_GREATER(m_First, rhs.m_First)
    LESS_OR_GREATER(m_Parity, rhs.m_Parity)
    LESS_OR_GREATER(m_RunLengths, rhs.m_RunLengths)
    return false;
}

CPackedBitVector CPackedBitVector::complement() const {
    CPackedBitVector result(*this);
    result.m_First = !result.m_First;
    return result;
}

double CPackedBitVector::inner(const CPackedBitVector& covector, EOperation op) const {
    // This is just a line scan over the run lengths keeping
    // track of the parities of both vectors.

    double result = 0.0;

    if (m_Dimension != covector.dimension()) {
        LOG_ERROR(<< "Dimension mismatch " << m_Dimension << " vs " << covector.dimension());
        return result;
    }

    int value = static_cast<int>(m_First);
    int covalue = static_cast<int>(covector.m_First);
    std::size_t length = static_cast<std::size_t>(m_RunLengths[0]);
    std::size_t colength = static_cast<std::size_t>(covector.m_RunLengths[0]);
    std::size_t pos = length;
    std::size_t copos = colength;

    for (std::size_t i = 0u, j = 0u; pos < m_Dimension || copos < m_Dimension;
         /**/) {
        std::size_t run = std::min(pos, copos) - std::max(pos - length, copos - colength);
        switch (op) {
        case E_AND:
            result += static_cast<double>((value & covalue) * run);
            break;
        case E_OR:
            result += static_cast<double>((value | covalue) * run);
            break;
        case E_XOR:
            result += static_cast<double>((value ^ covalue) * run);
            break;
        }

        if (pos < copos) {
            if (length != MAX_RUN_LENGTH) {
                value = 1 - value;
            }
            length = static_cast<std::size_t>(m_RunLengths[++i]);
            pos += length;
        } else if (copos < pos) {
            if (colength != MAX_RUN_LENGTH) {
                covalue = 1 - covalue;
            }
            colength = static_cast<std::size_t>(covector.m_RunLengths[++j]);
            copos += colength;
        } else {
            if (length != MAX_RUN_LENGTH) {
                value = 1 - value;
                covalue = 1 - covalue;
            }
            length = static_cast<std::size_t>(m_RunLengths[++i]);
            colength = static_cast<std::size_t>(covector.m_RunLengths[++j]);
            pos += length;
            copos += colength;
        }
    }

    std::size_t run = std::min(length, colength);
    switch (op) {
    case E_AND:
        result += static_cast<double>((value & covalue) * run);
        break;
    case E_OR:
        result += static_cast<double>((value | covalue) * run);
        break;
    case E_XOR:
        result += static_cast<double>((value ^ covalue) * run);
        break;
    }

    return result;
}

CPackedBitVector::TBoolVec CPackedBitVector::toBitVector() const {
    if (m_Dimension == 0) {
        return TBoolVec();
    }

    TBoolVec result;
    result.reserve(m_Dimension);

    bool parity = true;
    for (std::size_t i = 0u; i < m_RunLengths.size(); ++i) {
        std::fill_n(std::back_inserter(result),
                    static_cast<std::size_t>(m_RunLengths[i]), parity ? m_First : !m_First);
        if (m_RunLengths[i] != MAX_RUN_LENGTH) {
            parity = !parity;
        }
    }

    return result;
}

uint64_t CPackedBitVector::checksum() const {
    std::uint64_t seed = m_Dimension;
    seed = CHashing::hashCombine(seed, static_cast<std::uint64_t>(m_First));
    seed = CHashing::hashCombine(seed, static_cast<std::uint64_t>(m_Parity));
    return CHashing::murmurHash64(
        m_RunLengths.data(),
        static_cast<int>(sizeof(std::uint8_t) * m_RunLengths.size()), seed);
}

void CPackedBitVector::debugMemoryUsage(CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CPackedBitVector");
    CMemoryDebug::dynamicSize("m_RunLengths", m_RunLengths, mem);
}

std::size_t CPackedBitVector::memoryUsage() const {
    return CMemory::dynamicSize(m_RunLengths);
}

const uint8_t CPackedBitVector::MAX_RUN_LENGTH = std::numeric_limits<uint8_t>::max();

std::ostream& operator<<(std::ostream& o, const CPackedBitVector& v) {
    if (v.dimension() == 0) {
        return o << "[]";
    }

    o << '[' << CStringUtils::typeToString(static_cast<int>(v(0)));
    for (std::size_t i = 1u; i < v.dimension(); ++i) {
        o << ' ' << CStringUtils::typeToString(static_cast<int>(v(i)));
    }
    o << ']';

    return o;
}
}
}
