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
        std::size_t remainder{static_cast<std::size_t>(MAX_RUN_LENGTH)};
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
    std::size_t length{1};
    for (std::size_t i = 1; i < bits.size(); ++i) {
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
        std::size_t i{1};
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

void CPackedBitVector::extend(bool bit, std::size_t n) {
    if (n == 0) {
        return;
    }

    m_Dimension += n;

    if (m_Dimension == n) {
        m_First = bit;
        m_Parity = true;
        appendRun(n, m_RunLengths);
    } else if (m_Parity ? (bit != m_First) : (bit == m_First)) {
        m_Parity = !m_Parity;
        appendNewRun(n, m_RunLengths);
    } else if (m_RunLengths.back() + n < MAX_RUN_LENGTH) {
        m_RunLengths.back() += n;
    } else {
        n -= (MAX_RUN_LENGTH - m_RunLengths.back());
        m_RunLengths.back() = MAX_RUN_LENGTH;
        appendRun(n, m_RunLengths);
    }
}

bool CPackedBitVector::fromDelimited(const std::string& str) {
    std::size_t last{0};
    std::size_t pos = str.find_first_of(CPersistUtils::DELIMITER, last);
    if (pos == std::string::npos ||
        CStringUtils::stringToType(str.substr(last, pos - last), m_Dimension) == false) {
        LOG_ERROR(<< "Invalid packed vector in " << str);
        return false;
    }

    last = pos;
    pos = str.find_first_of(CPersistUtils::DELIMITER, last + 1);
    int first{0};
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

bool CPackedBitVector::operator==(const CPackedBitVector& other) const {
    return m_Dimension == other.m_Dimension && m_First == other.m_First &&
           m_Parity == other.m_Parity && m_RunLengths == other.m_RunLengths;
}

bool CPackedBitVector::operator<(const CPackedBitVector& rhs) const {
#define LESS_OR_GREATER(a, b)                                                  \
    if (a < b) {                                                               \
        return true;                                                           \
    } else if (b < a) {                                                        \
        return false;                                                          \
    }
    LESS_OR_GREATER(m_Dimension, rhs.m_Dimension)
    LESS_OR_GREATER(m_First, rhs.m_First)
    LESS_OR_GREATER(m_Parity, rhs.m_Parity)
    LESS_OR_GREATER(m_RunLengths, rhs.m_RunLengths)
    return false;
}

CPackedBitVector CPackedBitVector::operator~() const {
    CPackedBitVector result(*this);
    result.m_First = !result.m_First;
    return result;
}

const CPackedBitVector& CPackedBitVector::operator&=(const CPackedBitVector& other) {
    this->bitwise(E_AND, other);
    return *this;
}

const CPackedBitVector& CPackedBitVector::operator|=(const CPackedBitVector& other) {
    this->bitwise(E_OR, other);
    return *this;
}

const CPackedBitVector& CPackedBitVector::operator^=(const CPackedBitVector& other) {
    this->bitwise(E_XOR, other);
    return *this;
}

CPackedBitVector::COneBitIndexConstIterator CPackedBitVector::beginOneBits() const {
    return {m_First, m_RunLengths.begin(), m_RunLengths.end()};
}

CPackedBitVector::COneBitIndexConstIterator CPackedBitVector::endOneBits() const {
    return {m_Dimension, m_RunLengths.end()};
}

std::size_t CPackedBitVector::dimension() const {
    return m_Dimension;
}

bool CPackedBitVector::operator()(std::size_t i) const {
    bool parity{true};
    for (std::size_t j = 0, k = static_cast<std::size_t>(m_RunLengths[j]);
         k <= i; k += static_cast<std::size_t>(m_RunLengths[++j])) {
        if (m_RunLengths[j] != MAX_RUN_LENGTH) {
            parity = !parity;
        }
    }
    return parity ? m_First : !m_First;
}

double CPackedBitVector::inner(const CPackedBitVector& covector, EOperation op) const {

    double result{0.0};

    this->lineScan(covector, [op, &result](int value, int covalue, std::size_t run) {
        result += static_cast<double>(bit(op, value, covalue)) * run;
    });

    return result;
}

CPackedBitVector::TBoolVec CPackedBitVector::toBitVector() const {
    if (m_Dimension == 0) {
        return TBoolVec();
    }

    TBoolVec result;
    result.reserve(m_Dimension);

    bool parity = true;
    for (std::size_t i = 0; i < m_RunLengths.size(); ++i) {
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

void CPackedBitVector::bitwise(EOperation op, const CPackedBitVector& other) {

    bool first{bit(op, m_First, other.m_First)};
    bool parity{true};
    TUInt8Vec runLengths;

    int last{static_cast<int>(first)};
    std::size_t cumulativeRun{0};

    auto bitwiseOp = [&](int value, int covalue, std::size_t run) mutable {
        value = bit(op, value, covalue);
        if (last != value) {
            parity = !parity;
            appendNewRun(cumulativeRun, runLengths);
            last = 1 - last;
            cumulativeRun = run;
        } else {
            cumulativeRun += run;
        }
    };

    if (this->lineScan(other, bitwiseOp)) {

        m_First = first;
        m_Parity = parity;

        // Flush the last value.
        switch (op) {
        case E_AND:
            last == 0 ? bitwiseOp(1, 1, 0) : bitwiseOp(0, 0, 0);
            break;
        case E_OR:
            last == 0 ? bitwiseOp(1, 1, 0) : bitwiseOp(0, 0, 0);
            break;
        case E_XOR:
            last == 0 ? bitwiseOp(1, 0, 0) : bitwiseOp(0, 0, 0);
            break;
        }

        runLengths.shrink_to_fit();
        m_RunLengths = std::move(runLengths);
    }
}
template<typename RUN_ACTION>
bool CPackedBitVector::lineScan(const CPackedBitVector& covector, RUN_ACTION action) const {
    // This is just a line scan over the run lengths keeping track of the
    // parities of both vectors.

    if (m_Dimension != covector.dimension()) {
        LOG_ERROR(<< "Dimension mismatch " << m_Dimension << " vs " << covector.dimension());
        return false;
    }

    int value{static_cast<int>(m_First)};
    int covalue{static_cast<int>(covector.m_First)};
    std::size_t length{static_cast<std::size_t>(m_RunLengths[0])};
    std::size_t colength{static_cast<std::size_t>(covector.m_RunLengths[0])};
    std::size_t pos{length};
    std::size_t copos{colength};

    for (std::size_t i = 0, j = 0; pos < m_Dimension || copos < m_Dimension; /**/) {

        std::size_t run{std::min(pos, copos) - std::max(pos - length, copos - colength)};
        action(value, covalue, run);

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

    std::size_t run{std::min(length, colength)};
    action(value, covalue, run);

    return true;
}

void CPackedBitVector::appendNewRun(std::size_t run, TUInt8Vec& runLengths) {
    if (runLengths.size() > 0 && runLengths.back() == MAX_RUN_LENGTH) {
        runLengths.push_back(0);
    }
    appendRun(run, runLengths);
}

void CPackedBitVector::appendRun(std::size_t run, TUInt8Vec& runLengths) {
    for (/**/; run > MAX_RUN_LENGTH; run -= MAX_RUN_LENGTH) {
        runLengths.push_back(MAX_RUN_LENGTH);
    }
    runLengths.push_back(static_cast<std::uint8_t>(run));
}

const std::uint8_t CPackedBitVector::MAX_RUN_LENGTH =
    std::numeric_limits<std::uint8_t>::max();

CPackedBitVector::COneBitIndexConstIterator::COneBitIndexConstIterator(bool first,
                                                                       TUInt8VecCItr runLengthsItr,
                                                                       TUInt8VecCItr endRunLengthsItr)
    : m_RunLengthsItr{runLengthsItr}, m_EndRunLengthsItr{endRunLengthsItr} {
    if (first) {
        m_Current = 0;
        m_EndOfCurrentRun = this->advanceToEndOfRun();
    } else {
        this->skipRun();
    }
}

CPackedBitVector::COneBitIndexConstIterator::COneBitIndexConstIterator(std::size_t size, TUInt8VecCItr endRunLengthsItr)
    : m_Current{size}, m_EndOfCurrentRun{size},
      m_RunLengthsItr{endRunLengthsItr}, m_EndRunLengthsItr{endRunLengthsItr} {
}

void CPackedBitVector::COneBitIndexConstIterator::skipRun() {
    std::size_t skip{this->advanceToEndOfRun()};
    m_Current += skip;
    m_EndOfCurrentRun += skip + this->advanceToEndOfRun();
}

std::size_t CPackedBitVector::COneBitIndexConstIterator::advanceToEndOfRun() {
    std::size_t run{0};
    do {
        if (m_RunLengthsItr != m_EndRunLengthsItr) {
            run += *m_RunLengthsItr;
        } else {
            break;
        }
    } while (*m_RunLengthsItr++ == MAX_RUN_LENGTH);
    return run;
}

template<typename T>
T CPackedBitVector::bit(EOperation op, T lhs, T rhs) {
    T result{lhs};
    switch (op) {
    case E_AND:
        result &= rhs;
        break;
    case E_OR:
        result |= rhs;
        break;
    case E_XOR:
        result ^= rhs;
        break;
    }
    return result;
}

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
