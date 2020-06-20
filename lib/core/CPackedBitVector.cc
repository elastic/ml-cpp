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
    : m_Dimension{0}, m_First{false}, m_Parity{true} {
}

CPackedBitVector::CPackedBitVector(bool bit)
    : m_Dimension{1}, m_First{bit}, m_Parity{true}, m_LastRunBytes{1},
      m_RunLengthBytes(1, 1) {
}

CPackedBitVector::CPackedBitVector(std::size_t dimension, bool bit)
    : m_Dimension{dimension}, m_First{bit}, m_Parity{true} {
    if (dimension > 0) {
        appendRun(dimension, m_LastRunBytes, m_RunLengthBytes);
    }
}

CPackedBitVector::CPackedBitVector(const TBoolVec& bits)
    : m_Dimension{bits.size()}, m_First{bits.empty() ? false : bits[0]}, m_Parity{true} {
    std::size_t run{1};
    for (std::size_t i = 1; i < bits.size(); ++i, ++run) {
        if (bits[i] != bits[i - 1]) {
            m_Parity = !m_Parity;
            appendRun(run, m_LastRunBytes, m_RunLengthBytes);
            run = 0;
        }
    }
    appendRun(run, m_LastRunBytes, m_RunLengthBytes);
}

void CPackedBitVector::contract() {
    if (m_Dimension == 0) {
        return;
    }

    if (--m_Dimension == 0) {
        m_First = false;
        m_Parity = true;
        m_RunLengthBytes.clear();
        return;
    }

    std::size_t firstRunLength{readRunLength(m_RunLengthBytes.begin())};

    if (firstRunLength == 1) {
        m_First = !m_First;
        m_Parity = !m_Parity;
        m_RunLengthBytes.erase(m_RunLengthBytes.begin());
    } else {
        std::uint8_t firstRunBytes{bytes(firstRunLength)};
        std::uint8_t contractedFirstRunBytes{bytes(firstRunLength - 1)};
        m_RunLengthBytes.erase(m_RunLengthBytes.begin(),
                               m_RunLengthBytes.begin() + firstRunBytes - contractedFirstRunBytes);
        writeRunLength(firstRunLength - 1, m_RunLengthBytes.begin());
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
        appendRun(n, m_LastRunBytes, m_RunLengthBytes);
    } else if (m_Parity ? (bit != m_First) : (bit == m_First)) {
        m_Parity = !m_Parity;
        appendRun(n, m_LastRunBytes, m_RunLengthBytes);
    } else {
        extendLastRun(n, m_LastRunBytes, m_RunLengthBytes);
    }
}

bool CPackedBitVector::fromDelimited(const std::string& str) {
    std::size_t last{0};
    std::size_t pos{str.find_first_of(CPersistUtils::DELIMITER, last)};
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
    int parity{0};
    if (pos == std::string::npos ||
        CStringUtils::stringToType(str.substr(last + 1, pos - last - 1), parity) == false) {
        LOG_ERROR(<< "Invalid packed vector in " << str);
        return false;
    }
    m_Parity = (parity != 0);

    last = pos;
    pos = str.find_first_of(CPersistUtils::DELIMITER, last + 1);
    int lastRunBytes{0};
    if (pos == std::string::npos ||
        CStringUtils::stringToType(str.substr(last + 1, pos - last - 1), lastRunBytes) == false) {
        LOG_ERROR(<< "Invalid packed vector in " << str);
        return false;
    }
    m_LastRunBytes = static_cast<std::uint8_t>(lastRunBytes);

    if (CPersistUtils::fromString(str.substr(pos + 1), m_RunLengthBytes) == false) {
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
    result += CStringUtils::typeToString(static_cast<int>(m_LastRunBytes)) +
              CPersistUtils::DELIMITER;
    result += CPersistUtils::toString(m_RunLengthBytes);
    return result;
}

void CPackedBitVector::clear() {
    m_Dimension = 0;
    m_First = false;
    m_Parity = true;
    m_LastRunBytes = 0;
    m_RunLengthBytes.clear();
}

bool CPackedBitVector::operator==(const CPackedBitVector& other) const {
    return m_Dimension == other.m_Dimension && m_First == other.m_First &&
           m_Parity == other.m_Parity && m_LastRunBytes == other.m_LastRunBytes &&
           m_RunLengthBytes == other.m_RunLengthBytes;
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
    LESS_OR_GREATER(m_LastRunBytes, rhs.m_LastRunBytes)
    LESS_OR_GREATER(m_RunLengthBytes, rhs.m_RunLengthBytes)
    return false;
}

CPackedBitVector CPackedBitVector::operator~() const {
    CPackedBitVector result{*this};
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
    return {m_First, m_RunLengthBytes.begin(), m_RunLengthBytes.end()};
}

CPackedBitVector::COneBitIndexConstIterator CPackedBitVector::endOneBits() const {
    return {m_Dimension, m_RunLengthBytes.end()};
}

std::size_t CPackedBitVector::dimension() const {
    return m_Dimension;
}

bool CPackedBitVector::operator()(std::size_t i) const {
    bool parity{true};
    auto itr = m_RunLengthBytes.begin();
    for (std::size_t j = popRunLength(itr);
         j <= i && itr != m_RunLengthBytes.end(); j += popRunLength(itr)) {
        parity = !parity;
    }
    return parity ? m_First : !m_First;
}

double CPackedBitVector::inner(const CPackedBitVector& covector, EOperation op) const {
    std::size_t result{0};
    this->lineScan(covector, [op, &result](int value, int covalue, std::size_t run) {
        result += static_cast<std::size_t>(bit(op, value, covalue)) * run;
    });
    return static_cast<double>(result);
}

CPackedBitVector::TBoolVec CPackedBitVector::toBitVector() const {
    if (m_Dimension == 0) {
        return {};
    }

    TBoolVec result;
    result.reserve(m_Dimension);

    bool parity{true};
    for (auto itr = m_RunLengthBytes.begin(); itr != m_RunLengthBytes.end(); /**/) {
        std::fill_n(std::back_inserter(result), popRunLength(itr),
                    parity ? m_First : !m_First);
        parity = !parity;
    }

    return result;
}

std::uint64_t CPackedBitVector::checksum() const {
    std::uint64_t seed{m_Dimension};
    seed = CHashing::hashCombine(seed, static_cast<std::uint64_t>(m_LastRunBytes));
    seed = CHashing::hashCombine(seed, static_cast<std::uint64_t>(m_First));
    seed = CHashing::hashCombine(seed, static_cast<std::uint64_t>(m_Parity));
    return CHashing::murmurHash64(
        m_RunLengthBytes.data(),
        static_cast<int>(sizeof(std::uint8_t) * m_RunLengthBytes.size()), seed);
}

void CPackedBitVector::debugMemoryUsage(const CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CPackedBitVector");
    CMemoryDebug::dynamicSize("m_RunLengths", m_RunLengthBytes, mem);
}

std::size_t CPackedBitVector::memoryUsage() const {
    return CMemory::dynamicSize(m_RunLengthBytes);
}

void CPackedBitVector::bitwise(EOperation op, const CPackedBitVector& other) {

    bool first{bit(op, m_First, other.m_First)};
    bool parity{true};
    std::uint8_t lastRunBytes{0};
    TUInt8Vec runLengthBytes;
    runLengthBytes.reserve(other.m_RunLengthBytes.size() + m_RunLengthBytes.size());

    int last{static_cast<int>(first)};
    std::size_t cumulativeRun{0};

    auto bitwiseOp = [&](int value, int covalue, std::size_t run) mutable {
        value = bit(op, value, covalue);
        if (last != value) {
            parity = !parity;
            appendRun(cumulativeRun, lastRunBytes, runLengthBytes);
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

        m_LastRunBytes = lastRunBytes;
        m_RunLengthBytes = std::move(runLengthBytes);
    }
}

template<typename RUN_OP>
bool CPackedBitVector::lineScan(const CPackedBitVector& covector, RUN_OP op) const {
    // This is just a line scan over the run lengths keeping track of the
    // parities of both vectors.

    if (m_Dimension != covector.dimension()) {
        LOG_ERROR(<< "Dimension mismatch " << m_Dimension << " vs " << covector.dimension());
        return false;
    }

    int value{static_cast<int>(m_First)};
    int covalue{static_cast<int>(covector.m_First)};
    auto itr = m_RunLengthBytes.begin();
    auto coitr = covector.m_RunLengthBytes.begin();
    std::size_t run{popRunLength(itr)};
    std::size_t corun{popRunLength(coitr)};
    std::size_t pos{static_cast<std::size_t>(run)};
    std::size_t copos{static_cast<std::size_t>(corun)};

    while (pos < m_Dimension || copos < m_Dimension) {

        std::size_t step{std::min(pos, copos) - std::max(pos - run, copos - corun)};
        op(value, covalue, step);

        if (pos < copos) {
            value = 1 - value;
            run = popRunLength(itr);
            pos += run;
        } else if (copos < pos) {
            covalue = 1 - covalue;
            corun = popRunLength(coitr);
            copos += corun;
        } else {
            value = 1 - value;
            covalue = 1 - covalue;
            run = popRunLength(itr);
            corun = popRunLength(coitr);
            pos += run;
            copos += corun;
        }
    }

    std::size_t step{static_cast<std::size_t>(std::min(run, corun))};
    op(value, covalue, step);

    return true;
}

void CPackedBitVector::appendRun(std::size_t runLength,
                                 std::uint8_t& lastRunBytes,
                                 TUInt8Vec& runLengthBytes) {
    lastRunBytes = bytes(runLength);
    runLengthBytes.resize(runLengthBytes.size() + lastRunBytes);
    writeRunLength(runLength, runLengthBytes.end() - lastRunBytes);
}

void CPackedBitVector::extendLastRun(std::size_t runLength,
                                     std::uint8_t& lastRunBytes,
                                     TUInt8Vec& runLengthBytes) {
    std::size_t lastRunLength{readLastRunLength(lastRunBytes, runLengthBytes) + runLength};
    std::uint8_t extendedLastRunBytes{bytes(lastRunLength)};
    runLengthBytes.resize(runLengthBytes.size() + (extendedLastRunBytes - lastRunBytes));
    lastRunBytes = extendedLastRunBytes;
    writeRunLength(lastRunLength, runLengthBytes.end() - lastRunBytes);
}

std::uint8_t CPackedBitVector::bytes(std::size_t runLength) {
    if (runLength <= MAXIMUM_ONE_BYTE_RUN_LENGTH) {
        return 1;
    }
    if (runLength <= MAXIMUM_TWO_BYTE_RUN_LENGTH) {
        return 2;
    }
    if (runLength <= MAXIMUM_THREE_BYTE_RUN_LENGTH) {
        return 3;
    }
    return 4;
}

std::size_t CPackedBitVector::readLastRunLength(std::uint8_t lastRunBytes,
                                                const TUInt8Vec& runLengths) {
    return readRunLength(runLengths.end() - static_cast<std::ptrdiff_t>(lastRunBytes));
}

std::size_t CPackedBitVector::popRunLength(TUInt8VecCItr& runLengthBytes) {
    int bytes{(NUMBER_BYTES_MASK & *runLengthBytes) + 1};
    std::size_t result{static_cast<std::size_t>(*runLengthBytes >> NUMBER_BYTES_MASK_BITS)};
    ++runLengthBytes;
    for (int i = 1, scale = 64; i < bytes; ++i, scale *= 256) {
        result += scale * static_cast<std::size_t>(*runLengthBytes);
        ++runLengthBytes;
    }
    return result;
}

void CPackedBitVector::writeRunLength(std::size_t runLength, TUInt8VecItr runLengthBytes) {
    std::size_t lowestBits{runLength & ((0xFF ^ NUMBER_BYTES_MASK) >> NUMBER_BYTES_MASK_BITS)};
    *runLengthBytes = (bytes(runLength) - 1) +
                      static_cast<std::uint8_t>(lowestBits << NUMBER_BYTES_MASK_BITS);
    ++runLengthBytes;
    for (runLength /= 256 >> NUMBER_BYTES_MASK_BITS; runLength > 0; runLength /= 256) {
        *runLengthBytes = static_cast<std::uint8_t>(runLength & 0xFF);
        ++runLengthBytes;
    }
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

CPackedBitVector::COneBitIndexConstIterator::COneBitIndexConstIterator(bool first,
                                                                       TUInt8VecCItr runLengthsItr,
                                                                       TUInt8VecCItr endRunLengthsItr)
    : m_RunLengthsItr{runLengthsItr}, m_EndRunLengthsItr{endRunLengthsItr} {
    if (first) {
        m_Current = 0;
        m_EndOfCurrentRun = popRunLength(m_RunLengthsItr);
    } else {
        this->skipRun();
    }
}

CPackedBitVector::COneBitIndexConstIterator::COneBitIndexConstIterator(std::size_t size, TUInt8VecCItr endRunLengthsItr)
    : m_Current{size}, m_EndOfCurrentRun{size},
      m_RunLengthsItr{endRunLengthsItr}, m_EndRunLengthsItr{endRunLengthsItr} {
}

void CPackedBitVector::COneBitIndexConstIterator::skipRun() {
    if (m_RunLengthsItr == m_EndRunLengthsItr) {
        return;
    }
    std::size_t skip{popRunLength(m_RunLengthsItr)};
    m_Current += skip;
    m_EndOfCurrentRun = m_Current;
    if (m_RunLengthsItr != m_EndRunLengthsItr) {
        m_EndOfCurrentRun += popRunLength(m_RunLengthsItr);
    }
}

std::ostream& operator<<(std::ostream& o, const CPackedBitVector& v) {
    if (v.dimension() == 0) {
        return o << "[]";
    }

    o << '[' << CStringUtils::typeToString(static_cast<int>(v(0)));
    for (std::size_t i = 1; i < v.dimension(); ++i) {
        o << ' ' << CStringUtils::typeToString(static_cast<int>(v(i)));
    }
    o << ']';

    return o;
}
}
}
