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
namespace {
std::size_t read(std::uint8_t run) {
    return static_cast<std::size_t>(run == 0 ? CPackedBitVector::MAX_RUN_LENGTH : run);
}

std::uint8_t write(std::size_t run) {
    return static_cast<std::uint8_t>(run == CPackedBitVector::MAX_RUN_LENGTH ? 0 : run);
}

bool complete(std::uint8_t run) {
    return run != CPackedBitVector::MAX_RUN_LENGTH;
}
}

CPackedBitVector::CPackedBitVector()
    : m_Dimension(0), m_First(false), m_Parity(true) {
}

CPackedBitVector::CPackedBitVector(bool bit)
    : m_Dimension(1), m_First(bit), m_Parity(true), m_RunLengths(1, 1) {
}

CPackedBitVector::CPackedBitVector(std::size_t dimension, bool bit)
    : m_Dimension(static_cast<std::uint32_t>(dimension)), m_First(bit), m_Parity(true) {
    if (dimension > 0) {
        appendRun(dimension, m_RunLengths);
    }
}

CPackedBitVector::CPackedBitVector(const TBoolVec& bits)
    : m_Dimension(static_cast<std::uint32_t>(bits.size())),
      m_First(bits.empty() ? false : bits[0]), m_Parity(true) {
    std::size_t run{1};
    for (std::size_t i = 1; i < bits.size(); ++i, ++run) {
        if (bits[i] != bits[i - 1]) {
            m_Parity = !m_Parity;
            appendRun(run, m_RunLengths);
            run = 0;
        }
    }
    appendRun(run, m_RunLengths);
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

    auto front = std::find_if(m_RunLengths.begin(), m_RunLengths.end(),
                              [](std::uint8_t run) { return complete(run); });

    if (*front == 0) {
        *front = MAX_RUN_LENGTH - 1;
    } else if (--(*front) == 0) {
        if (front == m_RunLengths.begin()) {
            m_First = !m_First;
            m_Parity = !m_Parity;
        } else {
            *(front - 1) = 0;
        }
        m_RunLengths.erase(front);
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
        appendRun(n, m_RunLengths);
    } else {
        extendRun(n, m_RunLengths);
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
    for (std::size_t j = 0, k = read(m_RunLengths[j]); k <= i;
         k += read(m_RunLengths[++j])) {
        if (complete(m_RunLengths[j])) {
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
        std::fill_n(std::back_inserter(result), read(m_RunLengths[i]),
                    parity ? m_First : !m_First);
        if (complete(m_RunLengths[i])) {
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
            appendRun(cumulativeRun, runLengths);
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
    std::uint8_t run{m_RunLengths[0]};
    std::uint8_t corun{covector.m_RunLengths[0]};
    std::size_t pos{read(run)};
    std::size_t copos{read(corun)};

    for (std::size_t i = 0, j = 0; pos < m_Dimension || copos < m_Dimension; /**/) {

        std::size_t step{std::min(pos, copos) -
                         std::max(pos - read(run), copos - read(corun))};
        action(value, covalue, step);

        if (pos < copos) {
            if (complete(run)) {
                value = 1 - value;
            }
            run = m_RunLengths[++i];
            pos += read(run);
        } else if (copos < pos) {
            if (complete(corun)) {
                covalue = 1 - covalue;
            }
            corun = covector.m_RunLengths[++j];
            copos += read(corun);
        } else {
            if (complete(run)) {
                value = 1 - value;
            }
            if (complete(corun)) {
                covalue = 1 - covalue;
            }
            run = m_RunLengths[++i];
            corun = covector.m_RunLengths[++j];
            pos += read(run);
            copos += read(corun);
        }
    }

    std::size_t step{std::min(read(run), read(corun))};
    action(value, covalue, step);

    return true;
}

void CPackedBitVector::appendRun(std::size_t run, TUInt8Vec& runLengths) {
    for (/**/; run > MAX_RUN_LENGTH; run -= MAX_RUN_LENGTH) {
        runLengths.push_back(MAX_RUN_LENGTH);
    }
    runLengths.push_back(write(run));
}

void CPackedBitVector::extendRun(std::size_t run, TUInt8Vec& runLengths) {
    if (runLengths.back() == 0) {
        runLengths.back() = MAX_RUN_LENGTH;
    }
    if (runLengths.back() + run < MAX_RUN_LENGTH) {
        runLengths.back() += run;
    } else if (runLengths.back() + run == MAX_RUN_LENGTH) {
        runLengths.back() = 0;
    } else {
        run -= (MAX_RUN_LENGTH - runLengths.back());
        runLengths.back() = MAX_RUN_LENGTH;
        appendRun(run, runLengths);
    }
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
    while (m_RunLengthsItr != m_EndRunLengthsItr) {
        run += read(*m_RunLengthsItr);
        if (complete(*m_RunLengthsItr++)) {
            break;
        }
    }
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
    for (std::size_t i = 1; i < v.dimension(); ++i) {
        o << ' ' << CStringUtils::typeToString(static_cast<int>(v(i)));
    }
    o << ']';

    return o;
}
}
}
