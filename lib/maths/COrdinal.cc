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

#include <maths/COrdinal.h>

#include <boost/numeric/conversion/bounds.hpp>

#include <limits>
#include <math.h>
#include <ostream>
#include <stdint.h>

namespace ml {
namespace maths {

COrdinal::COrdinal(void) : m_Type(E_Nan) {
    m_Value.integer = 0;
}

COrdinal::COrdinal(int64_t value) : m_Type(E_Integer) {
    m_Value.integer = value;
}

COrdinal::COrdinal(uint64_t value) : m_Type(E_PositiveInteger) {
    m_Value.positiveInteger = value;
}

COrdinal::COrdinal(double value) : m_Type(E_Real) {
    m_Value.real = value;
}

bool COrdinal::operator==(COrdinal rhs) const {
    switch (m_Type) {
        case E_Integer:
            switch (rhs.m_Type) {
                case E_Integer:
                    return m_Value.integer == rhs.m_Value.integer;
                case E_PositiveInteger:
                    return this->equal(m_Value.integer, rhs.m_Value.positiveInteger);
                case E_Real:
                    return this->equal(m_Value.integer, rhs.m_Value.real);
                case E_Nan:
                    break;
            }
            break;
        case E_PositiveInteger:
            switch (rhs.m_Type) {
                case E_Integer:
                    return this->equal(rhs.m_Value.integer, m_Value.positiveInteger);
                case E_PositiveInteger:
                    return m_Value.positiveInteger == rhs.m_Value.positiveInteger;
                case E_Real:
                    return this->equal(m_Value.positiveInteger, rhs.m_Value.real);
                case E_Nan:
                    break;
            }
            break;
        case E_Real:
            switch (rhs.m_Type) {
                case E_Integer:
                    return this->equal(rhs.m_Value.integer, m_Value.real);
                case E_PositiveInteger:
                    return this->equal(rhs.m_Value.positiveInteger, m_Value.real);
                case E_Real:
                    return m_Value.real == rhs.m_Value.real;
                case E_Nan:
                    break;
            }
            break;
        case E_Nan:
            break;
    }
    return false;
}

bool COrdinal::operator<(COrdinal rhs) const {
    switch (m_Type) {
        case E_Integer:
            switch (rhs.m_Type) {
                case E_Integer:
                    return m_Value.integer < rhs.m_Value.integer;
                case E_PositiveInteger:
                    return this->less(m_Value.integer, rhs.m_Value.positiveInteger);
                case E_Real:
                    return this->less(m_Value.integer, rhs.m_Value.real);
                case E_Nan:
                    break;
            }
            break;
        case E_PositiveInteger:
            switch (rhs.m_Type) {
                case E_Integer:
                    return    !this->equal(rhs.m_Value.integer, m_Value.positiveInteger)
                              && !this->less(rhs.m_Value.integer, m_Value.positiveInteger);
                case E_PositiveInteger:
                    return m_Value.positiveInteger < rhs.m_Value.positiveInteger;
                case E_Real:
                    return this->less(m_Value.positiveInteger, rhs.m_Value.real);
                case E_Nan:
                    break;
            }
            break;
        case E_Real:
            switch (rhs.m_Type) {
                case E_Integer:
                    return    !this->equal(rhs.m_Value.integer, m_Value.real)
                              && !this->less(rhs.m_Value.integer, m_Value.real);
                case E_PositiveInteger:
                    return    !this->equal(rhs.m_Value.positiveInteger, m_Value.real)
                              && !this->less(rhs.m_Value.positiveInteger, m_Value.real);
                case E_Real:
                    return m_Value.real < rhs.m_Value.real;
                case E_Nan:
                    break;
            }
            break;
        case E_Nan:
            break;
    }
    return false;
}

bool COrdinal::isNan(void) const {
    return m_Type == E_Nan;
}

double COrdinal::asDouble(void) const {
    switch (m_Type) {
        case E_Integer:
            return static_cast<double>(m_Value.integer);
        case E_PositiveInteger:
            return static_cast<double>(m_Value.positiveInteger);
        case E_Real:
            return m_Value.real;
        case E_Nan:
            break;
    }
    return std::numeric_limits<double>::quiet_NaN();
}

uint64_t COrdinal::hash(void) {
    return m_Value.positiveInteger;
}

bool COrdinal::equal(int64_t lhs, uint64_t rhs) const {
    return lhs < 0 ? false : static_cast<uint64_t>(lhs) == rhs;
}

bool COrdinal::equal(int64_t lhs, double rhs) const {
    if (   rhs < static_cast<double>(boost::numeric::bounds<int64_t>::lowest())
            || rhs > static_cast<double>(boost::numeric::bounds<int64_t>::highest())) {
        return false;
    }
    double integerPart;
    double remainder = ::modf(rhs, &integerPart);
    return remainder > 0.0 ? false : lhs == static_cast<int64_t>(integerPart);
}

bool COrdinal::equal(uint64_t lhs, double rhs) const {
    if (   rhs < 0.0
            || rhs > static_cast<double>(boost::numeric::bounds<uint64_t>::highest())) {
        return false;
    }
    double integerPart;
    double remainder = ::modf(rhs, &integerPart);
    return remainder > 0.0 ? false : lhs == static_cast<uint64_t>(integerPart);
}

bool COrdinal::less(int64_t lhs, uint64_t rhs) const {
    return lhs < 0 ? true : static_cast<uint64_t>(lhs) < rhs;
}

bool COrdinal::less(int64_t lhs, double rhs) const {
    if (rhs < static_cast<double>(boost::numeric::bounds<int64_t>::lowest())) {
        return false;
    }
    if (rhs > static_cast<double>(boost::numeric::bounds<int64_t>::highest())) {
        return true;
    }
    double integerPart;
    double remainder = ::modf(rhs, &integerPart);
    return    lhs <  static_cast<int64_t>(integerPart)
              || (lhs == static_cast<int64_t>(integerPart) && remainder > 0.0);
}

bool COrdinal::less(uint64_t lhs, double rhs) const {
    if (rhs < 0.0) {
        return false;
    }
    if (rhs > static_cast<double>(boost::numeric::bounds<uint64_t>::highest())) {
        return true;
    }
    double integerPart;
    double remainder = ::modf(rhs, &integerPart);
    return    lhs <  static_cast<uint64_t>(integerPart)
              || (lhs == static_cast<uint64_t>(integerPart) && remainder > 0.0);
}

std::ostream &operator<<(std::ostream &o, COrdinal ord) {
    switch (ord.m_Type) {
        case COrdinal::E_Integer:
            o << ord.m_Value.integer;
            break;
        case COrdinal::E_PositiveInteger:
            o << ord.m_Value.positiveInteger;
            break;
        case COrdinal::E_Real:
            o << ord.m_Value.real;
            break;
        case COrdinal::E_Nan:
            o << "nan";
            break;
    }
    return o;
}

}
}
