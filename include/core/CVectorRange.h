/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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

#ifndef INCLUDED_ml_core_CVectorRange_h
#define INCLUDED_ml_core_CVectorRange_h

#include <algorithm>
#include <exception>
#include <sstream>

namespace ml {
namespace core {
template<typename VECTOR> class CVectorRange;

namespace vector_range_detail {
//! \brief Gets the reference type.
template<typename VECTOR>
struct SReferenceType {
    using type = typename VECTOR::reference;
};
template<typename VECTOR>
struct SReferenceType<const VECTOR> {
    using type = typename VECTOR::const_reference;
};

//! \brief Gets the iterator type.
template<typename VECTOR>
struct SIteratorType {
    using type = typename VECTOR::iterator;
};
template<typename VECTOR>
struct SIteratorType<const VECTOR> {
    using type = typename VECTOR::const_iterator;
};

//! \brief Implements assignment.
template<typename VECTOR>
struct SDoAssign {
    static const CVectorRange<VECTOR> &dispatch(CVectorRange<VECTOR> &lhs,
                                                const CVectorRange<VECTOR> &rhs) {
        if (rhs.base() != lhs.base()) {
            lhs.assign(rhs.begin(), rhs.end());
        } else   {
            VECTOR tmp{rhs.begin(), rhs.end()};
            lhs.assign(tmp.begin(), tmp.end());
        }
        return lhs;
    }
};
template<typename VECTOR>
struct SDoAssign<const VECTOR> {
    static const CVectorRange<const VECTOR> &dispatch(CVectorRange<const VECTOR> &lhs,
                                                      const CVectorRange<const VECTOR> &rhs) {
        CVectorRange<const VECTOR> tmp(*rhs.base(), rhs.a(), rhs.b());
        lhs.swap(tmp);
        return lhs;
    }
};
}

//! \name A vector subrange backed by a specified vector.
//!
//! DESCRIPTION:\n
//! A lightweight mostly c++11 compliant vector interface to a contiguous
//! sub-range of a specified vector type.
template<typename VECTOR>
class CVectorRange {
    public:
        using allocator_type = typename VECTOR::allocator_type;
        using size_type = typename VECTOR::size_type;
        using reference = typename vector_range_detail::SReferenceType<VECTOR>::type;
        using const_reference = typename VECTOR::const_reference;
        using iterator = typename vector_range_detail::SIteratorType<VECTOR>::type;
        using const_iterator = typename VECTOR::const_iterator;

    public:
        CVectorRange(VECTOR &vector, size_type a, size_type b) :
            m_Vector(&vector), m_A(a), m_B(b)
        {}

        //! Copy assignment.
        const CVectorRange &operator=(const CVectorRange &other) {
            return vector_range_detail::SDoAssign<VECTOR>::dispatch(*this, other);
        }

        //! Assign from value.
        template<typename T>
        void assign(size_type n, const T &value) {
            std::fill_n(this->begin(), std::min(this->size(), n), value);
            if (n > this->size()) {
                m_Vector->insert(this->end(), n - this->size(), value);
            } else if (n < this->size())   {
                m_Vector->erase(this->begin() + n, this->end());
            }
            m_B = m_A + n;
        }
        //! Assign from range.
        template<typename ITR>
        void assign(ITR begin, ITR end) {
            size_type size = std::distance(begin, end);
            std::copy(begin, begin + std::min(this->size(), size), this->begin());
            if (size > this->size()) {
                m_Vector->insert(this->end(), begin + this->size(), end);
            } else if (size < this->size())   {
                m_Vector->erase(this->begin() + size, this->end());
            }
            m_B = m_A + size;
        }

        //! Get the underlying vector allocator.
        allocator_type get_allocator(void) const {
            return m_Vector->get_allocator;
        }

        //! Get writable element at \p pos.
        reference at(size_type pos) {
            this->range_check(pos);
            return (*m_Vector)[m_A + pos];
        }
        //! Get read-only element at \p pos.
        const_reference at(size_type pos) const {
            this->range_check(pos);
            return (*m_Vector)[m_A + pos];
        }

        //! Get writable element at \p pos.
        reference operator[](size_type pos) {
            return (*m_Vector)[m_A + pos];
        }
        //! Get read-only element at \p pos.
        const_reference operator[](size_type pos) const {
            return (*m_Vector)[m_A + pos];
        }

        //! Get writable first element.
        reference front(void) { return this->operator[](0); }
        //! Get read-only first element.
        const_reference front(void) const { return this->operator[](0); }

        //! Get writable last element.
        reference back(void) { return this->operator[](m_B - m_A - 1); }
        //! Get read-only last element.
        const_reference back(void) const { return this->operator[](m_B - m_A - 1); }

        //! Input iterator to start of range.
        iterator begin(void) { return m_Vector->begin() + m_A; }
        //! Output iterator to start of range.
        const_iterator begin(void) const { return m_Vector->begin() + m_A; }
        //! Output iterator to start of range.
        const_iterator cbegin(void) const { return m_Vector->begin() + m_A; }

        //! Input iterator to end of range.
        iterator end(void) { return m_Vector->begin() + m_B; }
        //! Output iterator to end of range.
        const_iterator end(void) const { return m_Vector->begin() + m_B; }
        //! Output iterator to end of range.
        const_iterator cend(void) const { return m_Vector->begin() + m_B; }

        //! Check if the range is empty.
        bool empty(void) const { return m_B == m_A; }
        //! Size of range.
        size_type size(void) const { return m_B - m_A; }
        //! Get the maximum permitted size.
        size_type max_size(void) const { return m_Vector->max_size(); }
        //! Reserve space for \p size elements.
        void reserve(size_type size) {
            m_Vector->reserve((size + m_Vector->size()) - this->size());
        }
        //! Get the number of elements which can be held in the currently
        //! allocated storage.
        size_type capacity(void) const {
            return (m_Vector->capacity() - m_Vector->size()) + this->size();
        }

        //! Clear the contents.
        void clear(void) {
            this->erase(this->begin(), this->end());
            m_B = m_A;
        }
        //! Remove the element at \p pos.
        iterator erase(const_iterator pos) {
            --m_B;
            return m_Vector->erase(pos);
        }
        //! Remove elements in the range [begin, end).
        iterator erase(const_iterator begin, const_iterator end) {
            m_B -= std::distance(begin, end);
            return m_Vector->erase(begin, end);
        }
        //! Insert a value at \p pos.
        template<typename T>
        iterator insert(const_iterator pos, const T &value) {
            ++m_B;
            return m_Vector->insert(pos, value);
        }
        //! Insert \p n copies of \p value at \p pos.
        template<typename T>
        iterator insert(const_iterator pos, size_type n, const T &value) {
            m_B += n;
            return m_Vector->insert(pos, n, value);
        }
        //! Insert the value [\p begin, \p end) at \p pos.
        template<typename ITR>
        iterator insert(const_iterator pos, ITR begin, ITR end) {
            m_B += std::distance(begin, end);
            return m_Vector->insert(pos, begin, end);
        }
        //! Add an element at the end of the range.
        //!
        //! \warning This is not O(1).
        template<typename T>
        void push_back(const T &value) {
            this->insert(this->end(), value);
        }
        //! Remove an element from the end of the range.
        //!
        //! \warning This is not O(1).
        void pop_back(void) {
            this->erase(this->end() - 1);
        }
        //! Resize adding default constructed values if \p n is greater
        //! than the current size.
        void resize(size_type n) {
            this->resize(n, typename VECTOR::value_type());
        }
        //! Resize adding default constructed values if \p n is greater
        //! than the current size.
        template<typename T>
        void resize(size_type n, const T &value) {
            if (n > this->size()) {
                this->insert(this->end(), n - this->size(), value);
            } else if (n < this->size())   {
                this->erase(this->begin() + n, this->end());
            }
        }
        //! Swap two ranges.
        void swap(CVectorRange &other) {
            std::swap(m_Vector, other.m_Vector);
            std::swap(m_A, other.m_A);
            std::swap(m_B, other.m_B);
        }

        //! Get the base vector.
        VECTOR *base(void) const { return m_Vector; }

        //! Get the start of the range.
        size_type a(void) const { return m_A; }

        //! Get the end of the range.
        size_type b(void) const { return m_B; }

    private:
        //! Check if \p pos is in range.
        void range_check(size_type pos) const {
            if (m_A + pos >= m_B) {
                std::ostringstream message;
                message << "out of range: " << pos << " >= " << m_B - m_A;
                throw std::out_of_range(message.str());
            }
        }

    private:
        //! The underlying vector.
        VECTOR *m_Vector;
        //! The range [m_A, m_B).
        size_type m_A, m_B;
};

//! Check if \p lhs and \p rhs are equal.
template<typename VECTOR>
bool operator==(const CVectorRange<VECTOR> &lhs, const CVectorRange<VECTOR> &rhs) {
    return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}
//! Check if \p lhs and \p rhs are not equal.
template<typename VECTOR>
bool operator!=(const CVectorRange<VECTOR> &lhs, const CVectorRange<VECTOR> &rhs) {
    return !(lhs == rhs);
}
//! Check if \p lhs is lexicographically less than \p rhs.
template<typename VECTOR>
bool operator<(const CVectorRange<VECTOR> &lhs, const CVectorRange<VECTOR> &rhs) {
    return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}
//! Check if \p lhs is lexicographically less or equal to \p rhs.
template<typename VECTOR>
bool operator<=(const CVectorRange<VECTOR> &lhs, const CVectorRange<VECTOR> &rhs) {
    return lhs < rhs || lhs == rhs;
}
//! Check if \p lhs is lexicographically greater than \p rhs.
template<typename VECTOR>
bool operator>(const CVectorRange<VECTOR> &lhs, const CVectorRange<VECTOR> &rhs) {
    return rhs < lhs;
}
//! Check if \p lhs is lexicographically less or equal to \p rhs.
template<typename VECTOR>
bool operator>=(const CVectorRange<VECTOR> &lhs, const CVectorRange<VECTOR> &rhs) {
    return rhs <= lhs;
}

//! Free swap function to participate in Koenig lookup.
template<typename VECTOR>
void swap(CVectorRange<VECTOR> &lhs, CVectorRange<VECTOR> &rhs) {
    lhs.swap(rhs);
}

//! Make a vector subrange.
template<typename VECTOR>
CVectorRange<VECTOR> make_range(VECTOR &vector, std::size_t a, std::size_t b) {
    return CVectorRange<VECTOR>(vector, a, b);
}

}
}

#endif // INCLUDED_ml_core_CVectorRange_h
