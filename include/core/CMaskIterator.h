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

#ifndef INCLUDED_ml_core_CMaskIterator_h
#define INCLUDED_ml_core_CMaskIterator_h

#include <boost/operators.hpp>

#include <iterator>
#include <vector>

namespace ml
{
namespace core
{

//! \brief A random access iterator over a subset of the elements of
//! a random access container.
//!
//! DESCRIPTION:\n
//! This implements random access iteration over a subset of elements
//! of a container defined by a mask containing their offsets in that
//! container.
//!
//! IMPLEMENTATION:\n
//! A reference to the mask is taken to keep the iterator lightweight,
//! as such it's the responsibility of the caller to ensure it remains
//! valid for the lifetime of the iterator. Two iterators are only
//! comparable if both the underlying container and underlying mask
//! are the same, although the relevant comparison operators work for
//! both const and non-const versions of the underlying iterator.
template<typename ITR>
class CMaskIterator : private boost::incrementable< CMaskIterator<ITR>,
                              boost::decrementable< CMaskIterator<ITR>,
                              boost::addable2< CMaskIterator<ITR>, typename std::iterator_traits<ITR>::difference_type,
                              boost::subtractable2< CMaskIterator<ITR>, typename std::iterator_traits<ITR>::difference_type > > > >
{
    public:
        typedef typename std::iterator_traits<ITR>::difference_type difference_type;
        typedef typename std::iterator_traits<ITR>::value_type value_type;
        typedef typename std::iterator_traits<ITR>::pointer pointer;
        typedef typename std::iterator_traits<ITR>::reference reference;
        typedef typename std::iterator_traits<ITR>::iterator_category iterator_category;
        typedef std::vector<difference_type> TDifferenceVec;

    public:
        CMaskIterator(ITR begin, const TDifferenceVec &mask, difference_type index) :
            m_Begin(begin), m_Mask(&mask), m_Index(index)
        {}

        template<typename OTHER_ITR>
        bool operator==(const CMaskIterator<OTHER_ITR> &rhs) const
        {
            return this->baseEqual(rhs) && m_Index == rhs.m_Index;
        }
        template<typename OTHER_ITR>
        bool operator!=(const CMaskIterator<OTHER_ITR> &rhs) const
        {
            return !(*this == rhs);
        }
        template<typename OTHER_ITR>
        bool operator<(const CMaskIterator<OTHER_ITR> &rhs) const
        {
            return this->baseEqual(rhs) && m_Index < rhs.m_Index;
        }
        template<typename OTHER_ITR>
        bool operator<=(const CMaskIterator<OTHER_ITR> &rhs) const
        {
            return this->baseEqual(rhs) && m_Index <= rhs.m_Index;
        }
        template<typename OTHER_ITR>
        bool operator>(const CMaskIterator<OTHER_ITR> &rhs) const
        {
            return this->baseEqual(rhs) && m_Index > rhs.m_Index;
        }
        template<typename OTHER_ITR>
        bool operator>=(const CMaskIterator<OTHER_ITR> &rhs) const
        {
            return this->baseEqual(rhs) && m_Index <= rhs.m_Index;
        }

        reference operator*(void) const
        {
            return *(m_Begin + (*m_Mask)[m_Index]);
        }
        pointer operator->(void) const
        {
            return &(*(m_Begin + (*m_Mask)[m_Index]));
        }
        reference operator[](difference_type n) const
        {
            return *(m_Begin + (*m_Mask)[m_Index + n]);
        }

        const CMaskIterator &operator++(void) { ++m_Index; return *this; }
        const CMaskIterator &operator--(void) { --m_Index; return *this; }
        template<typename OTHER_ITR>
        difference_type operator-(const CMaskIterator<OTHER_ITR> &rhs) const
        {
            return  static_cast<difference_type>(m_Index)
                  - static_cast<difference_type>(rhs.m_Index);
        }
        const CMaskIterator &operator+=(difference_type n)
        {
            m_Index += n;
            return *this;
        }
        const CMaskIterator &operator-=(difference_type n)
        {
            m_Index -= n;
            return *this;
        }

    private:
        template<typename OTHER_ITR>
        bool baseEqual(const CMaskIterator<OTHER_ITR> &rhs) const
        {
            return m_Begin == rhs.m_Begin && m_Mask == rhs.m_Mask;
        }

    private:
        //! The start of the container.
        ITR m_Begin;
        //! The mask.
        const TDifferenceVec *m_Mask;
        //! The current element (in the mask).
        difference_type m_Index;
};

//! Get a non-constant mask iterator over a subset of the elements of a vector.
template<typename T>
CMaskIterator<typename std::vector<T>::iterator> begin_masked(std::vector<T> &v,
                                                              const std::vector<ptrdiff_t> &mask)
{
    return CMaskIterator<typename std::vector<T>::iterator>(v.begin(), mask, 0);
}
//! Get a non-constant mask iterator at the end of a subset of the elements of a vector.
template<typename T>
CMaskIterator<typename std::vector<T>::iterator> end_masked(std::vector<T> &v,
                                                            const std::vector<ptrdiff_t> &mask)
{
    return CMaskIterator<typename std::vector<T>::iterator>(v.begin(), mask, mask.size());
}

//! Get a constant mask iterator over a subset of the elements of a vector.
template<typename T>
CMaskIterator<typename std::vector<T>::const_iterator> begin_masked(const std::vector<T> &v,
                                                                    const std::vector<ptrdiff_t> &mask)
{
    return CMaskIterator<typename std::vector<T>::const_iterator>(v.begin(), mask, 0);
}
//! Get a constant mask iterator at the end of a subset of the elements of a vector.
template<typename T>
CMaskIterator<typename std::vector<T>::const_iterator> end_masked(const std::vector<T> &v,
                                                                  const std::vector<ptrdiff_t> &mask)
{
    return CMaskIterator<typename std::vector<T>::const_iterator>(v.begin(), mask, mask.size());
}

//! A mask iterator over a subset of an iterated sequence.
template<typename ITR>
CMaskIterator<ITR> begin_masked(ITR i, const std::vector<ptrdiff_t> &mask)
{
    return CMaskIterator<ITR>(i, mask, 0);
}
//! Get a mask iterator at the end of a subset of the elements of an iterated sequence.
template<typename ITR>
CMaskIterator<ITR> end_masked(ITR i, const std::vector<ptrdiff_t> &mask)
{
    return CMaskIterator<ITR>(i, mask, mask.size());
}

}
}

#endif // INCLUDED_ml_core_CMaskIterator_h
