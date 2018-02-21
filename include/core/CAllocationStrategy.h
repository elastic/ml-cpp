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
#ifndef INCLUDED_ml_core_CAllocationStrategy_h
#define INCLUDED_ml_core_CAllocationStrategy_h

#include <core/ImportExport.h>

#include <algorithm>
#include <vector>

namespace ml
{
namespace core
{

//! \brief
//! Container allocation strategy
//!
//! DESCRIPTION:\n
//! Container allocation strategy. The STL default allocation
//! strategy is to double the memory for containers when they
//! grow beyond their size.
//!
//! We are memory-conscious, and for some situations this
//! default strategy wastes precious space, so we introduce
//! our own strategy, which increases the space taken by 10%
//!
//! IMPLEMENTATION DECISIONS:\n
//! Template class to work with different containers
//!
class CORE_EXPORT CAllocationStrategy
{
    public:
        //! Reserve a container working around implementation-specific silliness
        template<typename T>
        static void reserve(T &t, std::size_t n)
        {
            t.reserve(n);
        }

        //! Resize a container using a 10% capacity increase
        template<typename T>
        static void resize(T &t, std::size_t n)
        {
            if (n > t.capacity())
            {
                CAllocationStrategy::reserve(t, n * 11 / 10);
            }
            t.resize(n);
        }

        //! Resize a container using a 10% capacity increase, with default value type
        template<typename T>
        static void resize(T &t, std::size_t n, const typename T::value_type &v)
        {
            if (n > t.capacity())
            {
                CAllocationStrategy::reserve(t, n * 11 / 10);
            }
            t.resize(n, v);
        }

        //! push_back an item to a container using a 10% capacity
        //! increase
        template<typename T>
        static void push_back(std::vector<T> &v, const T &t)
        {
            std::size_t capacity = v.capacity();
            if (v.size() == capacity)
            {
                CAllocationStrategy::reserve(v, (capacity * 11 / 10) + 1);
            }
            v.push_back(t);
        }
};

} // core

} // ml

#endif // INCLUDED_ml_core_CAllocationStrategy_h
