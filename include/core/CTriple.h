/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CTriple_h
#define INCLUDED_ml_core_CTriple_h

#include <core/CMemory.h>
#include <core/CMemoryUsage.h>

#include <boost/functional/hash.hpp>
#include <boost/operators.hpp>
#include <boost/type_traits/is_pod.hpp>

#include <cstddef>
#include <ostream>

#include <string.h>


namespace ml
{
namespace core
{

//! \brief A tuple with three elements.
//!
//! IMPLEMENTATION:\n
//! This implements a lightweight version of boost::tuple with syntax
//! similar to std::pair for the case of three elements.
//!
//! It also implements hash_value which isn't implemented by boost::tuple
//! meaning it can be used as a boost::unordered_map key provided the
//! underlying types can be hashed using a boost::hasher.
template<typename T1, typename T2, typename T3>
class CTriple : private boost::equality_comparable< CTriple<T1, T2, T3>,
                        boost::partially_ordered< CTriple<T1, T2, T3> > >
{
    public:
        //! See CMemory.
        static bool dynamicSizeAlwaysZero()
        {
            return    memory_detail::SDynamicSizeAlwaysZero<T1>::value()
                   && memory_detail::SDynamicSizeAlwaysZero<T2>::value()
                   && memory_detail::SDynamicSizeAlwaysZero<T3>::value();
        }

    public:
        CTriple() : first(), second(), third() {}
        CTriple(const T1 &first_, const T2 &second_, const T3 &third_) :
                first(first_), second(second_), third(third_)
        {
        }

        bool operator==(const CTriple &other) const
        {
            return first == other.first && second == other.second && third == other.third;
        }

        bool operator<(const CTriple &other) const
        {
            if (first == other.first)
            {
                if (second == other.second)
                {
                    return third < other.third;
                }
                return second < other.second;
            }
            return first < other.first;
        }

        std::size_t hash() const
        {
            std::size_t seed = 0;
            boost::hash_combine(seed, first);
            boost::hash_combine(seed, second);
            boost::hash_combine(seed, third);
            return seed;
        }

        void debugMemoryUsage(CMemoryUsage::TMemoryUsagePtr mem) const
        {
            mem->setName("CTriple");
            CMemoryDebug::dynamicSize("first", first, mem);
            CMemoryDebug::dynamicSize("second", second, mem);
            CMemoryDebug::dynamicSize("third", third, mem);
        }

        std::size_t memoryUsage() const
        {
            std::size_t mem = 0;
            mem += CMemory::dynamicSize(first);
            mem += CMemory::dynamicSize(second);
            mem += CMemory::dynamicSize(third);
            return mem;
        }

    public:
        T1 first;
        T2 second;
        T3 third;
};

template<typename T1, typename T2, typename T3>
CTriple<T1, T2, T3> make_triple(const T1 &first, const T2 &second, const T3 &third)
{
    return CTriple<T1, T2, T3>(first, second, third);
}

template<typename T1, typename T2, typename T3>
std::size_t hash_value(const CTriple<T1, T2, T3> &triple)
{
    return triple.hash();
}

template<typename T1, typename T2, typename T3>
std::ostream &operator<<(std::ostream &o, const CTriple<T1, T2, T3> &triple)
{
    return o << '(' << triple.first << ',' << triple.second << ',' << triple.third << ')';
}

}
}

#endif // INCLUDED_ml_core_CTriple_h

