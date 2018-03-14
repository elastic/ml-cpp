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

#ifndef INCLUDED_ml_core_CPolymorphicStackObjectCPtr_h
#define INCLUDED_ml_core_CPolymorphicStackObjectCPtr_h

#include <core/ImportExport.h>

#include <boost/type_traits/remove_const.hpp>
#include <boost/variant.hpp>

namespace ml {
namespace core {
class CORE_EXPORT CNullPolymorphicStackObjectCPtr {
};

//! \brief A stack based pointer to polymorphic object emulator.
//!
//! IMPLEMENTATION:\n
//! For small class hierarchies this allows one to emulate a pointer
//! to a polymorphic base class using stack based storage. In particular,
//! for up to four derived classes of an interface one can store the
//! object by value, but access its members through the interface, i.e.
//! it behaves exactly like a pointer to the base class in terms of usage.
//! This is to support runtime polymorphism without using the heap.
template<typename BASE, typename D1, typename D2, typename D3 = D2, typename D4 = D2>
class CPolymorphicStackObjectCPtr {
    private:
        typedef const typename boost::remove_const<BASE>::type TConstBase;
        typedef const typename boost::remove_const<D1>::type TConstD1;
        typedef const typename boost::remove_const<D2>::type TConstD2;
        typedef const typename boost::remove_const<D3>::type TConstD3;
        typedef const typename boost::remove_const<D4>::type TConstD4;

    public:
        CPolymorphicStackObjectCPtr(void) : m_Storage(CNullPolymorphicStackObjectCPtr()) {
        }

        template<typename T>
        explicit CPolymorphicStackObjectCPtr(const T &d) : m_Storage(d) {
        }

        template<typename O1, typename O2, typename O3, typename O4>
        CPolymorphicStackObjectCPtr(const CPolymorphicStackObjectCPtr<BASE, O1, O2, O3, O4> &other) {
#define MAYBE_SET(TYPE) {                                         \
        TYPE *d = other.template get<TYPE>(); \
        if (d)                                \
        {                                     \
            m_Storage = *d;                   \
            return;                           \
        }                                     \
}
            MAYBE_SET(TConstD1);
            MAYBE_SET(TConstD2);
            MAYBE_SET(TConstD3);
            MAYBE_SET(TConstD4);
#undef MAYBE_SET
            m_Storage = CNullPolymorphicStackObjectCPtr();
        }

        template<typename O1, typename O2, typename O3, typename O4>
        const CPolymorphicStackObjectCPtr &operator=(const CPolymorphicStackObjectCPtr<BASE, O1, O2, O3, O4> &other) {
            CPolymorphicStackObjectCPtr tmp(other);
            this->swap(tmp);
            return *this;
        }

        operator bool () const
        {
            return boost::relaxed_get<CNullPolymorphicStackObjectCPtr>(&m_Storage) == 0;
        }

        TConstBase *operator->(void) const {
#define MAYBE_RETURN(TYPE) {                                                        \
        TYPE *result = boost::relaxed_get<TYPE>(&m_Storage); \
        if (result)                                          \
        {                                                    \
            return static_cast<TConstBase*>(result);         \
        }                                                    \
}
            MAYBE_RETURN(TConstD1);
            MAYBE_RETURN(TConstD2);
            MAYBE_RETURN(TConstD3);
            MAYBE_RETURN(TConstD4);
#undef MAYBE_RETURN
            return 0;
        }

        TConstBase &operator*(void) const {
            return *(this->operator->());
        }

        template<typename T> const T *get(void) const {
            return boost::relaxed_get<T>(&m_Storage);
        }

    private:
        void swap(CPolymorphicStackObjectCPtr &other) {
            m_Storage.swap(other.m_Storage);
        }

    private:
        typedef boost::variant<D1, D2, D3, D4, CNullPolymorphicStackObjectCPtr> TStorage;

    private:
        //! The static storage of the actual type.
        TStorage m_Storage;
};

}
}

#endif // INCLUDED_ml_core_CPolymorphicStackObjectCPtr_h
