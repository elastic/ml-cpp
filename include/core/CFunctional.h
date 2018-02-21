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

#ifndef INCLUDED_ml_core_CFunctional_h
#define INCLUDED_ml_core_CFunctional_h

#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>

#include <boost/optional.hpp>
#include <boost/shared_ptr.hpp>

namespace ml
{
namespace core
{

//! \brief Useful extensions to the std:: functional collection of types.
class CORE_EXPORT CFunctional : CNonInstantiatable
{
    public:
        //! \brief Checks is a nullable type is null.
        struct CORE_EXPORT SIsNull
        {
            template<typename T>
            bool operator()(const T *ptr) const
            {
                return ptr == 0;
            }

            template<typename T>
            bool operator()(const boost::optional<T> &optional) const
            {
                return !optional;
            }

            template<typename T>
            bool operator()(boost::shared_ptr<T> &ptr) const
            {
                return ptr == 0;
            }
        };

        //! \brief Dereferences objects which support a unary operator *
        //! and calls the predicate \p PRED on them.
        template<typename PRED>
        struct SDereference
        {
            SDereference(const PRED &pred = PRED()) : s_Pred(pred) {}

            //! Version for unary predicates.
            //!
            //! \note SFINAE means this won't be a problem even if PRED
            //! is a unary predicate.
            template<typename T>
            inline bool operator()(const T &ptr) const
            {
                return s_Pred(*ptr);
            }

            //! Version for binary predicates.
            //!
            //! \note SFINAE means this won't be a problem even if PRED
            //! is a unary predicate.
            template<typename U, typename V>
            inline bool operator()(const U &lhs, const V &rhs) const
            {
                return s_Pred(*lhs, *rhs);
            }

            PRED s_Pred;
        };
};

}
}

#endif // INCLUDED_ml_core_CFunctional_h
