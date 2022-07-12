/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#ifndef INCLUDED_ml_core_CFunctional_h
#define INCLUDED_ml_core_CFunctional_h

#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>

#include <memory>

namespace ml {
namespace core {

//! \brief Useful extensions to the std:: functional collection of types.
class CORE_EXPORT CFunctional : CNonInstantiatable {
public:
    //! \brief Dereferences objects which support a unary operator *
    //! and calls the predicate \p PRED on them.
    template<typename PRED>
    struct SDereference {
        explicit SDereference(const PRED& pred = PRED()) : s_Pred(pred) {}

        //! Version for unary predicates.
        //!
        //! \note SFINAE means this won't be a problem even if PRED
        //! is a unary predicate.
        template<typename T>
        inline bool operator()(const T& ptr) const {
            return s_Pred(*ptr);
        }

        //! Version for binary predicates.
        //!
        //! \note SFINAE means this won't be a problem even if PRED
        //! is a unary predicate.
        template<typename U, typename V>
        inline bool operator()(const U& lhs, const V& rhs) const {
            return s_Pred(*lhs, *rhs);
        }

        PRED s_Pred;
    };
};
}
}

#endif // INCLUDED_ml_core_CFunctional_h
