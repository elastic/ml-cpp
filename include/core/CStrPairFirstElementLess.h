/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CStrPairFirstElementLess_h
#define INCLUDED_ml_core_CStrPairFirstElementLess_h

#include <core/ImportExport.h>

#include <string>
#include <utility>


namespace ml
{
namespace core
{


//! \brief
//! Specialised comparator for string pairs
//!
//! DESCRIPTION:\n
//! Less than comparator for std::pair objects where the first element is
//! convertible to std::string that only considers the first element of the
//! pair.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The two pairs don't have to have exactly the same type, although the first
//! element of both pairs must be either a std::string, or convertible to a
//! const std::string & (e.g. a boost::reference_wrapper<const std::string &>).
//!
class CORE_EXPORT CStrPairFirstElementLess
{
    public:
        template <typename PAIR1, typename PAIR2>
        bool operator()(const PAIR1 &pr1, const PAIR2 &pr2)
        {
            const std::string &pr1first = pr1.first;
            const std::string &pr2first = pr2.first;

            return pr1first < pr2first;
        }
};


}
}

#endif // INCLUDED_ml_core_CStrPairFirstElementLess_h

