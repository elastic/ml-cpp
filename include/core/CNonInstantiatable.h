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
#ifndef INCLUDED_ml_core_CNonInstantiatable_h
#define INCLUDED_ml_core_CNonInstantiatable_h

#include <core/ImportExport.h>


namespace ml
{
namespace core
{


//! \brief
//! Similar idea to boost::noncopyable, but for instantiation.
//!
//! DESCRIPTION:\n
//! Classes for which instantiation is not allowed, i.e. those that
//! have only static methods, should inherit privately from this
//! class.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The class is (seemingly pointlessly) exported from the DLL to
//! prevent Visual C++ warning C4275.
//!
class CORE_EXPORT CNonInstantiatable
{
    private:
        //! Prevent instantiation
        CNonInstantiatable();
        CNonInstantiatable(const CNonInstantiatable &);
};


}
}

#endif // INCLUDED_ml_core_CNonInstantiatable_h

