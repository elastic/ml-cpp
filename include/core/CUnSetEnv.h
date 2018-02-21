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
#ifndef INCLUDED_ml_core_CUnSetEnv_h
#define INCLUDED_ml_core_CUnSetEnv_h

#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>


namespace ml
{
namespace core
{


//! \brief
//! Portable wrapper for the unsetenv() function.
//!
//! DESCRIPTION:\n
//! Portable wrapper for the unsetenv() function.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This has been broken into a class of its own because Unix has an
//! unsetenv() function, whilst on Windows the Microsoft C runtime
//! library treats a request to set a variable to be empty as a
//! request to delete it from the environment.
//!
class CORE_EXPORT CUnSetEnv : private CNonInstantiatable
{
    public:
        static int unSetEnv(const char *name);
};


}
}

#endif // INCLUDED_ml_core_CUnSetEnv_h

