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
#ifndef INCLUDED_ml_core_CStrCaseCmp_h
#define INCLUDED_ml_core_CStrCaseCmp_h

#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>

#include <stddef.h>


namespace ml {
namespace core {


//! \brief
//! Portable wrapper for the strcasecmp() function.
//!
//! DESCRIPTION:\n
//! Portable wrapper for the strcasecmp() function and the closely
//! related strncasecmp() function.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This has been broken into a class of its own because Windows has a
//! _stricmp() function whilst Unix has strcasecmp().
//!
class CORE_EXPORT CStrCaseCmp : private CNonInstantiatable {
    public:
        static int strCaseCmp(const char *s1, const char *s2);
        static int strNCaseCmp(const char *s1, const char *s2, size_t n);
};


}
}

#endif // INCLUDED_ml_core_CStrCaseCmp_h

