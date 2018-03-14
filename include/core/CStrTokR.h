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
#ifndef INCLUDED_ml_core_CStrTokR_h
#define INCLUDED_ml_core_CStrTokR_h

#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>


namespace ml {
namespace core {


//! \brief
//! Portable wrapper for the strtok_r() function.
//!
//! DESCRIPTION:\n
//! Portable wrapper for the strtok_r() function.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This has been broken into a class of its own because Windows has a
//! strtok_s() function rather than Unix's strtok_r().
//!
class CORE_EXPORT CStrTokR : private CNonInstantiatable {
    public:
        static char *strTokR(char *str, const char *sep, char **lasts);
};


}
}

#endif // INCLUDED_ml_core_CStrTokR_h

