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
#ifndef INCLUDED_ml_core_CSleep_h
#define INCLUDED_ml_core_CSleep_h

#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>

#include <stdint.h>


namespace ml
{
namespace core
{


//! \brief
//! Functions related to sleeping
//!
//! DESCRIPTION:\n
//! This class should be used when there is a requirement to make the current
//! thread sleep for a period of time.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This is a static class - it's not possible to construct an instance of it.
//!
//! On Unix, the sleep method is NOT implemented using sleep(), because this
//! uses SIGALRM which can cause problems for other functionality that uses
//! SIGALRM.  Instead, it is implemented using nanosleep(), which does not use
//! SIGALRM.
//! Also, even though nanosleep() can theoretically sleep for a period measured
//! in nanoseconds, the sleep interval provided by this class is in
//! milliseconds.  The reason is that Windows can only sleep for a multiple of
//! milliseconds, and we can't use functionality that's only available on Unix.
//!
class CORE_EXPORT CSleep : private CNonInstantiatable
{
    public:
        //! A processing delay that has been found (by trial and error) to slow
        //! down a thread when required, but without causing unwanted MySQL
        //! disconnections.
        static const uint32_t DEFAULT_PROCESSING_DELAY;

    public:
        //! Sleep for the given period of time.  Be aware that the operating
        //! system may round this up. Windows sleeps are multiples of 1/64 seconds,
        //! i.e. multiples of 15.625 milliseconds.  Basically, don't expect this to
        //! be ultra-accurate.
        static void sleep(uint32_t milliseconds);

        //! Delay processing for a period of time that has been observed to not
        //! cause problems like database disconnections, socket overflows, etc.
        static void delayProcessing();
};


}
}

#endif // INCLUDED_ml_core_CSleep_h

