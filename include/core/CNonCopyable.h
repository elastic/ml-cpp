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
#ifndef INCLUDED_ml_core_CNonCopyable_h
#define INCLUDED_ml_core_CNonCopyable_h

#include <core/ImportExport.h>


namespace ml {
namespace core {


//! \brief
//! Equivalent to boost::noncopyable.
//!
//! DESCRIPTION:\n
//! This class is designed to be used wherever boost::noncopyable would
//! be used.  It has the advantage over boost::noncopyable that it does
//! not trigger Visual C++ warning C4275.
//!
//! Classes for which copying is not allowed should inherit privately
//! from this class.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The class is exported from the DLL, which means that non-inlined
//! versions of its methods will be generated even though they're all
//! inlined.  This is the difference compared to boost::noncopyable,
//! and what prevents Visual C++ warning C4275.
//!
class CORE_EXPORT CNonCopyable {
    protected:
        //! Inlined in the hope that the compiler will optimise it away
        CNonCopyable(void) {}

        //! Inlined in the hope that the compiler will optimise it away
        ~CNonCopyable(void) {}

    private:
        //! Prevent copying
        CNonCopyable(const CNonCopyable &);
        CNonCopyable &operator=(const CNonCopyable &);
};


}
}

#endif // INCLUDED_ml_core_CNonCopyable_h

