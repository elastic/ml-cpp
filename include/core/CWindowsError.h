/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CWindowsError_h
#define INCLUDED_ml_core_CWindowsError_h

#include <core/ImportExport.h>

#include <iosfwd>
#include <string>

#include <stdint.h>


namespace ml
{
namespace core
{

//! \brief
//! Encapsulate interpretation of Windows function errors.
//!
//! DESCRIPTION:\n
//! Encapsulate interpretation of Windows function errors.
//! On Windows most API functions set an error number that
//! is retrieved using the GetLastError() function.  The
//! code required to translate this to a human readable
//! string is quite verbose, so this class encapsulates it.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The default constructor initialises an object using the
//! last error that occurred.  However, it's also possible
//! to explicitly initialise an object with a specific error
//! code.
//!
class CORE_EXPORT CWindowsError
{
    public:
        //! Initialise using the last error to occur.  This
        //! is obtained GetLastError() on Windows.
        CWindowsError(void);

        //! Initialise using a specific error number
        CWindowsError(uint32_t errorCode);

        //! Access the raw error code number
        uint32_t errorCode(void) const;

        //! Textual representation of the error
        std::string errorString(void) const;

    private:
        //! The error code
        uint32_t m_ErrorCode;

    friend CORE_EXPORT std::ostream &operator<<(std::ostream &,
                                                const CWindowsError &);
};

//! Stream output prints textual representation of the error
CORE_EXPORT std::ostream &operator<<(std::ostream &os,
                                     const CWindowsError &windowsError);


}
}

#endif // INCLUDED_ml_core_CWindowsError_h

