/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CFileDeleter_h
#define INCLUDED_ml_core_CFileDeleter_h

#include <core/CNonCopyable.h>
#include <core/ImportExport.h>

#include <string>


namespace ml
{
namespace core
{

//! \brief
//! RAII file deleter
//!
//! DESCRIPTION:\n
//! On destruction an object of this class will attempt
//! to delete the file whose name was specified to its
//! constructor.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Only a "best effort" is made to remove the specified
//! file.  In particular, on Windows it will not be removed
//! if it is still open.
//!
//! This class does NOT currently attempt to close the file
//! before removing it.  In future it could be extended
//! to take a file descriptor/FILE pointer/stream reference
//! and close this before attempting to delete the file.
//!
class CORE_EXPORT CFileDeleter : private CNonCopyable
{
    public:
        //! Record the name of the file to delete
        CFileDeleter(const std::string &fileName);

        //! Attempt to remove the specified file
        ~CFileDeleter();

    private:
        std::string m_FileName;
};


}
}

#endif // INCLUDED_ml_core_CFileDeleter_h

