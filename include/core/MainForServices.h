/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_MainForServices_h
#define INCLUDED_ml_core_MainForServices_h

#include <core/CProcess.h>

#include <stdlib.h>

// Programs that can run as Windows services must have a function called
// mlMain() - this is the forward declaration of it.
extern int mlMain(int argc, char *argv[]);


//! \brief
//! Boilerplate implentation of the main() function for applications
//! that mayneed to be run as a Windows service.
//!
//! DESCRIPTION:\n
//! Boilerplate implentation of the main() function for applications
//! that may need to be run as a Windows service.
//!
//! Such programs MUST implement a mlMain() function instead of
//! a main() function in there Main.cc file, and include this header
//! in Main.cc only.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Although slightly unorthodox, factoring out the main() function
//! for Windows services into a single header avoids duplication into
//! every single application that may need to be run as a Windows
//! service.
//!
//! Despite being in the core library include directory, this file
//! should never be included in a library.
//!
int main(int argc, char *argv[])
{
    ml::core::CProcess &process = ml::core::CProcess::instance();

    // If this process is not running as a Windows service, this call will
    // immediately pass control to the application's own main() replacement.
    // If this process is running as a Windows service, the main thread will
    // become the service dispatcher thread.
    if (process.startDispatcher(&mlMain, argc, argv) == false)
    {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

#else

#error MainForServices.h should only be included once per application that may \
need to be run as a Windows service, in the Main.cc file.  It appears that \
this rule has not been followed.

#endif // INCLUDED_ml_core_MainForServices_h

