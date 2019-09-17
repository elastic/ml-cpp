/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_WindowsSafe_h
#define INCLUDED_ml_core_WindowsSafe_h

//! \brief
//! Wrapper around Windows.h that includes the underlying file and
//! then undefines macros that mess up other code

#ifdef Windows

#include <Windows.h>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#ifdef TEXT
#undef TEXT
#endif
#ifdef ERROR
#undef ERROR
#endif

#ifndef STDCALL
#define STDCALL __stdcall
#endif

#else

// Define STDCALL to nothing on Unix
#ifndef STDCALL
#define STDCALL
#endif

#endif

#endif // INCLUDED_ml_core_WindowsSafe_h
