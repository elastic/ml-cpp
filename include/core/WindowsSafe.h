/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
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
// GetObject in the Windows API is a graphics-related function from gdi32.dll.
// It clashes with GetObject from RapidJSON. Since we're unlikely to be writing
// Windows GUI code in the ML C++ codebase it should be safe to let the
// RapidJSON method take precedence always.
#ifdef GetObject
#undef GetObject
#endif

#ifndef STDCALL
#define STDCALL __stdcall
#endif

#ifndef EMPTY_BASE_OPT
//! See https://devblogs.microsoft.com/cppblog/optimizing-the-layout-of-empty-base-classes-in-vs2015-update-2-3/
#define EMPTY_BASE_OPT __declspec(empty_bases)
#endif

#else

// Define STDCALL to nothing on Unix
#ifndef STDCALL
#define STDCALL
#endif

// Define EMPTY_BASE_OPT to nothing on Unix
#ifndef EMPTY_BASE_OPT
#define EMPTY_BASE_OPT
#endif

#endif

#endif // INCLUDED_ml_core_WindowsSafe_h
