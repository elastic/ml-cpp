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

