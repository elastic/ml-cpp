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
#ifndef INCLUDED_ml_maths_common_ImportExport_h
#define INCLUDED_ml_maths_common_ImportExport_h

//! On Windows, it's necessary to explicitly export functions from
//! DLLs.  The simplest way to do this for C++ classes is to add
//! __declspec(dllexport) between the class keyword and the name
//! of the class when declaring it.  However, when using the class
//! from a different DLL or a program, it's necessary that the
//! __declspec(dllimport) modifier.  Since we only want one header
//! file declaring each class, this means that these modifiers
//! have to be held in a macro that expands to __declspec(dllexport)
//! when the DLL containing the class is being built, but to
//! __declspec(dllimport) when other code that uses the class is
//! being built.  On Unix the macro must evaluate to an empty string.

#ifdef Windows

#ifdef BUILDING_libMlMathsCommon
// We're building the maths common library
#define MATHS_COMMON_EXPORT __declspec(dllexport)
#else
// We're building some code other than the maths common library
#define MATHS_COMMON_EXPORT __declspec(dllimport)
#endif

#else

// Empty string on Unix
#define MATHS_COMMON_EXPORT

#endif

#endif // INCLUDED_ml_maths_common_ImportExport_h
