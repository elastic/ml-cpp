/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_maths_ImportExport_h
#define INCLUDED_ml_maths_ImportExport_h

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

#ifdef BUILDING_libMlMaths
// We're building the maths library
#define MATHS_EXPORT __declspec(dllexport)
#else
// We're building some code other than the maths library
#define MATHS_EXPORT __declspec(dllimport)
#endif

//! See https://devblogs.microsoft.com/cppblog/optimizing-the-layout-of-empty-base-classes-in-vs2015-update-2-3/
#define EMPTY_BASE_OPT __declspec(empty_bases)

#else

// Empty string on Unix
#define MATHS_EXPORT
#define EMPTY_BASE_OPT

#endif

#endif // INCLUDED_ml_maths_ImportExport_h
