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

#ifndef INCLUDED_ml_config_ImportExport_h
#define INCLUDED_ml_config_ImportExport_h

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

#ifdef BUILDING_libMlConfig
// We're building the config library
#define CONFIG_EXPORT __declspec(dllexport)
#else
// We're building some code other than the config library
#define CONFIG_EXPORT __declspec(dllimport)
#endif

#else

// Empty string on Unix
#define CONFIG_EXPORT

#endif

#endif// INCLUDED_ml_config_ImportExport_h
