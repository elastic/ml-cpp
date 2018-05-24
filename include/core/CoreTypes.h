/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_t_CoreTypes_h
#define INCLUDED_ml_core_t_CoreTypes_h

#include <time.h>

namespace ml {
namespace core_t {

//! For now just use seconds as the ml time granularity
//! This is a UTC value
using TTime = time_t;

//! The standard line ending for the platform - DON'T make this std::string as
//! that would cause many strings to be constructed (since the variable is
//! const at the namespace level, so is internal to each file this header is
//! included in)
#ifdef Windows
const char* const LINE_ENDING = "\r\n";
#else
const char* const LINE_ENDING = "\n";
#endif
}
}

#endif // INCLUDED_ml_core_t_CoreTypes_h
