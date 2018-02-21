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
#ifndef INCLUDED_ml_core_t_CoreTypes_h
#define INCLUDED_ml_core_t_CoreTypes_h

#include <time.h>


namespace ml
{
namespace core_t
{


//! For now just use seconds as the ml time granularity
//! This is a UTC value
typedef time_t TTime;


//! The standard line ending for the platform - DON'T make this std::string as
//! that would cause many strings to be constructed (since the variable is
//! static const at the namespace level, so is internal to each file this
//! header is included in)
#ifdef Windows
static const char *LINE_ENDING = "\r\n";
#else
#ifdef __GNUC__
// Tell g++ that it's reasonable that this variable isn't used
__attribute__ ((unused)) static const char *LINE_ENDING = "\n";
#else
static const char *LINE_ENDING = "\n";
#endif
#endif


}
}

#endif // INCLUDED_ml_core_t_CoreTypes_h

