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

#ifndef INCLUDED_ml_core_Constants_h
#define INCLUDED_ml_core_Constants_h

#include <core/CoreTypes.h>

#include <cmath>
#include <limits>

namespace ml {
namespace core {
namespace constants {

//! An hour in seconds.
const core_t::TTime HOUR = 3600;

//! A day in seconds.
const core_t::TTime DAY = 86400;

//! A (two day) weekend in seconds.
const core_t::TTime WEEKEND = 172800;

//! Five weekdays in seconds.
const core_t::TTime WEEKDAYS = 432000;

//! A week in seconds.
const core_t::TTime WEEK = 604800;

//! A (364 day) year in seconds.
const core_t::TTime YEAR = 31449600;

//! Log of min double.
const double LOG_MIN_DOUBLE = std::log(std::numeric_limits<double>::min());

//! Log of max double.
const double LOG_MAX_DOUBLE = std::log(std::numeric_limits<double>::max());

//! Log of double epsilon.
const double LOG_DOUBLE_EPSILON = std::log(std::numeric_limits<double>::epsilon());

//! Log of two.
const double LOG_TWO = 0.693147180559945;

//! Log of two pi.
const double LOG_TWO_PI = 1.83787706640935;

#ifdef Windows
const char PATH_SEPARATOR = '\\';
#else
const char PATH_SEPARATOR = '/';
#endif
}
}
}

#endif // INCLUDED_ml_core_Constants_h
