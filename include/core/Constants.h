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

#ifndef INCLUDED_ml_core_Constants_h
#define INCLUDED_ml_core_Constants_h

#include <core/CoreTypes.h>

#include <cmath>
#include <limits>

namespace ml {
namespace core {
namespace constants {

//! A minute in seconds.
constexpr core_t::TTime MINUTE{60};

//! An hour in seconds.
constexpr core_t::TTime HOUR{3600};

//! A day in seconds.
constexpr core_t::TTime DAY{86400};

//! Two days in seconds.
constexpr core_t::TTime WEEKEND{172800};

//! Five days in seconds.
constexpr core_t::TTime WEEKDAYS{432000};

//! A week in seconds.
constexpr core_t::TTime WEEK{604800};

//! 365 days in seconds.
//!
//! Note that this doesn't use the more standard 365.2425 days average length
//! of a Gregorian year because things work out slightly better if it is a
//! multiple time series bucket length.
constexpr core_t::TTime YEAR{31536000};

//! The number of bytes in a kilobyte
constexpr std::size_t BYTES_IN_KILOBYTES{1024ULL};

//! The number of bytes in a megabyte
constexpr std::size_t BYTES_IN_MEGABYTES{1024ULL * 1024};

//! The number of bytes in a gigabyte
constexpr std::size_t BYTES_IN_GIGABYTES{1024ULL * 1024 * 1024};

//! The number of bytes in a terabyte
constexpr std::size_t BYTES_IN_TERABYTES{1024ULL * 1024 * 1024 * 1024};

//! The number of bytes in a gigabyte
constexpr std::size_t BYTES_IN_PETABYTES{1024ULL * 1024 * 1024 * 1024 * 1024};

//! Log of min double.
const double LOG_MIN_DOUBLE{std::log(std::numeric_limits<double>::min())};

//! Log of max double.
const double LOG_MAX_DOUBLE{std::log(std::numeric_limits<double>::max())};

//! Log of double epsilon.
const double LOG_DOUBLE_EPSILON{std::log(std::numeric_limits<double>::epsilon())};

//! Log of two.
constexpr double LOG_TWO{0.693147180559945};

//! Log of two pi.
constexpr double LOG_TWO_PI{1.83787706640935};

#ifdef Windows
constexpr char PATH_SEPARATOR{'\\'};
#else
constexpr char PATH_SEPARATOR{'/'};
#endif
}
}
}

#endif // INCLUDED_ml_core_Constants_h
