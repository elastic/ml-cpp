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

#ifndef INCLUDED_ml_core_CBoostJsonConstants_h
#define INCLUDED_ml_core_CBoostJsonConstants_h

namespace ml {
namespace core {
namespace boost_json_constants {

// Constants that set upper limits for Boost.JSON SAX style parsing

// The maximum number of elements allowed in an object
constexpr std::size_t MAX_OBJECT_SIZE = 1'000'000;

// The maximum number of elements allowed in an array
constexpr std::size_t MAX_ARRAY_SIZE = 1'000'000;

// The maximum number of characters allowed in a key
constexpr std::size_t MAX_KEY_SIZE = 1 << 10;

// The maximum number of characters allowed in a string
constexpr std::size_t MAX_STRING_SIZE = 1 << 30;
}
}
}

#endif // INCLUDED_ml_core_CBoostJsonConstants_h
