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
#include "CCmdLineParser.h"

#include <core/CLogger.h>
#include <core/CStringUtils.h>
#include <core/CTimeUtils.h>
#include <core/CoreTypes.h>

#include <iostream>

#include <stdlib.h>

using namespace ml;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Utility to convert a Unix time to a string" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <unixtime>" << std::endl;
        return EXIT_FAILURE;
    }

    core_t::TTime t(0);
    if (core::CStringUtils::stringToType(argv[1], t) == false) {
        LOG_FATAL(<< "Unable to convert " << argv[1] << " to integer");
        return EXIT_FAILURE;
    }

    std::cout << core::CTimeUtils::toIso8601(t) << std::endl;

    return EXIT_SUCCESS;
}
