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
