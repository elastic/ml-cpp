/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "seccomp/CSystemCallFilter.h"

#include <core/CLogger.h>

#include <fstream>
#include <iostream>

#include <sandbox.h>

namespace ml {
namespace seccomp {

static const std::string SANDBOX_RULES("(version 1) (deny default)");
static const char* PROFILE_NAME = "/tmp/sandbox_rules.sb";

CSystemCallFilter::CSystemCallFilter() {

    std::ofstream tempRulesFile;
    tempRulesFile.open(PROFILE_NAME);
    tempRulesFile << SANDBOX_RULES << std::endl;
    tempRulesFile.close();

    char* errorbuf = nullptr;
    if (sandbox_init(PROFILE_NAME, SANDBOX_NAMED, &errorbuf) != 0) {
        LOG_INFO("Error initialising sandbox");
        if (errorbuf != nullptr) {
            std::cout << errorbuf << std::endl;
            sandbox_free_error(errorbuf);
        }

    } else {
        LOG_INFO("sandbox good");
    }
}
}
}
