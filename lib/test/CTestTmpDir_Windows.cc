/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <test/CTestTmpDir.h>

#include <core/CLogger.h>

#include <stdlib.h>


namespace ml
{
namespace test
{


std::string CTestTmpDir::tmpDir()
{
    const char *temp(::getenv("TEMP"));
    if (temp == 0)
    {
        LOG_ERROR("%TEMP% environment variable not set");
        return ".";
    }

    return temp;
}


}
}

