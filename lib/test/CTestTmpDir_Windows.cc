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
#include <test/CTestTmpDir.h>

#include <core/CLogger.h>

#include <stdlib.h>

namespace ml {
namespace test {

std::string CTestTmpDir::tmpDir(void) {
    const char *temp(::getenv("TEMP"));
    if (temp == 0) {
        LOG_ERROR("%TEMP% environment variable not set");
        return ".";
    }

    return temp;
}
}
}
