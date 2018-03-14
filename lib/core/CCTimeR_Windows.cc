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
#include <core/CCTimeR.h>

namespace ml {
namespace core {

char* CCTimeR::cTimeR(const time_t* clock, char* result) {
    // This is effectively bypassing the security feature of the Windows
    // ctime_s() call, but the wrapper function has the arguments of the
    // vulnerable Unix ctime_r() function, so we don't know the real buffer
    // size, and must assume it's big enough
    static const size_t MIN_BUF_SIZE(26);

    ::ctime_s(result, MIN_BUF_SIZE, clock);

    return result;
}
}
}
