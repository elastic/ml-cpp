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
#include "CIncrementer.h"


namespace ml {
namespace vfprog {


CIncrementer::~CIncrementer(void) {}

size_t CIncrementer::nonVirtualIncrement(size_t val) {
    return val + 1;
}

size_t CIncrementer::virtualIncrement(size_t val) {
    return val + 1;
}


}
}

