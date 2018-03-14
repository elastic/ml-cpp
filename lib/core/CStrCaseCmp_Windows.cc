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
#include <core/CStrCaseCmp.h>

#include <string.h>


namespace ml {
namespace core {


int CStrCaseCmp::strCaseCmp(const char *s1, const char *s2) {
    return ::_stricmp(s1, s2);
}

int CStrCaseCmp::strNCaseCmp(const char *s1, const char *s2, size_t n) {
    return ::_strnicmp(s1, s2, n);
}


}
}

