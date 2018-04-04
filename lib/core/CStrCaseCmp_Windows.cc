/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CStrCaseCmp.h>

#include <string.h>

namespace ml {
namespace core {

int CStrCaseCmp::strCaseCmp(const char* s1, const char* s2) {
    return ::_stricmp(s1, s2);
}

int CStrCaseCmp::strNCaseCmp(const char* s1, const char* s2, size_t n) {
    return ::_strnicmp(s1, s2, n);
}
}
}
