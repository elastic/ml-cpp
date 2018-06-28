/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CMemory.h>

namespace ml {
namespace core {

const std::string CMemory::EMPTY_STRING;
CMemory::CAnyVisitor CMemory::ms_AnyVisitor = CMemory::CAnyVisitor();

const std::string CMemoryDebug::EMPTY_STRING;
CMemoryDebug::CAnyVisitor CMemoryDebug::ms_AnyVisitor = CMemoryDebug::CAnyVisitor();
}
}
