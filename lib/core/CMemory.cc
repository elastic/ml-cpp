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
#include <core/CMemory.h>


namespace ml {
namespace core {

const std::string    CMemory::EMPTY_STRING;
CMemory::CAnyVisitor CMemory::ms_AnyVisitor = CMemory::CAnyVisitor();


const std::string         CMemoryDebug::EMPTY_STRING;
CMemoryDebug::CAnyVisitor CMemoryDebug::ms_AnyVisitor = CMemoryDebug::CAnyVisitor();

}
}

