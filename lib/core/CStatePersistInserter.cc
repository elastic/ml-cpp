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
#include <core/CStatePersistInserter.h>

namespace ml {
namespace core {

CStatePersistInserter::~CStatePersistInserter(void) {}

void CStatePersistInserter::insertValue(const std::string& name,
                                        double value,
                                        CIEEE754::EPrecision precision) {
    this->insertValue(name, CStringUtils::typeToStringPrecise(value, precision));
}

CStatePersistInserter::CAutoLevel::CAutoLevel(const std::string& name,
                                              CStatePersistInserter& inserter)
    : m_Inserter(inserter) {
    m_Inserter.newLevel(name);
}

CStatePersistInserter::CAutoLevel::~CAutoLevel(void) {
    m_Inserter.endLevel();
}
}
}
