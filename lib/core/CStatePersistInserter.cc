/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CStatePersistInserter.h>

namespace ml {
namespace core {

CStatePersistInserter::~CStatePersistInserter() {
}

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

CStatePersistInserter::CAutoLevel::~CAutoLevel() {
    m_Inserter.endLevel();
}
}
}
