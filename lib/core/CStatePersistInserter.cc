/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
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

bool operator==(const std::string& lhs, const CPersistenceTag& rhs) {
    return lhs == rhs.m_ShortTag || lhs == rhs.m_LongTag;
}

bool operator!=(const std::string& lhs, const CPersistenceTag& rhs) {
    return lhs != rhs.m_ShortTag && lhs != rhs.m_LongTag;
}

std::ostream& operator<<(std::ostream& os, const CPersistenceTag& tag) {
    os << tag.m_ShortTag << ": " << tag.m_LongTag;
    return os;
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
