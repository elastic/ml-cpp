/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/COutputHandler.h>

namespace ml {
namespace api {

// Initialise statics
const COutputHandler::TStrVec COutputHandler::EMPTY_FIELD_NAMES;
const COutputHandler::TStrStrUMap COutputHandler::EMPTY_FIELD_OVERRIDES;

void COutputHandler::newOutputStream() {
    // NOOP unless overridden
}

bool COutputHandler::fieldNames(const TStrVec& fieldNames) {
    return this->fieldNames(fieldNames, EMPTY_FIELD_NAMES);
}

bool COutputHandler::writeRow(const TStrStrUMap& dataRowFields) {
    // Since the overrides are checked first, but we know there aren't any, it's
    // most efficient to pretend everything's an override
    return this->writeRow(EMPTY_FIELD_OVERRIDES, dataRowFields);
}

void COutputHandler::finalise() {
    // NOOP unless overridden
}

bool COutputHandler::restoreState(core::CDataSearcher& /* restoreSearcher */,
                                  core_t::TTime& /* completeToTime */) {
    // NOOP unless overridden
    return true;
}

bool COutputHandler::persistState(core::CDataAdder& /* persister */) {
    // NOOP unless overridden
    return true;
}

bool COutputHandler::periodicPersistState(CBackgroundPersister& /* persister */) {
    // NOOP unless overridden
    return true;
}

bool COutputHandler::isPersistenceNeeded(const std::string& /*description*/) const {
    // NOOP unless overridden
    return false;
}

COutputHandler::CPreComputedHash::CPreComputedHash(size_t hash) : m_Hash(hash) {
}

size_t COutputHandler::CPreComputedHash::operator()(const std::string&) const {
    return m_Hash;
}

bool COutputHandler::consumesControlMessages() {
    return false;
}
}
}
