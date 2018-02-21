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
#include <api/COutputHandler.h>


namespace ml
{
namespace api
{

// Initialise statics
const COutputHandler::TStrVec     COutputHandler::EMPTY_FIELD_NAMES;
const COutputHandler::TStrStrUMap COutputHandler::EMPTY_FIELD_OVERRIDES;


COutputHandler::COutputHandler(void)
{
}

COutputHandler::~COutputHandler(void)
{
}

void COutputHandler::newOutputStream(void)
{
    // NOOP unless overridden
}

bool COutputHandler::fieldNames(const TStrVec &fieldNames)
{
    return this->fieldNames(fieldNames, EMPTY_FIELD_NAMES);
}

bool COutputHandler::writeRow(const TStrStrUMap &dataRowFields)
{
    // Since the overrides are checked first, but we know there aren't any, it's
    // most efficient to pretend everything's an override
    return this->writeRow(EMPTY_FIELD_OVERRIDES, dataRowFields);
}

void COutputHandler::finalise(void)
{
    // NOOP unless overridden
}

bool COutputHandler::restoreState(core::CDataSearcher & /* restoreSearcher */,
                                  core_t::TTime & /* completeToTime */)
{
    // NOOP unless overridden
    return true;
}

bool COutputHandler::persistState(core::CDataAdder & /* persister */)
{
    // NOOP unless overridden
    return true;
}

bool COutputHandler::periodicPersistState(CBackgroundPersister & /* persister */)
{
    // NOOP unless overridden
    return true;
}

COutputHandler::CPreComputedHash::CPreComputedHash(size_t hash)
    : m_Hash(hash)
{
}

size_t COutputHandler::CPreComputedHash::operator()(const std::string &) const
{
    return m_Hash;
}

bool COutputHandler::consumesControlMessages()
{
    return false;
}


}
}

