/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CPerPartitionCategoryIdMapper.h>

#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>

#include <model/CDataCategorizer.h>

// Using boost functionality rather than std here is dictated by boost::multi_index
#include <boost/ref.hpp>
#include <boost/tuple/tuple.hpp>

#include <sstream>

namespace {
const std::string HIGHEST_GLOBAL_ID_TAG{"a"};
const std::string GLOBAL_ID_TAG{"b"};
const std::string CATEGORIZER_KEY_TAG{"c"};
const std::string LOCAL_ID_TAG{"d"};
const std::string EMPTY_STRING;
}

namespace ml {
namespace api {

int CPerPartitionCategoryIdMapper::globalCategoryIdForLocalCategoryId(const std::string& categorizerKey,
                                                                      int localCategoryId) {
    if (localCategoryId < 1) {
        return localCategoryId;
    }
    auto iter = m_Mapper.get<SLocalKey>().find(
        boost::make_tuple(boost::cref(categorizerKey), localCategoryId));
    if (iter == m_Mapper.get<SLocalKey>().end()) {
        ++m_HighestGlobalId;
        m_Mapper.emplace(m_HighestGlobalId, categorizerKey, localCategoryId);
        return m_HighestGlobalId;
    }
    return iter->first;
}

const std::string&
CPerPartitionCategoryIdMapper::categorizerKeyForGlobalCategoryId(int globalCategoryId) const {
    if (globalCategoryId < 1) {
        return EMPTY_STRING;
    }
    auto iter = m_Mapper.get<SGlobalKey>().find(globalCategoryId);
    if (iter == m_Mapper.get<SGlobalKey>().end()) {
        return EMPTY_STRING;
    }
    return iter->second;
}

int CPerPartitionCategoryIdMapper::localCategoryIdForGlobalCategoryId(int globalCategoryId) const {
    if (globalCategoryId < 1) {
        return globalCategoryId;
    }
    auto iter = m_Mapper.get<SGlobalKey>().find(globalCategoryId);
    if (iter == m_Mapper.get<SGlobalKey>().end()) {
        return model::CDataCategorizer::HARD_CATEGORIZATION_FAILURE_ERROR;
    }
    return iter->third;
}

CCategoryIdMapper::TCategoryIdMapperUPtr CPerPartitionCategoryIdMapper::clone() const {
    return std::make_unique<CPerPartitionCategoryIdMapper>(*this);
}

std::string CPerPartitionCategoryIdMapper::printMapping(const std::string& categorizerKey,
                                                        int localCategoryId) const {
    if (localCategoryId < 1) {
        return std::to_string(localCategoryId);
    }
    auto iter = m_Mapper.get<SLocalKey>().find(
        boost::make_tuple(boost::cref(categorizerKey), localCategoryId));
    if (iter == m_Mapper.get<SLocalKey>().end()) {
        return "unknown";
    }
    return CPerPartitionCategoryIdMapper::printMapping(*iter);
}

std::string CPerPartitionCategoryIdMapper::printMapping(int globalCategoryId) const {
    if (globalCategoryId < 1) {
        return std::to_string(globalCategoryId);
    }
    auto iter = m_Mapper.get<SGlobalKey>().find(globalCategoryId);
    if (iter == m_Mapper.get<SGlobalKey>().end()) {
        return "unknown";
    }
    return CPerPartitionCategoryIdMapper::printMapping(*iter);
}

void CPerPartitionCategoryIdMapper::acceptPersistInserter(core::CStatePersistInserter& inserter) const {

    inserter.insertValue(HIGHEST_GLOBAL_ID_TAG, m_HighestGlobalId);

    for (const auto& mapping : m_Mapper) {
        inserter.insertValue(GLOBAL_ID_TAG, mapping.first);
        inserter.insertValue(CATEGORIZER_KEY_TAG, mapping.second);
        inserter.insertValue(LOCAL_ID_TAG, mapping.third);
    }
}

bool CPerPartitionCategoryIdMapper::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {

    m_HighestGlobalId = 0;
    m_Mapper.clear();

    TIntStrIntTriple workMapping;

    do {
        const std::string& name{traverser.name()};
        if (name == HIGHEST_GLOBAL_ID_TAG) {
            if (core::CStringUtils::stringToType(traverser.value(), m_HighestGlobalId) == false) {
                LOG_ERROR(<< "Invalid highest global ID in " << traverser.value());
                return false;
            }
        } else if (name == GLOBAL_ID_TAG) {
            if (core::CStringUtils::stringToType(traverser.value(), workMapping.first) == false) {
                LOG_ERROR(<< "Invalid global ID in " << traverser.value());
                return false;
            }
        } else if (name == CATEGORIZER_KEY_TAG) {
            workMapping.second = traverser.value();
        } else if (name == LOCAL_ID_TAG) {
            if (core::CStringUtils::stringToType(traverser.value(), workMapping.third) == false) {
                LOG_ERROR(<< "Invalid local ID in " << traverser.value());
                return false;
            }

            // Insert on the basis that the contents of the triple was
            // persisted in the order first, second, third, so on restoring
            // third we have a complete mapping
            m_Mapper.emplace(std::move(workMapping));
        }
    } while (traverser.next());

    return true;
}

std::string CPerPartitionCategoryIdMapper::printMapping(const TIntStrIntTriple& mapping) {
    std::ostringstream strm;
    strm << mapping.second << '/' << mapping.third << ';' << mapping.first;
    return strm.str();
}
}
}
