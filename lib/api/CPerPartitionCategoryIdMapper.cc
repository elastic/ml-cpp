/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CPerPartitionCategoryIdMapper.h>

#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>

namespace {
const std::string HIGHEST_GLOBAL_ID_TAG{"a"};
const std::string CATEGORIZER_KEY_TAG{"b"};
const std::string GLOBAL_ID_TAG{"c"};
}

namespace ml {
namespace api {

CGlobalCategoryId CPerPartitionCategoryIdMapper::map(const std::string& categorizerKey,
                                                     model::CLocalCategoryId localCategoryId) {
    if (localCategoryId.isValid() == false) {
        return CGlobalCategoryId{localCategoryId.id()};
    }
    auto iter = m_Mapper.find(categorizerKey);
    if (iter == m_Mapper.end()) {
        iter = m_Mapper.emplace(categorizerKey, TGlobalCategoryIdVec{}).first;
    }
    TGlobalCategoryIdVec& partitionVec{iter->second};
    std::size_t index{localCategoryId.index()};
    if (index > partitionVec.size()) {
        LOG_ERROR(<< "Bad category mappings: " << (index - partitionVec.size())
                  << " local to global category ID mappings missing for partition "
                  << categorizerKey);
        partitionVec.resize(index);
    }
    if (index == partitionVec.size()) {
        partitionVec.emplace_back(++m_HighestGlobalId, iter->first, localCategoryId);
    }
    return partitionVec[index];
}

CCategoryIdMapper::TCategoryIdMapperUPtr CPerPartitionCategoryIdMapper::clone() const {
    return std::make_unique<CPerPartitionCategoryIdMapper>(*this);
}

void CPerPartitionCategoryIdMapper::acceptPersistInserter(core::CStatePersistInserter& inserter) const {

    inserter.insertValue(HIGHEST_GLOBAL_ID_TAG, m_HighestGlobalId);

    for (const auto& mapEntry : m_Mapper) {
        inserter.insertValue(CATEGORIZER_KEY_TAG, mapEntry.first);
        for (const auto& globalCategoryId : mapEntry.second) {
            inserter.insertValue(GLOBAL_ID_TAG, globalCategoryId.globalId());
        }
    }
}

bool CPerPartitionCategoryIdMapper::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {

    m_HighestGlobalId = 0;
    m_Mapper.clear();

    auto iter = m_Mapper.end();
    do {
        const std::string& name{traverser.name()};
        if (name == HIGHEST_GLOBAL_ID_TAG) {
            if (core::CStringUtils::stringToType(traverser.value(), m_HighestGlobalId) == false) {
                LOG_ERROR(<< "Invalid highest global ID in " << traverser.value());
                return false;
            }
        } else if (name == CATEGORIZER_KEY_TAG) {
            iter = m_Mapper.emplace(traverser.value(), TGlobalCategoryIdVec{}).first;
        } else if (name == GLOBAL_ID_TAG) {
            if (iter == m_Mapper.end()) {
                LOG_ERROR(<< "Global ID seen before categorizer key "
                          << traverser.value());
                return false;
            }
            int globalId{model::CLocalCategoryId::SOFT_CATEGORIZATION_FAILURE_ERROR};
            if (core::CStringUtils::stringToType(traverser.value(), globalId) == false) {
                LOG_ERROR(<< "Invalid global ID in " << traverser.value());
                return false;
            }
            iter->second.emplace_back(globalId, iter->first,
                                      model::CLocalCategoryId{iter->second.size()});
        }
    } while (traverser.next());

    return true;
}
}
}
