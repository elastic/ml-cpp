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

#include <api/CPerPartitionCategoryIdMapper.h>

#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>

namespace {
const std::string GLOBAL_ID_TAG{"a"};
}

namespace ml {
namespace api {

CPerPartitionCategoryIdMapper::CPerPartitionCategoryIdMapper(std::string categorizerKey,
                                                             TNextGlobalIdSupplier nextGlobalIdSupplier)
    : m_CategorizerKey{std::move(categorizerKey)}, m_NextGlobalIdSupplier{
                                                       std::move(nextGlobalIdSupplier)} {
}

CGlobalCategoryId CPerPartitionCategoryIdMapper::map(model::CLocalCategoryId localCategoryId) {
    if (localCategoryId.isValid() == false) {
        return CGlobalCategoryId{localCategoryId.id()};
    }
    std::size_t index{localCategoryId.index()};
    if (index > m_Mappings.size()) {
        LOG_ERROR(<< "Bad category mappings: " << (index - m_Mappings.size())
                  << " local to global category ID mappings missing for partition "
                  << m_CategorizerKey);
        m_Mappings.resize(index);
    }
    if (index == m_Mappings.size()) {
        m_Mappings.emplace_back(m_NextGlobalIdSupplier(), m_CategorizerKey, localCategoryId);
    }
    return m_Mappings[index];
}

const std::string& CPerPartitionCategoryIdMapper::categorizerKey() const {
    return m_CategorizerKey;
}

CCategoryIdMapper::TCategoryIdMapperPtr CPerPartitionCategoryIdMapper::clone() const {
    return std::make_shared<CPerPartitionCategoryIdMapper>(*this);
}

void CPerPartitionCategoryIdMapper::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    for (const auto& globalCategoryId : m_Mappings) {
        inserter.insertValue(GLOBAL_ID_TAG, globalCategoryId.globalId());
    }
}

bool CPerPartitionCategoryIdMapper::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {

    m_Mappings.clear();

    do {
        const std::string& name{traverser.name()};
        if (name == GLOBAL_ID_TAG) {
            int globalId{model::CLocalCategoryId::SOFT_CATEGORIZATION_FAILURE_ERROR};
            if (core::CStringUtils::stringToType(traverser.value(), globalId) == false) {
                LOG_ERROR(<< "Invalid global ID in " << traverser.value());
                return false;
            }
            m_Mappings.emplace_back(globalId, m_CategorizerKey,
                                    model::CLocalCategoryId{m_Mappings.size()});
        }
    } while (traverser.next());

    return true;
}
}
}
