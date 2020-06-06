/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CCategoryIdMapper.h>

namespace ml {
namespace api {

CCategoryIdMapper::TGlobalCategoryIdVec
CCategoryIdMapper::mapVec(const std::string& categorizerKey,
                          const TLocalCategoryIdVec& localCategoryIds) {
    TGlobalCategoryIdVec mapped;
    mapped.reserve(localCategoryIds.size());

    for (const auto& localCategoryId : localCategoryIds) {
        mapped.emplace_back(this->map(categorizerKey, localCategoryId));
    }

    return mapped;
}

void CCategoryIdMapper::acceptPersistInserter(core::CStatePersistInserter& /*inserter*/) const {
    // No-op
}

bool CCategoryIdMapper::acceptRestoreTraverser(core::CStateRestoreTraverser& /*traverser*/) {
    // No-op
    return true;
}
}
}
