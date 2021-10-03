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

#include <api/CCategoryIdMapper.h>

namespace ml {
namespace api {

CCategoryIdMapper::TGlobalCategoryIdVec
CCategoryIdMapper::mapVec(const TLocalCategoryIdVec& localCategoryIds) {
    TGlobalCategoryIdVec mapped;
    mapped.reserve(localCategoryIds.size());

    for (const auto& localCategoryId : localCategoryIds) {
        mapped.emplace_back(this->map(localCategoryId));
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
