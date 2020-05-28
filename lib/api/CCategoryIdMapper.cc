/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CCategoryIdMapper.h>

namespace ml {
namespace api {

void CCategoryIdMapper::acceptPersistInserter(core::CStatePersistInserter& /*inserter*/) const {
    // No-op
}

bool CCategoryIdMapper::acceptRestoreTraverser(core::CStateRestoreTraverser& /*traverser*/) {
    // No-op
    return true;
}
}
}
