/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CNoopCategoryIdMapper.h>

namespace ml {
namespace api {

CGlobalCategoryId CNoopCategoryIdMapper::map(const std::string& /*categorizerKey*/,
                                             model::CLocalCategoryId localCategoryId) {
    return CGlobalCategoryId{localCategoryId.id()};
}

CCategoryIdMapper::TCategoryIdMapperUPtr CNoopCategoryIdMapper::clone() const {
    return std::make_unique<CNoopCategoryIdMapper>(*this);
}
}
}
