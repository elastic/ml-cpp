/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CNoopCategoryIdMapper.h>

namespace {
std::string EMPTY_STRING;
}

namespace ml {
namespace api {

CGlobalCategoryId CNoopCategoryIdMapper::map(model::CLocalCategoryId localCategoryId) {
    return CGlobalCategoryId{localCategoryId.id()};
}

const std::string& CNoopCategoryIdMapper::categorizerKey() const {
    return EMPTY_STRING;
}

CCategoryIdMapper::TCategoryIdMapperPtr CNoopCategoryIdMapper::clone() const {
    return std::make_shared<CNoopCategoryIdMapper>(*this);
}
}
}
