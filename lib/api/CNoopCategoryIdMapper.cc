/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <api/CNoopCategoryIdMapper.h>

namespace {
const std::string EMPTY_STRING;
}

namespace ml {
namespace api {

int CNoopCategoryIdMapper::globalCategoryIdForLocalCategoryId(const std::string& /*categorizerKey*/,
                                                              int localCategoryId) {
    return localCategoryId;
}

const std::string&
CNoopCategoryIdMapper::categorizerKeyForGlobalCategoryId(int /*globalCategoryId*/) const {
    return EMPTY_STRING;
}

int CNoopCategoryIdMapper::localCategoryIdForGlobalCategoryId(int globalCategoryId) const {
    return globalCategoryId;
}

CCategoryIdMapper::TCategoryIdMapperUPtr CNoopCategoryIdMapper::clone() const {
    return std::make_unique<CNoopCategoryIdMapper>(*this);
}

std::string CNoopCategoryIdMapper::printMapping(const std::string& /*categorizerKey*/,
                                                int localCategoryId) const {
    return std::to_string(localCategoryId);
}

std::string CNoopCategoryIdMapper::printMapping(int globalCategoryId) const {
    return std::to_string(globalCategoryId);
}
}
}
