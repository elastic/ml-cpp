/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <config/Constants.h>

namespace ml {
namespace config {
namespace constants {
namespace {

const std::string FIELD_NAME[] = {std::string("argument"), std::string("by"),
                                  std::string("over"), std::string("partition")};
}

const std::size_t CFieldIndices::PARTITIONING[] = {BY_INDEX, OVER_INDEX, PARTITION_INDEX};
const std::size_t CFieldIndices::ALL[] = {ARGUMENT_INDEX, BY_INDEX, OVER_INDEX, PARTITION_INDEX};

const std::string& name(std::size_t index) {
    return FIELD_NAME[index];
}
}
}
}
