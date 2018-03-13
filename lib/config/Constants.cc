/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#include <config/Constants.h>

namespace ml {
namespace config {
namespace constants {
namespace {

const std::string FIELD_NAME[] = {std::string("argument"),
                                  std::string("by"),
                                  std::string("over"),
                                  std::string("partition")};
}

const std::size_t CFieldIndices::PARTITIONING[] = {BY_INDEX, OVER_INDEX, PARTITION_INDEX};
const std::size_t CFieldIndices::ALL[] = {ARGUMENT_INDEX, BY_INDEX, OVER_INDEX, PARTITION_INDEX};

const std::string &name(std::size_t index) { return FIELD_NAME[index]; }
}
}
}
