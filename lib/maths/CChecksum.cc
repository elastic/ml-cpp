/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CChecksum.h>

namespace ml {
namespace maths {
namespace checksum_detail {
const std::hash<std::vector<bool>> CChecksumImpl<ContainerChecksum>::ms_VectorBoolHasher{};
}
}
}
