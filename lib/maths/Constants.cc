/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/Constants.h>

namespace ml {
namespace maths {

double maxModelPenalty(double numberSamples) {
    return 10.0 + numberSamples;
}
}
}
