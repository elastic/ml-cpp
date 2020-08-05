/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CFuzzyLogic.h>

namespace ml {
namespace maths {
const CFuzzyTruthValue CFuzzyTruthValue::TRUE{1.0};
const CFuzzyTruthValue CFuzzyTruthValue::FALSE{0.0};
const CFuzzyTruthValue CFuzzyTruthValue::OR_UNDETERMINED{0.0, 0.0};
const CFuzzyTruthValue CFuzzyTruthValue::AND_UNDETERMINED{1.0, 1.0};
}
}
