/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CFuzzyLogic.h>

namespace ml {
namespace maths {
//! This is choosen so TRUE && expression is true iff expression is true.
const CFuzzyTruthValue CFuzzyTruthValue::TRUE{1.0, 1.0};
//! This is choosen so FALSE || expression is true iff expression is true.
const CFuzzyTruthValue CFuzzyTruthValue::FALSE{0.0, 0.0};
}
}
