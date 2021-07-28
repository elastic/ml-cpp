/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <maths/CFuzzyLogic.h>

namespace ml {
namespace maths {
const CFuzzyTruthValue CFuzzyTruthValue::TRUE_VALUE{1.0};
const CFuzzyTruthValue CFuzzyTruthValue::FALSE_VALUE{0.0};
const CFuzzyTruthValue CFuzzyTruthValue::OR_UNDETERMINED_VALUE{0.0, 0.0};
const CFuzzyTruthValue CFuzzyTruthValue::AND_UNDETERMINED_VALUE{1.0, 1.0};
}
}
