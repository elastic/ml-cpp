/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CLeastSquaresOnlineRegression.h>

namespace ml {
namespace maths {
namespace least_squares_online_regression_detail {
const double CMaxCondition<CFloatStorage>::VALUE = 1e7;
}
}
}
