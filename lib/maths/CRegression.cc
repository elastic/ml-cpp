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

#include <maths/CRegression.h>

namespace ml
{
namespace maths
{
namespace regression_detail
{
const double CMaxCondition<CFloatStorage>::VALUE = 1e7;
}

const double CRegression::MINIMUM_RANGE_TO_PREDICT = 1.0;
}
}
