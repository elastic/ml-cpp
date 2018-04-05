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

#include <maths/Constants.h>

namespace ml {
namespace maths {

const maths_t::TWeightStyleVec CConstantWeights::COUNT{maths_t::E_SampleCountWeight};
const maths_t::TWeightStyleVec CConstantWeights::COUNT_VARIANCE{maths_t::E_SampleCountVarianceScaleWeight};
const maths_t::TWeightStyleVec CConstantWeights::SEASONAL_VARIANCE{maths_t::E_SampleSeasonalVarianceScaleWeight};
const CConstantWeights::TDouble4Vec CConstantWeights::UNIT{1.0};
const CConstantWeights::TDouble4Vec1Vec CConstantWeights::SINGLE_UNIT{UNIT};

double maxModelPenalty(double numberSamples) {
    return 10.0 + numberSamples;
}
}
}
