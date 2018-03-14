/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
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

#include <maths/CRestoreParams.h>

#include <maths/CModel.h>

namespace ml {
namespace maths {

STimeSeriesDecompositionRestoreParams::STimeSeriesDecompositionRestoreParams(double decayRate,
                                                                             core_t::TTime minimumBucketLength,
                                                                             std::size_t componentSize) :
    s_DecayRate{decayRate},
    s_MinimumBucketLength{minimumBucketLength},
    s_ComponentSize{componentSize} {
}

SDistributionRestoreParams::SDistributionRestoreParams(maths_t::EDataType dataType,
                                                       double decayRate,
                                                       double minimumClusterFraction,
                                                       double minimumClusterCount,
                                                       double minimumCategoryCount) :
    s_DataType{dataType},
    s_DecayRate{decayRate},
    s_MinimumClusterFraction{minimumClusterFraction},
    s_MinimumClusterCount{minimumClusterCount},
    s_MinimumCategoryCount{minimumCategoryCount} {
}

SModelRestoreParams::SModelRestoreParams(const CModelParams &params,
                                         const STimeSeriesDecompositionRestoreParams &decompositionParams,
                                         const SDistributionRestoreParams &distributionParams) :
    s_Params{params},
    s_DecompositionParams{decompositionParams},
    s_DistributionParams{distributionParams} {
}

}
}
