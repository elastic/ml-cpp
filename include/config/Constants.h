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

#ifndef INCLUDED_ml_config_Constants_h
#define INCLUDED_ml_config_Constants_h

#include <core/Constants.h>

#include <config/ImportExport.h>

#include <cstddef>
#include <string>

namespace ml {
namespace config {
namespace constants {

//! The initial score to apply to a detector. This is reduced to a minimum
//! of zero, at which point the detector is discarded, based on a set of
//! penalties by the auto-confugration process.
const double MAXIMUM_DETECTOR_SCORE = 100.0;

//! Detectors are assigned a score based on various penalty factors such
//! that lower scores correspond to worse detectors. This defines the
//! resolution of detector scores. In particular, all detector scores take
//! the form of "some integer" x "this epsilon" for the product less than
//! or equal to one.
const double DETECTOR_SCORE_EPSILON = 1e-10;

//! The maximum bucket length we'll configure.
const core_t::TTime LONGEST_BUCKET_LENGTH = core::constants::DAY;

//! The index of a detector's argument field.
const std::size_t ARGUMENT_INDEX = 0u;

//! The index of a detector's by field.
const std::size_t BY_INDEX = 1u;

//! The index of a detector's over field.
const std::size_t OVER_INDEX = 2u;

//! The index of a detector's partition field.
const std::size_t PARTITION_INDEX = 3u;

//! The total number of field indices in a detector.
const std::size_t NUMBER_FIELD_INDICES = 4u;

//! \brief Useful collections of field indices.
class CONFIG_EXPORT CFieldIndices {
public:
    //! The detector partitioning fields, i.e. by, over and partition.
    static const std::size_t PARTITIONING[3];

    //! All detector fields.
    static const std::size_t ALL[4];
};

//! The field name for \p index.
CONFIG_EXPORT const std::string& name(std::size_t index);
}
}
}

#endif // INCLUDED_ml_config_Constants_h
