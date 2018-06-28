/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <maths/CBasicStatistics.h>

#include <core/CLogger.h>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ml {
namespace maths {
namespace {

//! Compute the median reordering \p samples in the process.
double medianInPlace(std::vector<double>& data) {
    std::size_t size{data.size()};

    // If sample size is even (1,2,3,4) then take mean of 2,3 = 2.5
    // If sample size is odd (1,2,3,4,5) then take middle value = 3
    bool useMean{size % 2 == 0};

    // For an odd number of elements, this will get the median element into
    // place.  For an even number of elements, it will get the second element
    // of the middle pair into place.
    size_t index{size / 2};
    std::nth_element(data.begin(), data.begin() + index, data.end());

    if (useMean) {
        // Since the nth element is the second of the two we need to average,
        // the first element to be averaged will be the largest of all those
        // before the nth one in the vector.
        auto left = std::max_element(data.begin(), data.begin() + index);

        return (*left + data[index]) / 2.0;
    }

    return data[index];
}
}

double CBasicStatistics::mean(const TDoubleDoublePr& data) {
    return 0.5 * (data.first + data.second);
}

double CBasicStatistics::mean(const TDoubleVec& data) {
    return std::accumulate(data.begin(), data.end(), 0.0) /
           static_cast<double>(data.size());
}

double CBasicStatistics::median(const TDoubleVec& data_) {
    if (data_.empty()) {
        return 0.0;
    }
    if (data_.size() == 1) {
        return data_[0];
    }
    TDoubleVec data{data_};
    return medianInPlace(data);
}

double CBasicStatistics::mad(const TDoubleVec& data_) {
    if (data_.size() < 2) {
        return 0.0;
    }
    TDoubleVec data{data_};
    double median{medianInPlace(data)};
    for (auto& datum : data) {
        datum = std::fabs(datum - median);
    }
    return medianInPlace(data);
}

const char CBasicStatistics::INTERNAL_DELIMITER(':');
const char CBasicStatistics::EXTERNAL_DELIMITER(';');
}
}
