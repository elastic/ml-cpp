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
#include <maths/CBasicStatistics.h>

#include <core/CLogger.h>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ml
{
namespace maths
{

double CBasicStatistics::mean(const TDoubleDoublePr &samples)
{
    return 0.5 * (samples.first + samples.second);
}

double CBasicStatistics::mean(const TDoubleVec &sample)
{
    return  std::accumulate(sample.begin(), sample.end(), 0.0)
          / static_cast<double>(sample.size());
}

double CBasicStatistics::median(const TDoubleVec &dataIn)
{
    if (dataIn.empty())
    {
        return 0.0;
    }

    std::size_t size{dataIn.size()};
    if (size == 1)
    {
        return dataIn[0];
    }

    TDoubleVec data{dataIn};

    // If data size is even (1,2,3,4) then take mean of 2,3 = 2.5
    // If data size is odd (1,2,3,4,5) then take middle value = 3
    double median{0.0};

    // For an odd number of elements, this will get the median element into
    // place.  For an even number of elements, it will get the second element
    // of the middle pair into place.
    bool useMean{size % 2 == 0};
    size_t index{size / 2};
    std::nth_element(data.begin(), data.begin() + index, data.end());

    if (useMean)
    {
        // Since the nth element is the second of the two we need to average,
        // the first element to be averaged will be the largest of all those
        // before the nth one in the vector.
        auto left = std::max_element(data.begin(), data.begin() + index);

        median = (*left + data[index]) / 2.0;
    }
    else
    {
        median = data[index];
    }

    return median;
}

const char CBasicStatistics::INTERNAL_DELIMITER(':');
const char CBasicStatistics::EXTERNAL_DELIMITER(';');

}
}

