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

#ifndef INCLUDED_ml_maths_common_COrderingsSimultaenousSort_h
#define INCLUDED_ml_maths_common_COrderingsSimultaenousSort_h

#include <maths/common/COrderings.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace ml {
namespace maths {
namespace common {

// The logic in this function is rather subtle because we want to sort the
// collections in place. In particular, we create a sorted collection of
// indices where each index tells us where to get the element from at that
// location and we want to re-order all the collections by that ordering in
// place. If an index matches its position then we can move to the next
// position. Otherwise, we need to swap the items at the index in to its
// position. To work in place we need to do something with the items which
// are displaced. If these are the items required at the swapped in position
// then we are done. Otherwise, we just repeat until we find this position.
// It is easy to verify that this process finds a closed cycle with at most
// N steps. Each time a swap is made at least one more item is in its correct
// place, and we update the ordering accordingly. So the containers are sorted
// in at most O(N) additional steps to the N * log(N) taken to sort the indices.
template<typename PRED, typename K, typename... V>
bool COrderings::simultaneousSortWith(const PRED& pred, K&& keys, V&&... values) {
    if (((keys.size() != values.size()) && ...)) {
        return false;
    }
    if (std::is_sorted(keys.begin(), keys.end(), pred)) {
        return true;
    }

    std::vector<std::size_t> ordering(keys.size());
    std::iota(ordering.begin(), ordering.end(), 0);
    std::stable_sort(ordering.begin(), ordering.end(), [&](auto lhs, auto rhs) {
        return pred(keys[lhs], keys[rhs]);
    });

    using std::swap;
    auto swapValue = [](auto& value, auto j_, auto j) {
        swap(value[j_], value[j]);
    };
    for (std::size_t i = 0; i < ordering.size(); ++i) {
        std::size_t j_ = i;
        std::size_t j = ordering[j_];
        while (i != j) {
            swap(keys[j_], keys[j]);
            (swapValue(values, j_, j), ...);
            ordering[j_] = j_;
            j_ = j;
            j = ordering[j_];
        }
        ordering[j_] = j_;
    }
    return true;
}
}
}
}

#endif // INCLUDED_ml_maths_common_COrderingsSimultaenousSort_h
