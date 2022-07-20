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
#define SIMULTANEOUS_SORT_IMPL                                                 \
    if (std::is_sorted(keys.begin(), keys.end(), comp)) {                      \
        return true;                                                           \
    }                                                                          \
    using TSizeVec = std::vector<std::size_t>;                                 \
    TSizeVec ordering;                                                         \
    ordering.reserve(keys.size());                                             \
    for (std::size_t i = 0; i < keys.size(); ++i) {                            \
        ordering.push_back(i);                                                 \
    }                                                                          \
    std::stable_sort(ordering.begin(), ordering.end(),                         \
                     CIndexLess<KEY_VECTOR, COMP>(keys, comp));                \
    for (std::size_t i = 0; i < ordering.size(); ++i) {                        \
        std::size_t j_ = i;                                                    \
        std::size_t j = ordering[j_];                                          \
        while (i != j) {                                                       \
            using std::swap;                                                   \
            swap(keys[j_], keys[j]);                                           \
            CUSTOM_SWAP_VALUES                                                 \
            ordering[j_] = j_;                                                 \
            j_ = j;                                                            \
            j = ordering[j_];                                                  \
        }                                                                      \
        ordering[j_] = j_;                                                     \
    }                                                                          \
    return true;

template<typename KEY_VECTOR, typename VALUE_VECTOR, typename COMP>
bool COrderings::simultaneousSort(KEY_VECTOR& keys, VALUE_VECTOR& values, const COMP& comp) {
    if (keys.size() != values.size()) {
        return false;
    }
#define CUSTOM_SWAP_VALUES swap(values[j_], values[j]);
    SIMULTANEOUS_SORT_IMPL
#undef CUSTOM_SWAP_VALUES
}

template<typename KEY_VECTOR, typename VALUE1_VECTOR, typename VALUE2_VECTOR, typename COMP>
bool COrderings::simultaneousSort(KEY_VECTOR& keys,
                                  VALUE1_VECTOR& values1,
                                  VALUE2_VECTOR& values2,
                                  const COMP& comp) {
    if (keys.size() != values1.size() || values1.size() != values2.size()) {
        return false;
    }
#define CUSTOM_SWAP_VALUES                                                     \
    swap(values1[j_], values1[j]);                                             \
    swap(values2[j_], values2[j]);
    SIMULTANEOUS_SORT_IMPL
#undef CUSTOM_SWAP_VALUES
}

template<typename KEY_VECTOR, typename VALUE1_VECTOR, typename VALUE2_VECTOR, typename VALUE3_VECTOR, typename COMP>
bool COrderings::simultaneousSort(KEY_VECTOR& keys,
                                  VALUE1_VECTOR& values1,
                                  VALUE2_VECTOR& values2,
                                  VALUE3_VECTOR& values3,
                                  const COMP& comp) {
#define CUSTOM_SWAP_VALUES                                                     \
    swap(values1[j_], values1[j]);                                             \
    swap(values2[j_], values2[j]);                                             \
    swap(values3[j_], values3[j]);
    if (keys.size() != values1.size() || values1.size() != values2.size() ||
        values2.size() != values3.size()) {
        return false;
    }
    SIMULTANEOUS_SORT_IMPL
#undef CUSTOM_SWAP_VALUES
}

template<typename KEY_VECTOR, typename VALUE1_VECTOR, typename VALUE2_VECTOR, typename VALUE3_VECTOR, typename VALUE4_VECTOR, typename COMP>
bool COrderings::simultaneousSort(KEY_VECTOR& keys,
                                  VALUE1_VECTOR& values1,
                                  VALUE2_VECTOR& values2,
                                  VALUE3_VECTOR& values3,
                                  VALUE4_VECTOR& values4,
                                  const COMP& comp) {
    if (keys.size() != values1.size() || values1.size() != values2.size() ||
        values2.size() != values3.size() || values3.size() != values4.size()) {
        return false;
    }
#define CUSTOM_SWAP_VALUES                                                     \
    swap(values1[j_], values1[j]);                                             \
    swap(values2[j_], values2[j]);                                             \
    swap(values3[j_], values3[j]);                                             \
    swap(values4[j_], values4[j]);
    SIMULTANEOUS_SORT_IMPL
#undef CUSTOM_SWAP_VALUES
}

#undef SIMULTANEOUS_SORT_IMPL
}
}
}

#endif // INCLUDED_ml_maths_common_COrderingsSimultaenousSort_h
