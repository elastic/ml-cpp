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

#include <maths/analytics/CBoostedTreeUtils.h>

#include <core/CContainerPrinter.h>
#include <core/Concurrency.h>

#include <maths/analytics/CBoostedTree.h>
#include <maths/analytics/CBoostedTreeHyperparameters.h>
#include <maths/analytics/CBoostedTreeLoss.h>
#include <maths/analytics/CDataFrameCategoryEncoder.h>

#include <maths/common/CLinearAlgebraEigen.h>
#include <maths/common/MathsTypes.h>

#if defined(__SSE4_2__)

#include <xmmintrin.h>

#elif defined(__ARM_NEON__)

#include <arm_neon.h>

using __m128 = float32x4_t;

#define _mm_load_ps1 _mm_load1_ps

#define _mm_load_ps(x) vreinterpretq_m128_f32(vld1q_f32(x))

#define _mm_cmplt_ps(lhs, rhs)                                                 \
    vreinterpretq_m128_u32(                                                    \
        vcltq_f32(vreinterpretq_f32_m128(lhs), vreinterpretq_f32_m128(rhs)))

static constexpr int32x4_t _MM_MOVEMASK_PS_SHIFT{0, 1, 2, 3};

static inline __attribute__((always_inline)) int _mm_movemask_ps(__m128 a) {
    // We only build for aarch 64.
    uint32x4_t input = vreinterpretq_u32_m128(a);
    uint32x4_t tmp = vshrq_n_u32(input, 31);
    return vaddvq_u32(vshlq_u32(tmp, _MM_MOVEMASK_PS_SHIFT));
}

#else

using __m128 = std::array<float, 4>;

#define _mm_load_ps1(x) __m128{*(x), *(x), *(x), *(x)};

#define _mm_load_ps(x) __m128{*(x), *((x) + 1), *((x) + 2), *((x) + 3)};

#define _mm_cmplt_ps(lhs, rhs)                                                   \
    std::array<int, 4>{(lhs)[0] < (rhs)[0] ? 1 : 0, (lhs)[1] < (rhs)[1] ? 1 : 0, \
                       (lhs)[2] < (rhs)[2] ? 1 : 0, (lhs)[3] < (rhs)[3] ? 1 : 0};

#define _mm_movemask_ps(x)                                                     \
    (((x)[0] << 3) + ((x)[1] << 2) + ((x)[2] << 1) + (x)[3])

#endif

namespace ml {
namespace maths {
namespace analytics {
namespace boosted_tree_detail {
using namespace boosted_tree;
using TSizeAry = std::array<std::size_t, 16>;

namespace {
constexpr TSizeAry BRANCH{4, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 1, 0};
std::size_t selectBranch(const float* values, __m128 vecx) {
    auto vecv = _mm_load_ps(values);
    auto less = _mm_cmplt_ps(vecx, vecv);
    auto mask = _mm_movemask_ps(less);
    return BRANCH[mask];
}
}

CSearchTree::CSearchTree(const TFloatVec& values) : m_Size{values.size()} {
    if (m_Size > 0) {
        m_Min = values[0].cstorage();
        std::size_t n{nextPow5(values.size())};
        m_InitialTreeSize = n / 5;
        m_Values.reserve(m_Size + 4);
        this->build(values, 0, n);
        LOG_TRACE(<< "flat tree = " << m_Values);
    }
    LOG_TRACE(<< "size = " << m_Size << ", padded size = " << m_InitialTreeSize);
}

std::size_t CSearchTree::upperBound(common::CFloatStorage x) const {
    // These branch should be predictably false most of the time and so almost free.
    if (m_Size == 0 || x < m_Min) {
        return 0;
    }
    if (x >= INF) {
        return m_Size;
    }

    std::size_t node{0};
    std::size_t offset{0};
    auto vecx = _mm_load_ps1(&x.cstorage());

    for (std::size_t treeSize = m_InitialTreeSize; treeSize > 1; treeSize /= 5) {
        std::size_t branch{selectBranch(&m_Values[node], vecx)};
        LOG_TRACE(<< node << " = " << this->printNode(node) << ", branch = " << branch);

        // Note that node is a multiple of 4. This follows from the fact that
        // the step size 4 + 5^n - 1 is a multiple of 4 which can be shown by
        // induction:
        //
        //   5^n - 1 = 5 * (5^(n - 1) - 1) + 4 and 5 - 1 = 1 * 4.
        //
        // This means that since m_Values are 16 byte aligned the values at node
        // are 16 byte aligned and we can safely read them using an aligned load.
        node += 4 + (treeSize - 1) * branch;

        // Each branch point which is greater than x is out of order w.r.t. this
        // point and must be subtracted from node to get the correct upper bound.
        offset += 4 - branch;
    }

    std::size_t branch{selectBranch(&m_Values[node], vecx)};
    LOG_TRACE(<< "x = " << x << ", node = " << node << "/" << this->printNode(node)
              << ", branch = " << branch << ", offset = " << offset);

    return node + branch + 1 - offset;
}

void CSearchTree::build(const TFloatVec& values, std::size_t a, std::size_t b) {
    // Building depth first gives the lower expected jump size.
    std::size_t stride{(b - a) / 5};
    if (stride == 0) {
        return;
    }
    LOG_TRACE(<< "stride = " << stride);
    for (std::size_t i = a + stride; i < b; i += stride) {
        m_Values.push_back(i < values.size() ? values[i].cstorage() : INF);
    }
    for (std::size_t i = a; i < b; i += stride) {
        if (i < values.size()) {
            this->build(values, i, i + stride);
        }
    }
}

std::string CSearchTree::printNode(std::size_t node) const {
    return core::CContainerPrinter::print(m_Values.begin() + node,
                                          m_Values.begin() + node + 4);
}

std::size_t CSearchTree::nextPow5(std::size_t n) {
    std::size_t result{5};
    for (/**/; result < n; result *= 5) {
    }
    return result;
}

const CBoostedTreeNode& root(const std::vector<CBoostedTreeNode>& tree) {
    return tree[rootIndex()];
}

CBoostedTreeNode& root(std::vector<CBoostedTreeNode>& tree) {
    return tree[rootIndex()];
}

void zeroPrediction(const TRowRef& row, const TSizeVec& extraColumns, std::size_t numberLossParameters) {
    for (std::size_t i = 0; i < numberLossParameters; ++i) {
        row.writeColumn(extraColumns[E_Prediction] + i, 0.0);
    }
}

void writePrediction(const TRowRef& row,
                     const TSizeVec& extraColumns,
                     std::size_t numberLossParameters,
                     const TMemoryMappedFloatVector& value) {
    for (std::size_t i = 0; i < numberLossParameters; ++i) {
        row.writeColumn(extraColumns[E_Prediction] + i, value(i));
    }
}

void writePreviousPrediction(const TRowRef& row,
                             const TSizeVec& extraColumns,
                             std::size_t numberLossParameters,
                             const TMemoryMappedFloatVector& value) {
    for (std::size_t i = 0; i < numberLossParameters; ++i) {
        row.writeColumn(extraColumns[E_PreviousPrediction] + i, value(i));
    }
}

void zeroLossGradient(const TRowRef& row, const TSizeVec& extraColumns, std::size_t numberLossParameters) {
    for (std::size_t i = 0; i < numberLossParameters; ++i) {
        row.writeColumn(extraColumns[E_Gradient] + i, 0.0);
    }
}

void writeLossGradient(const TRowRef& row,
                       const CEncodedDataFrameRowRef& encodedRow,
                       bool newExample,
                       const TSizeVec& extraColumns,
                       const CLoss& loss,
                       const TMemoryMappedFloatVector& prediction,
                       double actual,
                       double weight) {
    auto writer = [&row, &extraColumns](std::size_t i, double value) {
        row.writeColumn(extraColumns[E_Gradient] + i, value);
    };
    // We wrap the writer in another lambda which we know takes advantage
    // of std::function small size optimization to avoid heap allocations.
    loss.gradient(encodedRow, newExample, prediction, actual,
                  [&writer](std::size_t i, double value) { writer(i, value); }, weight);
}

void zeroLossCurvature(const TRowRef& row, const TSizeVec& extraColumns, std::size_t numberLossParameters) {
    for (std::size_t i = 0, size = lossHessianUpperTriangleSize(numberLossParameters);
         i < size; ++i) {
        row.writeColumn(extraColumns[E_Curvature] + i, 0.0);
    }
}

void writeLossCurvature(const TRowRef& row,
                        const CEncodedDataFrameRowRef& encodedRow,
                        bool newExample,
                        const TSizeVec& extraColumns,
                        const CLoss& loss,
                        const TMemoryMappedFloatVector& prediction,
                        double actual,
                        double weight) {
    auto writer = [&row, &extraColumns](std::size_t i, double value) {
        row.writeColumn(extraColumns[E_Curvature] + i, value);
    };
    // We wrap the writer in another lambda which we know takes advantage
    // of std::function small size optimization to avoid heap allocations.
    loss.curvature(encodedRow, newExample, prediction, actual,
                   [&writer](std::size_t i, double value) { writer(i, value); }, weight);
}
}
}
}
}
