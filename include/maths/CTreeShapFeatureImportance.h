/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CTreeShapFeatureImportance_h
#define INCLUDED_ml_maths_CTreeShapFeatureImportance_h

#include <maths/CBoostedTree.h>
#include <maths/ImportExport.h>

#include <vector>

namespace ml {
namespace maths {

// TODO add extensive comments

class MATHS_EXPORT CTreeShapFeatureImportance {
public:
    using TTree = std::vector<CBoostedTreeNode>;
    using TTreeVec = std::vector<TTree>;
    using TIntVec = std::vector<int>;
    using TDoubleVec = std::vector<double>;
    using TDoubleVecVec = std::vector<TDoubleVec>;

    struct SPath {
        explicit SPath(std::size_t length)
            : s_FractionOnes(length), s_FractionZeros(length),
              s_FeatureIndex(length, -1), s_Scale(length), s_NextIndex(0),
              s_MaxLength(length) {}

        void extend(int featureIndex, double fractionZero, double fractionOne) {
            if (s_NextIndex < s_MaxLength) {
                s_FeatureIndex[s_NextIndex] = featureIndex;
                s_FractionZeros[s_NextIndex] = fractionZero;
                s_FractionOnes[s_NextIndex] = fractionOne;
                if (s_NextIndex == 0) {
                    s_Scale[s_NextIndex] = 1.0;
                } else {
                    s_Scale[s_NextIndex] = 0.0;
                }
                ++s_NextIndex;
            }
        }

        void reduce(int pathIndex) {
            for (int i = pathIndex; i < this->depth(); ++i) {
                s_FeatureIndex[i] = s_FeatureIndex[i + 1];
                s_FractionZeros[i] = s_FractionZeros[i + 1];
                s_FractionOnes[i] = s_FractionOnes[i + 1];
            }
            --s_NextIndex;
        }

        double fractionOnes(std::size_t pathIndex) const {
            return s_FractionOnes[pathIndex];
        }

        double fractionZeros(std::size_t pathIndex) const {
            return s_FractionZeros[pathIndex];
        }

        int featureIndex(std::size_t pathIndex) const {
            return s_FeatureIndex[pathIndex];
        }

        double scale(std::size_t pathIndex) const { return s_Scale[pathIndex]; }

        //! Current depth in the tree
        std::size_t depth() const { return s_NextIndex - 1; };

        TDoubleVec s_FractionOnes;
        TDoubleVec s_FractionZeros;
        TIntVec s_FeatureIndex;
        TDoubleVec s_Scale;
        std::size_t s_NextIndex;
        std::size_t s_MaxLength;
    };

public:
    explicit CTreeShapFeatureImportance(TTreeVec trees, std::size_t threads = 1);

    //! Compute SHAP values for the data in \p frame using the specified \p encoder.
    //!
    //!\param[in] numberFeatures number of features. If set to -1, it's assumed that the number of feature is equal to the
    //! number of columns in the \p frame.
    //!\return The sum of magnitudes of SHAP values for every feature.
    TDoubleVec shap(core::CDataFrame& frame,
                    const CDataFrameCategoryEncoder& encoder,
                    std::size_t numberFeatures,
                    std::size_t offset);

    static CTreeShapFeatureImportance::TDoubleVec
    samplesPerNode(const TTree& tree,
                   const core::CDataFrame& frame,
                   const CDataFrameCategoryEncoder& encoder,
                   std::size_t numThreads);

    //! Recursively computes inner node values as weighted average of the children (leaf) values
    //! \returns The maximum depth the the tree.
    static std::size_t updateNodeValues(TTree& tree,
                                        std::size_t nodeIndex,
                                        const TDoubleVec& samplesPerNode,
                                        std::size_t depth);

    TTreeVec& trees() { return m_Trees; };

private:
    static void shapRecursive(const TTree& tree,
                              const TDoubleVec& samplesPerNode,
                              const CDataFrameCategoryEncoder& encoder,
                              const CEncodedDataFrameRowRef& encodedRow,
                              SPath splitPath,
                              std::size_t nodeIndex,
                              double parentFractionZero,
                              double parentFractionOne,
                              int parentFeatureIndex,
                              std::size_t offset,
                              core::CDataFrame::TRowItr& row);
    static void extendPath(SPath& path, double fractionZero, double fractionOne, int featureIndex);
    static double sumUnwoundPath(const SPath& path, int pathIndex);
    static void unwindPath(SPath& path, int pathIndex);

private:
    TTreeVec m_Trees;
    std::size_t m_NumberThreads;
    TDoubleVecVec m_SamplesPerNode;
};
}
}

#endif // INCLUDED_ml_maths_CTreeShapFeatureImportance_h
