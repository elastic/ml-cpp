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
        SPath(std::size_t length)
            : s_FractionOnes(length), s_FractionZeros(length),
              s_FeatureIndex(length, -1), s_Scale(length), s_NextIndex(0),
              s_MaxLength(length) {}
        ~SPath() = default;

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
    explicit CTreeShapFeatureImportance(TTreeVec trees);

    //! Compute SHAP values for the data in \p frame using the specified \p encoder.
    //!
    //!\param[in] numberFeatures number of features. If set to -1, it's assument that the number of feature is equal to the
    //! number of columns in the \p frame.
    //!\return pair shap values for all points, sum of magnitues of shap values for every feature.
    std::pair<CTreeShapFeatureImportance::TDoubleVecVec, CTreeShapFeatureImportance::TDoubleVec>
    shap(const core::CDataFrame& frame,
         const CDataFrameCategoryEncoder& encoder,
         int numberFeatures = -1);

    CTreeShapFeatureImportance::TDoubleVec
    samplesPerNode(const TTree& tree,
                   const core::CDataFrame& frame,
                   const CDataFrameCategoryEncoder& encoder) const;

    //! Recursively computes inner node values as weighted average of the children (leaf) values
    //! Returns maximum depth the the tree
    std::size_t updateNodeValues(TTree& tree,
                                 std::size_t nodeIndex,
                                 const TDoubleVec& samplesPerNode,
                                 std::size_t depth);

    TTreeVec& trees() { return m_Trees; };

private:
    void shapRecursive(const TTree& tree,
                       const TDoubleVec& samplesPerNode,
                       const CDataFrameCategoryEncoder& encoder,
                       const CEncodedDataFrameRowRef& encodedRow,
                       CTreeShapFeatureImportance::TDoubleVec& phi,
                       SPath splitPath,
                       std::size_t nodeIndex,
                       double parentFractionZero,
                       double parentFractionOne,
                       int parentFeatureIndex);
    void extendPath(CTreeShapFeatureImportance::SPath& path,
                    double fractionZero,
                    double fractionOne,
                    int featureIndex);
    double sumUnwoundPath(const CTreeShapFeatureImportance::SPath& path, int pathIndex) const;
    void unwindPath(CTreeShapFeatureImportance::SPath& path, int pathIndex);

private:
    TTreeVec m_Trees;
    TDoubleVecVec m_SamplesPerNode;
};
}
}

#endif // INCLUDED_ml_maths_CTreeShapFeatureImportance_h
