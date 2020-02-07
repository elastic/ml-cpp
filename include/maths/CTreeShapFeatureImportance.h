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

//! \brief Computes SHAP (SHapley Additive exPlanation) values for feature importance estimation for gradient boosting
//! trees.
//!
//! DESCRIPTION:\n
//! SHAP values is a unique consistent and locally accurate attribution value. This mean that the sum of the SHAP
//! feature importance values approximates the model prediction up to a constant bias. This implementation follows the
//! algorithm "Consistent Individualized Feature Attribution for Tree Ensembles" by  Lundberg, Erion, and Lee.
//! The algorithm has the complexity O(TLD^2) where T is the number of trees, L is the maximum number of leaves in the
//! tree, and D is the maximum depth of a tree in the ensemble.
class MATHS_EXPORT CTreeShapFeatureImportance {
public:
    using TTree = std::vector<CBoostedTreeNode>;
    using TTreeVec = std::vector<TTree>;
    using TIntVec = std::vector<int>;
    using TDoubleVec = std::vector<double>;
    using TDoubleVecVec = std::vector<TDoubleVec>;

public:
    explicit CTreeShapFeatureImportance(TTreeVec trees, std::size_t threads = 1);

    //! Compute SHAP values for the data in \p frame using the specified \p encoder.
    //! The results are written directly back into the \p frame, the index of the first result column is controller
    //! by \p offset.
    void shap(core::CDataFrame& frame, const CDataFrameCategoryEncoder& encoder, std::size_t offset);

    //! Return the array with number of training samples affected by every node of the \p tree.
    static CTreeShapFeatureImportance::TDoubleVec numberSamples(const TTree& tree);

    //! Recursively computes inner node values as weighted average of the children (leaf) values
    //! \returns The maximum depth the the tree.
    static std::size_t updateNodeValues(TTree& tree,
                                        std::size_t nodeIndex,
                                        const TDoubleVec& numberSamples,
                                        std::size_t depth);

    //! Get the reference to the trees.
    TTreeVec& trees() { return m_Trees; }

private:
    using TSizeVec = std::vector<std::size_t>;

    //! Manages variables for the current path through the tree as the main algorithm proceeds.
    struct SPath {
        explicit SPath(std::size_t length)
            : s_FractionOnes(length), s_FractionZeros(length),
              s_FeatureIndex(length, -1), s_Scale(length), s_MaxLength(length) {}

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

        void reduce(std::size_t pathIndex) {
            for (int i = static_cast<int>(pathIndex); i < this->depth(); ++i) {
                s_FeatureIndex[i] = s_FeatureIndex[i + 1];
                s_FractionZeros[i] = s_FractionZeros[i + 1];
                s_FractionOnes[i] = s_FractionOnes[i + 1];
            }
            --s_NextIndex;
        }

        //! Indicator whether or not the feature \p pathIndex is decicive for the path.
        double fractionOnes(std::size_t pathIndex) const {
            return s_FractionOnes[pathIndex];
        }

        //! Fraction of all training data that reached the \pathIndex in the path.
        double fractionZeros(std::size_t pathIndex) const {
            return s_FractionZeros[pathIndex];
        }

        int featureIndex(std::size_t pathIndex) const {
            return s_FeatureIndex[pathIndex];
        }

        //! Scaling coefficients (factorials), see. Equation (2) in the paper by Lundberg et al.
        double scale(std::size_t pathIndex) const { return s_Scale[pathIndex]; }

        //! Current depth in the tree
        int depth() const { return static_cast<int>(s_NextIndex) - 1; }

        //! Get next index.
        std::size_t nextIndex() const { return s_NextIndex; }

        //! Set next index.
        void nextIndex(std::size_t nextIndex) { s_NextIndex = nextIndex; }

        TDoubleVec s_FractionOnes;
        TDoubleVec s_FractionZeros;
        TIntVec s_FeatureIndex;
        TDoubleVec s_Scale;
        std::size_t s_NextIndex = 0;
        std::size_t s_MaxLength = 0;
    };

private:
    //! Recursively traverses all pathes in the \p tree and updated SHAP values once it hits a leaf.
    //! Ref. Algorithm 2 in the paper by Lundberg et al.
    void shapRecursive(const TTree& tree,
                       const TDoubleVec& numberSamples,
                       const CDataFrameCategoryEncoder& encoder,
                       const CEncodedDataFrameRowRef& encodedRow,
                       SPath& splitPath,
                       std::size_t nodeIndex,
                       double parentFractionZero,
                       double parentFractionOne,
                       int parentFeatureIndex,
                       std::size_t offset,
                       core::CDataFrame::TRowItr& row) const;
    //! Extend the \p path object, update the variables and factorial scaling coefficients.
    static void extendPath(SPath& path, double fractionZero, double fractionOne, int featureIndex);
    //! Sum the scaling coefficients for the \p path without the feature defined in \p pathIndex.
    static double sumUnwoundPath(const SPath& path, std::size_t pathIndex);
    //! Updated the scaling coefficients in the \p path if the feature defined in \p pathIndex was seen again.
    static void unwindPath(SPath& path, std::size_t pathIndex);

private:
    TTreeVec m_Trees;
    std::size_t m_NumberThreads;
    TDoubleVecVec m_NumberSamples;
};
}
}

#endif // INCLUDED_ml_maths_CTreeShapFeatureImportance_h
