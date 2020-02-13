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

    //! Compute number of training samples from \p frame that pass every node in the \p tree.
    static TDoubleVec samplesPerNode(const TTree& tree,
                                     const core::CDataFrame& frame,
                                     const CDataFrameCategoryEncoder& encoder,
                                     std::size_t numThreads);

    //! Recursively computes inner node values as weighted average of the children (leaf) values
    //! \returns The maximum depth the the tree.
    static std::size_t updateNodeValues(TTree& tree,
                                        std::size_t nodeIndex,
                                        const TDoubleVec& samplesPerNode,
                                        std::size_t depth);

    //! Get the reference to the trees.
    TTreeVec& trees() { return m_Trees; }

private:
    using TSizeVec = std::vector<std::size_t>;

    struct SPathElement {
        double s_FractionOnes;
        double s_FractionZeros;
        double s_Scale;
        int s_FeatureIndex;
    };

    //! Manages variables for the current path through the tree as the main algorithm proceeds.
    struct SPath {
        using TElementVec = std::vector<SPathElement>;
        explicit SPath(std::size_t length)
            : s_Elements(length), s_MaxLength(length) {};

//        SPath(const SPath & other) {
//            *this = other;
//        }
//
//        SPath& operator=(const SPath& other) {
//            s_FractionOnes = other.s_FractionOnes;
//            s_FractionZeros = other.s_FractionZeros;
//            s_FeatureIndex = other.s_FeatureIndex;
//            s_Scale = other.s_Scale;
//            s_MaxLength = other.s_MaxLength;
//            s_NextIndex = other.s_NextIndex;
//
//            return *this;
//        }

        void extend(int featureIndex, double fractionZero, double fractionOne) {
            if (s_NextIndex < s_MaxLength) {
                s_Elements[s_NextIndex].s_FeatureIndex = featureIndex;
                s_Elements[s_NextIndex].s_FractionZeros = fractionZero;
                s_Elements[s_NextIndex].s_FractionOnes = fractionOne;
                if (s_NextIndex == 0) {
                    s_Elements[s_NextIndex].s_Scale = 1.0;
                } else {
                    s_Elements[s_NextIndex].s_Scale = 0.0;
                }
                ++s_NextIndex;
            }
        }

        void reduce(std::size_t pathIndex) {
            for (int i = static_cast<int>(pathIndex); i < this->depth(); ++i) {
                s_Elements[i].s_FeatureIndex = s_Elements[i + 1].s_FeatureIndex;
                s_Elements[i].s_FractionZeros = s_Elements[i + 1].s_FractionZeros;
                s_Elements[i].s_FractionOnes = s_Elements[i + 1].s_FractionOnes;
            }
            --s_NextIndex;
        }

        //! Indicator whether or not the feature \p pathIndex is decisive for the path.
        double fractionOnes(std::size_t pathIndex) const {
            return s_Elements[pathIndex].s_FractionOnes;
        }

        //! Fraction of all training data that reached the \p pathIndex in the path.
        double fractionZeros(std::size_t pathIndex) const {
            return s_Elements[pathIndex].s_FractionZeros;
        }

        int featureIndex(std::size_t pathIndex) const {
            return s_Elements[pathIndex].s_FeatureIndex;
        }

        //! Scaling coefficients (factorials), see. Equation (2) in the paper by Lundberg et al.
        double scale(std::size_t pathIndex) const { return s_Elements[pathIndex].s_Scale; }

        //! Current depth in the tree
        int depth() const { return static_cast<int>(s_NextIndex) - 1; }

        //! Get next index.
        std::size_t nextIndex() const { return s_NextIndex; }

        //! Set next index.
        void nextIndex(std::size_t nextIndex) { s_NextIndex = nextIndex; }

//        TDoubleVec s_FractionOnes;
//        TDoubleVec s_FractionZeros;
//        TIntVec s_FeatureIndex;
//        TDoubleVec s_Scale;
        TElementVec s_Elements;
        std::size_t s_NextIndex = 0;
        std::size_t s_MaxLength = 0;
    };
    using TSPathVec = std::vector<SPath>;

private:
    //! Recursively traverses all pathes in the \p tree and updated SHAP values once it hits a leaf.
    //! Ref. Algorithm 2 in the paper by Lundberg et al.
    void shapRecursive(const TTree &tree, const TDoubleVec &samplesPerNode, const CDataFrameCategoryEncoder &encoder,
                       const CEncodedDataFrameRowRef &encodedRow, SPath &splitPath, std::size_t nodeIndex,
                       double parentFractionZero, double parentFractionOne, int parentFeatureIndex, std::size_t offset,
                       core::CDataFrame::TRowItr &row, std::size_t treeDepth, TSPathVec &backupPathVec) const;
    //! Extend the \p path object, update the variables and factorial scaling coefficients.
    static void extendPath(SPath& path, double fractionZero, double fractionOne, int featureIndex);
    //! Sum the scaling coefficients for the \p path without the feature defined in \p pathIndex.
    static double sumUnwoundPath(const SPath& path, std::size_t pathIndex);
    //! Updated the scaling coefficients in the \p path if the feature defined in \p pathIndex was seen again.
    static void unwindPath(SPath& path, std::size_t pathIndex);

private:
    TTreeVec m_Trees;
    std::size_t m_NumberThreads;
    TDoubleVecVec m_SamplesPerNode;
};
}
}

#endif // INCLUDED_ml_maths_CTreeShapFeatureImportance_h
