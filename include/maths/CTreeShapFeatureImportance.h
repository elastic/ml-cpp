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
        int s_FeatureIndex;
    };

    using TElementVec = std::vector<SPathElement>;
    using TElementIt = TElementVec::iterator;
    using TDoubleVecIt = TDoubleVec::iterator;

    class ElementAccessor {
    public:
        explicit ElementAccessor(TElementIt iterator) { m_Iterator = iterator; }

        inline SPathElement& operator[](int index) { return m_Iterator[index]; }

        inline TElementIt& begin() { return m_Iterator; }

        inline void setValues(int index, double fractionOnes, double fractionZeros, int featureIndex) {
            m_Iterator[index].s_FractionOnes = fractionOnes;
            m_Iterator[index].s_FractionZeros = fractionZeros;
            m_Iterator[index].s_FeatureIndex = featureIndex;
        }

        inline int featureIndex(int nextIndex) const {
            return m_Iterator[nextIndex].s_FeatureIndex;
        }

        inline double fractionZeros(int nextIndex) const {
            return m_Iterator[nextIndex].s_FractionZeros;
        }

        inline double fractionOnes(int nextIndex) const {
            return m_Iterator[nextIndex].s_FractionOnes;
        }

    private:
        TElementIt m_Iterator;
    };

private:
    //! Recursively traverses all pathes in the \p tree and updated SHAP values once it hits a leaf.
    //! Ref. Algorithm 2 in the paper by Lundberg et al.
    void shapRecursive(const TTree& tree,
                       const TDoubleVec& samplesPerNode,
                       const CDataFrameCategoryEncoder& encoder,
                       const CEncodedDataFrameRowRef& encodedRow,
                       std::size_t nodeIndex,
                       double parentFractionZero,
                       double parentFractionOne,
                       int parentFeatureIndex,
                       std::size_t offset,
                       core::CDataFrame::TRowItr& row,
                       TElementIt parentSplitPath,
                       int nextIndex,
                       TDoubleVecIt parentScalePath) const;
    //! Extend the \p path object, update the variables and factorial scaling coefficients.
    static void extendPath(ElementAccessor& path,
                           TDoubleVecIt& scalePath,
                           double fractionZero,
                           double fractionOne,
                           int featureIndex,
                           int& nextIndex);
    //! Sum the scaling coefficients for the \p path without the feature defined in \p pathIndex.
    static double sumUnwoundPath(const ElementAccessor& path,
                                 int pathIndex,
                                 int nextIndex,
                                 const TDoubleVecIt& scalePath);
    //! Updated the scaling coefficients in the \p path if the feature defined in \p pathIndex was seen again.
    static void unwindPath(ElementAccessor& path, int pathIndex, int& nextIndex, TDoubleVecIt& scalePath);

private:
    TTreeVec m_Trees;
    std::size_t m_NumberThreads;
    TDoubleVecVec m_SamplesPerNode;
};
}
}

#endif // INCLUDED_ml_maths_CTreeShapFeatureImportance_h
