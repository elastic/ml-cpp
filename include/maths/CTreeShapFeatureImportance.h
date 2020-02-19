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

    //! Collects the elements of the path through decision tree that are updated together
    struct SPathElement {
        double s_FractionOnes;
        double s_FractionZeros;
        int s_FeatureIndex;
    };

    using TElementVec = std::vector<SPathElement>;
    using TElementItr = TElementVec::iterator;
    using TDoubleVecItr = TDoubleVec::iterator;

    class CSplitPath {
    public:
        CSplitPath(TElementItr fractionsIterator, TDoubleVecItr scaleIterator) {
            m_FractionsIterator = fractionsIterator;
            m_ScaleIterator = scaleIterator;
        }

        CSplitPath(const CSplitPath& parentSplitPath, int nextIndex)
            : CSplitPath(parentSplitPath.fractionsBegin() + nextIndex,
                         parentSplitPath.scaleBegin() + nextIndex) {
            std::copy(parentSplitPath.fractionsBegin(),
                      parentSplitPath.fractionsBegin() + nextIndex,
                      this->fractionsBegin());
            std::copy(parentSplitPath.scaleBegin(),
                      parentSplitPath.scaleBegin() + nextIndex, this->scaleBegin());
        }

        TElementItr& fractions() { return m_FractionsIterator; }
        const TElementItr& fractions() const { return m_FractionsIterator; }
        TDoubleVecItr& scale() { return m_ScaleIterator; }
        const TDoubleVecItr& scale() const { return m_ScaleIterator; }

        SPathElement& operator[](int index) {
            return m_FractionsIterator[index];
        }

        TElementItr& fractionsBegin() { return m_FractionsIterator; }
        const TElementItr& fractionsBegin() const {
            return m_FractionsIterator;
        }

        TDoubleVecItr& scaleBegin() { return m_ScaleIterator; }
        const TDoubleVecItr& scaleBegin() const { return m_ScaleIterator; }

        void setValues(int index, double fractionOnes, double fractionZeros, int featureIndex) {
            m_FractionsIterator[index].s_FractionOnes = fractionOnes;
            m_FractionsIterator[index].s_FractionZeros = fractionZeros;
            m_FractionsIterator[index].s_FeatureIndex = featureIndex;
        }

        void scale(int index, double value) { m_ScaleIterator[index] = value; }

        double scale(int index) const { return m_ScaleIterator[index]; }

        int featureIndex(int nextIndex) const {
            return m_FractionsIterator[nextIndex].s_FeatureIndex;
        }

        double fractionZeros(int nextIndex) const {
            return m_FractionsIterator[nextIndex].s_FractionZeros;
        }

        double fractionOnes(int nextIndex) const {
            return m_FractionsIterator[nextIndex].s_FractionOnes;
        }

        int find(int feature, int nextIndex) {
            auto featureIndexEnd{(this->fractionsBegin() + nextIndex)};
            auto it = std::find_if(this->fractionsBegin(), featureIndexEnd,
                                   [feature](const SPathElement& el) {
                                       return el.s_FeatureIndex == feature;
                                   });
            if (it != featureIndexEnd) {
                return std::distance(this->fractionsBegin(), it);
            } else {
                return -1;
            }
        }

    private:
        TElementItr m_FractionsIterator;
        TDoubleVecItr m_ScaleIterator;
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
                       int nextIndex,
                       const CSplitPath& path) const;
    //! Extend the \p path object, update the variables and factorial scaling coefficients.
    static void extendPath(CSplitPath& splitPath,
                           double fractionZero,
                           double fractionOne,
                           int featureIndex,
                           int& nextIndex);
    //! Sum the scaling coefficients for the \p scalePath without the feature defined in \p pathIndex.
    static double sumUnwoundPath(const CSplitPath& path, int pathIndex, int nextIndex);
    //! Updated the scaling coefficients in the \p path if the feature defined in \p pathIndex was seen again.
    static void unwindPath(CSplitPath& path, int pathIndex, int& nextIndex);

private:
    TTreeVec m_Trees;
    std::size_t m_NumberThreads;
    TDoubleVecVec m_SamplesPerNode;
};
}
}

#endif // INCLUDED_ml_maths_CTreeShapFeatureImportance_h
