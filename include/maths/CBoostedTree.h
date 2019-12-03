/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CBoostedTree_h
#define INCLUDED_ml_maths_CBoostedTree_h

#include <core/CDataFrame.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBoostedTreeHyperparameters.h>
#include <maths/CDataFrameCategoryEncoder.h>
#include <maths/CDataFrameRegressionModel.h>
#include <maths/CLinearAlgebra.h>
#include <maths/ImportExport.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace ml {
namespace core {
class CPackedBitVector;
}
namespace maths {
class CDataFrameCategoryEncoder;
class CEncodedDataFrameRowRef;

namespace boosted_tree_detail {
class MATHS_EXPORT CArgMinLossImpl {
public:
    CArgMinLossImpl(double lambda);
    virtual ~CArgMinLossImpl() = default;

    virtual std::unique_ptr<CArgMinLossImpl> clone() const = 0;
    virtual bool nextPass() = 0;
    virtual void add(double prediction, double actual, double weight = 1.0) = 0;
    virtual void merge(const CArgMinLossImpl& other) = 0;
    virtual double value() const = 0;

protected:
    double lambda() const;

private:
    double m_Lambda;
};

//! \brief Finds the value to add to a set of predictions which minimises the
//! regularized MSE w.r.t. the actual values.
class MATHS_EXPORT CArgMinMseImpl final : public CArgMinLossImpl {
public:
    CArgMinMseImpl(double lambda);
    std::unique_ptr<CArgMinLossImpl> clone() const override;
    bool nextPass() override;
    void add(double prediction, double actual, double weight = 1.0) override;
    void merge(const CArgMinLossImpl& other) override;
    double value() const override;

private:
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

private:
    TMeanAccumulator m_MeanError;
};

//! \brief Finds the value to add to a set of predicted log-odds which minimises
//! regularised cross entropy loss w.r.t. the actual categories.
class MATHS_EXPORT CArgMinLogisticImpl final : public CArgMinLossImpl {
public:
    CArgMinLogisticImpl(double lambda);
    std::unique_ptr<CArgMinLossImpl> clone() const override;
    bool nextPass() override;
    void add(double prediction, double actual, double weight = 1.0) override;
    void merge(const CArgMinLossImpl& other) override;
    double value() const override;

private:
    using TMinMaxAccumulator = CBasicStatistics::CMinMax<double>;
    using TVector = CVectorNx1<double, 2>;
    using TVectorVec = std::vector<TVector>;

private:
    std::size_t bucket(double prediction) const {
        double bucket{(prediction - m_PredictionMinMax.min()) / this->bucketWidth()};
        return std::min(static_cast<std::size_t>(bucket),
                        m_BucketCategoryCounts.size() - 1);
    }

    double bucketCentre(std::size_t bucket) const {
        return m_PredictionMinMax.min() +
               (static_cast<double>(bucket) + 0.5) * this->bucketWidth();
    }

    double bucketWidth() const {
        return m_PredictionMinMax.range() /
               static_cast<double>(m_BucketCategoryCounts.size());
    }

private:
    std::size_t m_CurrentPass = 0;
    TMinMaxAccumulator m_PredictionMinMax;
    TVector m_CategoryCounts;
    TVectorVec m_BucketCategoryCounts;
};
}

namespace boosted_tree {

//! \brief Computes the leaf value which minimizes the loss function.
class MATHS_EXPORT CArgMinLoss {
public:
    CArgMinLoss(const CArgMinLoss& other);
    CArgMinLoss(CArgMinLoss&& other) = default;

    CArgMinLoss& operator=(const CArgMinLoss& other);
    CArgMinLoss& operator=(CArgMinLoss&& other) = default;

    //! Start another pass over the predictions and actuals.
    //!
    //! \return True if we need to perform another pass to compute value().
    bool nextPass() const;

    //! Update with a point prediction and actual value.
    void add(double prediction, double actual, double weight = 1.0);

    //! Get the minimiser over the predictions and actual values added to both
    //! this and \p other.
    void merge(CArgMinLoss& other);

    //! Returns the value to add to the predictions which minimises the loss
    //! with respect to the actuals.
    //!
    //! Formally, returns \f$x^* = arg\min_x\{\sum_i{L(p_i + x, a_i)}\}\f$
    //! for predictions and actuals \f$p_i\f$ and \f$a_i\f$, respectively.
    double value() const;

private:
    using TArgMinLossImplUPtr = std::unique_ptr<boosted_tree_detail::CArgMinLossImpl>;

private:
    CArgMinLoss(const boosted_tree_detail::CArgMinLossImpl& impl);

private:
    TArgMinLossImplUPtr m_Impl;

    friend class CLoss;
};

//! \brief Defines the loss function for the regression problem.
class MATHS_EXPORT CLoss {
public:
    virtual ~CLoss() = default;
    //! Clone the loss.
    virtual std::unique_ptr<CLoss> clone() const = 0;
    //! The value of the loss function.
    virtual double value(double prediction, double actual, double weight = 1.0) const = 0;
    //! The slope of the loss function.
    virtual double gradient(double prediction, double actual, double weight = 1.0) const = 0;
    //! The curvature of the loss function.
    virtual double curvature(double prediction, double actual, double weight = 1.0) const = 0;
    //! Returns true if the loss curvature is constant.
    virtual bool isCurvatureConstant() const = 0;
    //! Get an object which computes the leaf value that minimises loss.
    virtual CArgMinLoss minimizer(double lambda) const = 0;
    //! Get the name of the loss function
    virtual const std::string& name() const = 0;

protected:
    CArgMinLoss makeMinimizer(const boosted_tree_detail::CArgMinLossImpl& impl) const;
};

//! \brief The MSE loss function.
class MATHS_EXPORT CMse final : public CLoss {
public:
    std::unique_ptr<CLoss> clone() const override;
    double value(double prediction, double actual, double weight = 1.0) const override;
    double gradient(double prediction, double actual, double weight = 1.0) const override;
    double curvature(double prediction, double actual, double weight = 1.0) const override;
    bool isCurvatureConstant() const override;
    CArgMinLoss minimizer(double lambda) const override;
    const std::string& name() const override;

public:
    static const std::string NAME;
};

//! \brief Implements loss for binomial logistic regression.
//!
//! DESCRIPTION:\n
//! This targets the cross entropy loss using the tree to predict class log-odds:
//! <pre class="fragment">
//!   \f$\displaystyle l_i(p) = -(1 - a_i) \log(1 - S(p)) - a_i \log(S(p))\f$
//! </pre>
//! where \f$a_i\f$ denotes the actual class of the i'th example, \f$p\f$ is the
//! prediction and \f$S(\cdot)\f$ denotes the logistic function.
class MATHS_EXPORT CLogistic final : public CLoss {
public:
    std::unique_ptr<CLoss> clone() const override;
    double value(double prediction, double actual, double weight = 1.0) const override;
    double gradient(double prediction, double actual, double weight = 1.0) const override;
    double curvature(double prediction, double actual, double weight = 1.0) const override;
    bool isCurvatureConstant() const override;
    CArgMinLoss minimizer(double lambda) const override;
    const std::string& name() const override;

public:
    static const std::string NAME;
};
}

class CBoostedTreeImpl;

//! \brief A node of a regression tree.
//!
//! DESCRIPTION:\n
//! This defines a tree structure on a vector of nodes (maintaining the parent
//! child relationships as indexes into the vector). It holds the (binary)
//! splitting criterion (feature and value) and the tree's prediction at each
//! leaf. The intervals are open above so the left node contains feature vectors
//! for which the feature value is _strictly_ less than the split value.
//!
//! During training row masks are maintained for each node (so the data can be
//! efficiently traversed). This supports extracting the left and right child
//! node bit masks from the node's bit mask.
class MATHS_EXPORT CBoostedTreeNode final {
public:
    using TNodeIndex = std::uint32_t;
    using TSizeSizePr = std::pair<TNodeIndex, TNodeIndex>;
    using TPackedBitVectorPackedBitVectorPr =
        std::pair<core::CPackedBitVector, core::CPackedBitVector>;
    using TNodeVec = std::vector<CBoostedTreeNode>;
    using TOptionalNodeIndex = boost::optional<TNodeIndex>;

    class MATHS_EXPORT CVisitor {
    public:
        virtual ~CVisitor() = default;
        //! Adds to last added tree.
        virtual void addNode(std::size_t splitFeature,
                             double splitValue,
                             bool assignMissingToLeft,
                             double nodeValue,
                             double gain,
                             TOptionalNodeIndex leftChild,
                             TOptionalNodeIndex rightChild) = 0;
    };

public:
    //! See core::CMemory.
    static bool dynamicSizeAlwaysZero() { return true; }

    //! Check if this is a leaf node.
    bool isLeaf() const { return m_LeftChild.is_initialized() == false; }

    //! Get the leaf index for \p row.
    TNodeIndex leafIndex(const CEncodedDataFrameRowRef& row,
                         const TNodeVec& tree,
                         TNodeIndex index = 0) const;

    //! Check if we should assign \p row to the left leaf.
    bool assignToLeft(const CEncodedDataFrameRowRef& row) const {
        double value{row[m_SplitFeature]};
        bool missing{CDataFrameUtils::isMissing(value)};
        return (missing && m_AssignMissingToLeft) ||
               (missing == false && value < m_SplitValue);
    }

    //! Get the value predicted by \p tree for the feature vector \p row.
    double value(const CEncodedDataFrameRowRef& row, const TNodeVec& tree) const {
        return tree[this->leafIndex(row, tree)].m_NodeValue;
    }

    //! Get the value of this node.
    double value() const { return m_NodeValue; }

    //! Set the node value to \p value.
    void value(double value) { m_NodeValue = value; }

    //! Get the gain of the split.
    double gain() const { return m_Gain; }

    //! Get the total curvature at the rows below this node.
    double curvature() const { return m_Curvature; }

    //! Get the index of the left child node.
    TNodeIndex leftChildIndex() const { return m_LeftChild.get(); }

    //! Get the index of the right child node.
    TNodeIndex rightChildIndex() const { return m_RightChild.get(); }

    //! Split this node and add its child nodes to \p tree.
    TSizeSizePr split(std::size_t splitFeature,
                      double splitValue,
                      bool assignMissingToLeft,
                      double gain,
                      double curvature,
                      TNodeVec& tree);

    //! Get the feature index of the split.
    std::size_t splitFeature() const { return m_SplitFeature; };

    //! Persist by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Populate the object from serialized data.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Visit this node.
    void accept(CVisitor& visitor) const;

    //! Get a human readable description of this tree.
    std::string print(const TNodeVec& tree) const;

private:
    std::ostringstream&
    doPrint(std::string pad, const TNodeVec& tree, std::ostringstream& result) const;

private:
    std::size_t m_SplitFeature = 0;
    double m_SplitValue = 0.0;
    bool m_AssignMissingToLeft = true;
    TOptionalNodeIndex m_LeftChild;
    TOptionalNodeIndex m_RightChild;
    double m_NodeValue = 0.0;
    double m_Gain = 0.0;
    double m_Curvature = 0.0;
};

//! \brief A boosted regression tree model.
//!
//! DESCRIPTION:\n
//! This is strongly based on xgboost. We deviate in two important respect: we have
//! hyperparameters which control the chance of selecting a feature in the feature
//! bag for a tree, we have automatic handling of categorical fields, we roll in a
//! hyperparameter optimisation loop based on Bayesian Optimisation seeded with a
//! random search and we use an increasing learn rate training a single forest.
//!
//! The probability of selecting a feature behave like a feature weight, allowing us
//! to:
//!   1. Incorporate an estimate of strength of relationship between a regressor and
//!      the target variable upfront,
//!   2. Use optimisation techniques suited for smooth cost functions to fine tune
//!      the regressors the tree will focus on during training.
//! All in all this gives us improved resilience to nuisance regressors and allows
//! us to perform feature selection by imposing a hard cutoff on the minimum probability
//! of a feature we will accept in the final model.
//!
//! The original xgboost paper doesn't explicitly deal with categorical data, it assumes
//! there is a well ordering on each feature and looks for binary splits subject to this
//! ordering. We use a mixed strategy which considers one-hot, target mean and frequency
//! encoding. We choose the "best" strategy based on simultaneously maximising measures
//! of relevancy and redundancy in the feature set as a whole. We use the MICe statistic
//! proposed by Reshef for this purpose. See CDataFrameCategoryEncoder for more details.
class MATHS_EXPORT CBoostedTree final : public CDataFrameRegressionModel {
public:
    using TStrVec = std::vector<std::string>;
    using TRowRef = core::CDataFrame::TRowRef;
    using TLossFunctionUPtr = std::unique_ptr<boosted_tree::CLoss>;
    using TDataFramePtr = core::CDataFrame*;
    using TNodeVec = std::vector<CBoostedTreeNode>;
    using TNodeVecVec = std::vector<TNodeVec>;

    class MATHS_EXPORT CVisitor : public CDataFrameCategoryEncoder::CVisitor,
                                  public CBoostedTreeNode::CVisitor {
    public:
        virtual ~CVisitor() = default;
        virtual void addTree() = 0;
    };

public:
    ~CBoostedTree() override;

    CBoostedTree(const CBoostedTree&) = delete;
    CBoostedTree& operator=(const CBoostedTree&) = delete;

    //! Train on the examples in the data frame supplied to the constructor.
    void train() override;

    //! Write the predictions to the data frame supplied to the constructor.
    //!
    //! \warning This can only be called after train.
    void predict() const override;

    //! Write SHAP values to the data frame supplied to the constructor.
    //!
    //! \warning This can only be called after train.
    void computeShapValues(int topShapValues) override;

    //! Get the feature weights the model has chosen.
    const TDoubleVec& featureWeights() const override;

    //! Get the column containing the dependent variable.
    std::size_t columnHoldingDependentVariable() const override;

    //! Get the column containing the model's prediction for the dependent variable.
    std::size_t columnHoldingPrediction(std::size_t numberColumns) const override;

    const TOptionalSizeVec& columnsHoldingShapValues() const override;

    //! Get the model produced by training if it has been run.
    const TNodeVecVec& trainedModel() const;

    //! The name of the object holding the best hyperaparameters in the state document.
    static const std::string& bestHyperparametersName();

    //! The name of the object holding the best regularisation hyperparameters in the
    //! state document.
    static const std::string& bestRegularizationHyperparametersName();

    //! A list of the names of the best individual hyperparameters in the state document.
    static TStrVec bestHyperparameterNames();

    //! \return Class containing best hyperparameters.
    const CBoostedTreeHyperparameters& bestHyperparameters() const;

    //! Persist by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Populate the object from serialized data.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Visit this tree trainer.
    void accept(CVisitor& visitor) const;

private:
    using TImplUPtr = std::unique_ptr<CBoostedTreeImpl>;

private:
    CBoostedTree(core::CDataFrame& frame,
                 TProgressCallback recordProgress,
                 TMemoryUsageCallback recordMemoryUsage,
                 TTrainingStateCallback recordTrainingState,
                 TImplUPtr&& impl);

private:
    TImplUPtr m_Impl;

    friend class CBoostedTreeFactory;
};
}
}

#endif // INCLUDED_ml_maths_CBoostedTree_h
