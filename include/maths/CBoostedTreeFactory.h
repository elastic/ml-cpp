#ifndef INCLUDED_ml_maths_CBoostedTreeFactory_h
#define INCLUDED_ml_maths_CBoostedTreeFactory_h

#include <core/CContainerPrinter.h>
#include <core/CDataFrame.h>
#include <core/CPackedBitVector.h>

#include <maths/CBoostedTree.h>

#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

namespace ml {
namespace maths {

class CNode;


//! SimpleFactory for CBoostedTree object
class CBoostedTreeFactory final {
public:
    using TPackedBitVectorVec = std::vector<core::CPackedBitVector>;
    using TNodeVec = std::vector<CNode>;

public:
    void constructBoostedTree(std::size_t numberThreads, std::size_t dependentVariable, CBoostedTree::TLossFunctionUPtr loss);
    void constructBoostedTree(core::CDataFrame& frame);
    void initializeMissingFeatureMasks(const core::CDataFrame& frame);
    std::pair<TPackedBitVectorVec, TPackedBitVectorVec> crossValidationRowMasks() const;
    //! Initialize the regressors sample distribution.
    void initializeFeatureSampleDistribution(const core::CDataFrame& frame);

    //! Read overrides for hyperparameters and if necessary estimate the initial
    //! values for \f$\lambda\f$ and \f$\gamma\f$ which match the gain from an
    //! overfit tree.
    void initializeHyperparameters(core::CDataFrame& frame, CBoostedTree::TProgressCallback recordProgress);

    //! Initialize the predictions and loss function derivatives for the masked
    //! rows in \p frame.
    TNodeVec initializePredictionsAndLossDerivatives(core::CDataFrame& frame,
                                                     const core::CPackedBitVector& trainingRowMask) const;

private:
    using TDoubleVec = std::vector<double>;
    using TDoubleVecVec = std::vector<TDoubleVec>;
    using TSizeVec = std::vector<std::size_t>;
    using TSizeDoublePr = std::pair<std::size_t, double>;
    using TDoubleDoubleDoubleTr = std::tuple<double, double, double>;
    using TRowItr = core::CDataFrame::TRowItr;
    using TRowRef = core::CDataFrame::TRowRef;


private:
    CBoostedTree::CImpl treeImpl;

};


}
}


#endif // INCLUDED_ml_maths_CBoostedTreeFactory_h
