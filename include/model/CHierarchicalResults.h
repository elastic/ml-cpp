/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CHierarchicalResults_h
#define INCLUDED_ml_model_CHierarchicalResults_h

#include <core/CSmallVector.h>
#include <core/CStoredStringPtr.h>

#include <maths/COrderings.h>

#include <model/CAnnotatedProbability.h>
#include <model/FunctionTypes.h>
#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <boost/optional.hpp>

#include <cstddef>
#include <deque>
#include <map>
#include <string>
#include <vector>

namespace CHierarchicalResultsTest {
struct testShouldWritePartition;
}

namespace ml {
namespace model {
class CAnomalyDetectorModel;
class CLimits;

namespace hierarchical_results_detail {

using TStoredStringPtrVec = std::vector<core::CStoredStringPtr>;
using TStoredStringPtrStoredStringPtrPr =
    std::pair<core::CStoredStringPtr, core::CStoredStringPtr>;
using TStoredStringPtrStoredStringPtrPrDoublePr =
    std::pair<TStoredStringPtrStoredStringPtrPr, double>;
using TStoredStringPtrStoredStringPtrPrDoublePrVec =
    std::vector<TStoredStringPtrStoredStringPtrPrDoublePr>;
using TStr1Vec = core::CSmallVector<std::string, 1>;

//! \brief The data fully describing a result node.
//!
//! DESCRIPTION:\n
//! A result is fully described by the complete set of field
//! names and values in a simple search. If one or more of
//! these unset then the node corresponds to come aggregation
//! of simple results.
//!
//! A simple search corresponds to a single clause in the search
//! config file. It corresponds to the following command line
//! syntax:
//! <pre>
//!   autodetect <function>[(X)] [by Y] [over Z]
//! </pre>
//!
//! So, examples include
//!   -# autodetect count
//!   -# autodetect count by status
//!   -# autodetect sum(bytes) over host
//!   -# autodetect rare by uri_path over clientip
//!   -# and so on.
struct MODEL_EXPORT SResultSpec {
    SResultSpec();

    //! Print of the specification for debugging.
    std::string print() const;

    //! A unique identifier for the search's detector.
    int s_Detector;
    //! True if this is a simple counting detector result.
    bool s_IsSimpleCount;
    //! True if this is a population search result.
    bool s_IsPopulation;
    //! True if the model was configured to use null values.
    bool s_UseNull;
    //! The name of the partitioning field.
    core::CStoredStringPtr s_PartitionFieldName;
    //! The value of the partitioning field.
    core::CStoredStringPtr s_PartitionFieldValue;
    //! The person field name. This is the name of field identifying
    //! a person, i.e. the by field name if there is no over field
    //! and over field name otherwise.
    core::CStoredStringPtr s_PersonFieldName;
    //! The value of the person field if applicable or an empty string
    //! otherwise
    core::CStoredStringPtr s_PersonFieldValue;
    //! The name of the field identifying the metric value if this is
    //! metric analysis and an empty string otherwise.
    core::CStoredStringPtr s_ValueFieldName;
    //! The optional function. Only leaf nodes have populated functions.
    core::CStoredStringPtr s_FunctionName;
    //! The "by" field name.
    core::CStoredStringPtr s_ByFieldName;
    //! The function identifier.
    function_t::EFunction s_Function;
    //! The list of scheduled event descriptions if any occured
    TStr1Vec s_ScheduledEventDescriptions;
};

//! \brief A node of the hierarchical results tree.
//!
//! DESCRIPTION:\n
//! A node of the hierarchical results tree. By default we build a tree
//! on top of our simple search results which allows us to obtain partial
//! aggregate results for each object of interest.
//!
//! For example, if the following searches are run:
//!   - autodetect count by host
//!   - autodetect sum(bytes) over host
//!   - autodetect rare(process) by host
//!
//! The common field of interest is "host" and we obtain aggregate
//! results for each host as well as an overall aggregate result.
//!
//! This is used to represent a node of the tree corresponding to this
//! aggregation process.
//!
//! \see buildHierarchicalResults for more details.
struct MODEL_EXPORT SNode {
    using TAttributeProbabilityVec = std::vector<SAttributeProbability>;
    using TNodeCPtr = const SNode*;
    using TNodeCPtrVec = std::vector<TNodeCPtr>;
    using TNodePtrSizeUMap = boost::unordered_map<TNodeCPtr, std::size_t>;
    using TSizeNodePtrUMap = boost::unordered_map<std::size_t, TNodeCPtr>;

    SNode();
    SNode(const SResultSpec& simpleSearch, SAnnotatedProbability& annotatedProbability);

    //! Returns the aggregate probability for the node
    double probability() const;

    //! Propagate consistent field names and values from the nodes children.
    void propagateFields();

    //! Print of the node for debugging.
    std::string print() const;

    //! Efficient swap
    void swap(SNode& other);

    //! \name Connectivity
    //@{
    //! The node's parent.
    TNodeCPtr s_Parent;
    //! The node's children.
    TNodeCPtrVec s_Children;
    //@}

    //! Data describing the common field of the simple searches below
    //! this node. (Note that for internal nodes the not equal field
    //! names are set to empty strings.)
    SResultSpec s_Spec;

    //! The aggregate annotated probability of the node.
    mutable SAnnotatedProbability s_AnnotatedProbability;

    //! The detector identifier.
    mutable int s_Detector;

    //! The aggregation style to use for this probability.
    mutable int s_AggregationStyle;

    //! The smallest aggregate probability of this node's children.
    mutable double s_SmallestChildProbability;

    //! The smallest aggregate probability of any of this node's descendants.
    mutable double s_SmallestDescendantProbability;

    //! The raw anomaly score of the node.
    mutable double s_RawAnomalyScore;

    //! The normalized anomaly score of the node.
    mutable double s_NormalizedAnomalyScore;

    //! \name Extra State for Results Output
    //@{
    //! The model which generated the result.
    const CAnomalyDetectorModel* s_Model;

    //! The start time of the bucket generating the anomaly.
    core_t::TTime s_BucketStartTime;

    //! The length of the bucket for this result.
    core_t::TTime s_BucketLength;
    //@}
};

//! Non-member node swap to work with standard algorithms
MODEL_EXPORT
void swap(SNode& node1, SNode& node2);

} // hierarchical_results_detail::

class CHierarchicalResultsVisitor;

//! \brief Represents the bucket result of running a full analysis.
//!
//! DESCRIPTION:\n
//! This wraps up the logic to build a hierarchy on top of a collection
//! of simple search results. A simple search would, for example, be
//! one clause of a model configuration file and has the command line
//! syntax:
//! <pre>
//!   [partitionfield = w] function[(x)] [by y] [over z]
//! </pre>
//!
//! An abstract visitor pattern is implemented, with the intention of
//! factoring out logic to, for example, output hierarchical results and
//! aggregate the probabilities up the tree. Both bottom up depth and
//! breadth first visiting strategies have been implemented.
//!
//! IMPLEMENTATION:\n
//! This loosely implements a builder pattern: each simple search result
//! is added and the intention is that all results are first added and
//! then the hierarchical object is built (although buildHierarchy can
//! be called repeatedly).
//!
//! Most of the state of this class is held by reference and could become
//! invalid if it is kept longer than to output a single result. This is
//! to minimize the amount of state that needs to be copied when outputting
//! results (to minimize both runtime and transient memory usage).
class MODEL_EXPORT CHierarchicalResults {
public:
    using TDoubleVec = std::vector<double>;
    using TAttributeProbabilityVec = std::vector<SAttributeProbability>;
    using TResultSpec = hierarchical_results_detail::SResultSpec;
    using TStoredStringPtr = core::CStoredStringPtr;
    using TStoredStringPtrStoredStringPtrPr =
        hierarchical_results_detail::TStoredStringPtrStoredStringPtrPr;
    using TStoredStringPtrStoredStringPtrPrDoublePr =
        hierarchical_results_detail::TStoredStringPtrStoredStringPtrPrDoublePr;
    using TStoredStringPtrStoredStringPtrPrDoublePrVec =
        hierarchical_results_detail::TStoredStringPtrStoredStringPtrPrDoublePrVec;
    using TNode = hierarchical_results_detail::SNode;
    using TNodePtrSizeUMap = hierarchical_results_detail::SNode::TNodePtrSizeUMap;
    using TSizeNodePtrUMap = hierarchical_results_detail::SNode::TSizeNodePtrUMap;
    using TNodeDeque = std::deque<TNode>;
    using TStoredStringPtrStoredStringPtrPrNodeMap =
        std::map<TStoredStringPtrStoredStringPtrPr, TNode, maths::COrderings::SLexicographicalCompare>;
    using TStoredStringPtrNodeMap =
        std::map<TStoredStringPtr, TNode, maths::COrderings::SLess>;

public:
    CHierarchicalResults();

    //! Add a dummy result for a simple count detector.
    void addSimpleCountResult(SAnnotatedProbability& annotatedProbability,
                              const CAnomalyDetectorModel* model = nullptr,
                              core_t::TTime bucketStartTime = 0);

    //! Add a simple search result.
    //!
    //! The general syntax for a simple search is
    //! <pre>
    //!   [partitionfield = w] function[(x)] [by y] [over z]
    //! </pre>
    //!
    //! Examples include:
    //!   -# count
    //!   -# rare by x
    //!   -# partitionfield = x mean(y)
    //!   -# min(x) over z
    //!   -# partitionfield = x dc(y) over z
    //!   -# partitionfield = w max(x) by y over z
    //!
    //! If a given search doesn't have a field pass the empty string.
    //!
    //! \param[in] detector An identifier of the detector generating this
    //! result.
    //! \param[in] isPopulation True if this is a population result and
    //! false otherwise.
    //! \param[in] functionName The name of the function of the model's search.
    //! \param[in] function The function of the model's search.
    //! \param[in] partitionFieldName The partition field name or empty.
    //! \param[in] partitionFieldValue The partition field value or empty.
    //! \param[in] personFieldName The over field name or empty.
    //! \param[in] personFieldValue The over field value or empty.
    //! \param[in] valueFieldName The name of the field containing the
    //! metric value.
    //! \param[out] annotatedProbability A struct containing the probability,
    //! the smallest attribute probabilities, the influencers,
    //! and any extra descriptive data
    //! \param[in] model The model which generated the result.
    //! \note Values which are passed by non-constant reference are swapped
    //! in to place.
    void addModelResult(int detector,
                        bool isPopulation,
                        const std::string& functionName,
                        function_t::EFunction function,
                        const std::string& partitionFieldName,
                        const std::string& partitionFieldValue,
                        const std::string& personFieldName,
                        const std::string& personFieldValue,
                        const std::string& valueFieldName,
                        SAnnotatedProbability& annotatedProbability,
                        const CAnomalyDetectorModel* model = nullptr,
                        core_t::TTime bucketStartTime = 0);

    //! Add the influencer called \p name.
    void addInfluencer(const std::string& name);

    //! Build a hierarchy from the current flat node list using the
    //! default aggregation rules.
    //!
    //! The aggregation rules in priority order are:
    //!   -# Only aggregate searches with the same partition field name
    //!      and value.
    //!   -# Subject to 1, aggregate searches with the same person field
    //!      name and value: this is the by field name and value if no
    //!      over field is specified otherwise it is the over field name
    //!      name and value.
    void buildHierarchy();

    //! Creates the pivot nodes for influencing field values.
    void createPivots();

    //! Get the root node of the hierarchy.
    const TNode* root() const;

    //! Get the influencer identified by \p influencerName and
    //! \p influencerValue if one exists.
    const TNode* influencer(const TStoredStringPtr& influencerName,
                            const TStoredStringPtr& influencerValue) const;

    //! Bottom up first visit the tree.
    void bottomUpBreadthFirst(CHierarchicalResultsVisitor& visitor) const;

    //! Top down first visit the tree.
    void topDownBreadthFirst(CHierarchicalResultsVisitor& visitor) const;

    //! Post-order depth first visit the tree.
    void postorderDepthFirst(CHierarchicalResultsVisitor& visitor) const;

    //! Visit all the pivot nodes bottom up first.
    void pivotsBottomUpBreadthFirst(CHierarchicalResultsVisitor& visitor) const;

    //! Visit all the pivot nodes top down first.
    void pivotsTopDownBreadthFirst(CHierarchicalResultsVisitor& visitor) const;

    //! Check if there are no results at all including the simple
    //! count result.
    bool empty() const;

    //! Get the count of leaf (search) results, i.e. excluding the
    //! simple count result.
    std::size_t resultCount() const;

    //! Sets the result to be interm
    void setInterim();

    //! Get type of result
    model_t::CResultType resultType() const;

    //! Print the results for debug.
    std::string print() const;

private:
    //! Create a new node.
    TNode& newNode();

    //! Create a new leaf node for the simple search \p simpleSearch.
    TNode& newLeaf(const TResultSpec& simpleSearch, SAnnotatedProbability& annotatedProbability);

    //! Create or retrieve a pivot node for the \p key.
    TNode& newPivot(TStoredStringPtrStoredStringPtrPr key);

    //! Create or retrieve a pivot root node for the \p key.
    TNode& newPivotRoot(const TStoredStringPtr& key);

    //! Post-order depth first visit the tree.
    void postorderDepthFirst(const TNode* node, CHierarchicalResultsVisitor& visitor) const;

private:
    //! Storage for the nodes.
    TNodeDeque m_Nodes;

    //! Storage for the pivot nodes.
    TStoredStringPtrStoredStringPtrPrNodeMap m_PivotNodes;

    //! Pivot root nodes.
    TStoredStringPtrNodeMap m_PivotRootNodes;

    //! Is the result final or interim?
    //! This field is transient - does not get persisted because interim results
    //! never get persisted.
    model_t::CResultType m_ResultType;
};

//! \brief Interface for visiting the results.
class MODEL_EXPORT CHierarchicalResultsVisitor {
public:
    using TNode = CHierarchicalResults::TNode;

public:
    virtual ~CHierarchicalResultsVisitor();

    //! Visit a node.
    virtual void visit(const CHierarchicalResults& results, const TNode& node, bool pivot) = 0;

protected:
    //! Check if this node is the root node.
    static bool isRoot(const TNode& node);

    //! Check if the node is a leaf.
    static bool isLeaf(const TNode& node);

    //! Check if the node is partition, i.e. if its children are
    //! one or more partitions.
    static bool isPartitioned(const TNode& node);

    //! Check if this is a named partition.
    static bool isPartition(const TNode& node);

    //! Check if the node is a named person.
    static bool isPerson(const TNode& node);

    //! Check if the node is an attribute of a person.
    static bool isAttribute(const TNode& node);

    //! Check if the node is simple counting result.
    static bool isSimpleCount(const TNode& node);

    //! Check if the node is a population result.
    static bool isPopulation(const TNode& node);

    //! Check if we can ever write a result for the node.
    static bool isTypeForWhichWeWriteResults(const TNode& node, bool pivot);

    //! Get the nearest ancestor of the node for which we write results.
    static const TNode* nearestAncestorForWhichWeWriteResults(const TNode& node);

    //! Check if we'll write a result for the node.
    static bool shouldWriteResult(const CLimits& limits,
                                  const CHierarchicalResults& results,
                                  const TNode& node,
                                  bool pivot);

    friend struct CHierarchicalResultsTest::testShouldWritePartition;
};
}
}

#endif // INCLUDED_ml_model_CHierarchicalResults_h
