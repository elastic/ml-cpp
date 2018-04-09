/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CHierarchicalResults.h>

#include <core/CContainerPrinter.h>
#include <core/CFunctional.h>
#include <core/CLogger.h>
#include <core/CStringUtils.h>
#include <core/RestoreMacros.h>

#include <maths/COrderings.h>

#include <model/CAnomalyDetectorModel.h>
#include <model/CDataGatherer.h>
#include <model/CLimits.h>
#include <model/CSearchKey.h>
#include <model/CStringStore.h>

#include <algorithm>
#include <limits>

namespace ml {
namespace model {

namespace hierarchical_results_detail {

namespace {

using TNodeCPtr = SNode::TNodeCPtr;

//! CHierarchicalResults tags
const std::string NODES_1_TAG("a");
const std::string NODES_2_TAG("b");
const std::string PIVOT_NAME_TAG("c");
const std::string PIVOT_VALUE_TAG("d");
const std::string PIVOT_NODES_1_TAG("e");
const std::string PIVOT_NODES_2_TAG("f");
const std::string PIVOT_ROOT_NODES_1_TAG("g");
const std::string PIVOT_ROOT_NODES_2_TAG("h");

//! SNode tags
const std::string PARENT_TAG("a");
const std::string CHILD_TAG("b");
const std::string SELF_TAG("c");
const std::string SPEC_TAG("d");
const std::string ANNOTATED_PROBABILITY_TAG("e");
const std::string DETECTOR_TAG("f");
const std::string AGGREGATION_STYLE_TAG("g");
const std::string SMALLEST_CHILD_TAG("h");
const std::string SMALLEST_DESCENDANT_TAG("i");
const std::string RAW_ANOMALY_SCORE_TAG("j");
const std::string NORMALIZED_ANOMALY_SCORE_TAG("k");
const std::string BUCKET_START_TAG("l");
const std::string BUCKET_LENGTH_TAG("m");

//! SResultSpec tags
const std::string DETECTOR_ID_TAG("a");
const std::string SIMPLE_COUNT_TAG("b");
const std::string POPULATION_TAG("c");
const std::string PARTITION_FIELD_NAME_TAG("d");
const std::string PARTITION_FIELD_VALUE_TAG("e");
const std::string PERSON_FIELD_NAME_TAG("f");
const std::string PERSON_FIELD_VALUE_TAG("g");
const std::string VALUE_FIELD_NAME_TAG("h");
const std::string USE_NULL_TAG("j");
const std::string BY_FIELD_NAME_TAG("k");
const std::string FUNCTION_NAME_TAG("i");
const std::string FUNCTION_TAG("l");

const std::string COUNT("count");
// This is intentionally NOT an empty string from the string store, but instead
// a completely separate empty string, such that its pointer will be different
// to other empty string pointers.  (In general, if you need a pointer to an
// empty string call CStringStore::getEmpty() instead of doing this.)
core::CStoredStringPtr UNSET_STRING(core::CStoredStringPtr::makeStoredString(std::string()));

//! Check if a string reference is unset.
bool unset(core::CStoredStringPtr value) {
    return value.get() == UNSET_STRING.get();
}

//! True if the node is a leaf.
bool isLeaf(const SNode& node) {
    return node.s_Children.empty();
}

//! True if the node is aggregate.
bool isAggregate(const SNode& node) {
    return node.s_Children.size() > 0;
}

//! Check if the underlying strings are equal.
bool equal(const core::CStoredStringPtr& lhs, const core::CStoredStringPtr& rhs) {
    return unset(lhs) == unset(rhs) && *lhs == *rhs;
}

//! Check if both underlying strings are equal.
bool equal(const TStoredStringPtrStoredStringPtrPr& lhs, const TStoredStringPtrStoredStringPtrPr& rhs) {
    return unset(lhs.first) == unset(rhs.first) && *lhs.first == *rhs.first && unset(lhs.second) == unset(rhs.second) &&
           *lhs.second == *rhs.second;
}

//! Orders nodes by the value of their person field.
struct SPersonValueLess {
    bool operator()(const TNodeCPtr& lhs, const TNodeCPtr& rhs) const {
        return maths::COrderings::lexicographical_compare(*lhs->s_Spec.s_PartitionFieldName,
                                                          *lhs->s_Spec.s_PartitionFieldValue,
                                                          *lhs->s_Spec.s_PersonFieldName,
                                                          *lhs->s_Spec.s_PersonFieldValue,
                                                          lhs->s_Spec.s_IsPopulation,
                                                          *rhs->s_Spec.s_PartitionFieldName,
                                                          *rhs->s_Spec.s_PartitionFieldValue,
                                                          *rhs->s_Spec.s_PersonFieldName,
                                                          *rhs->s_Spec.s_PersonFieldValue,
                                                          rhs->s_Spec.s_IsPopulation);
    }
};

//! Orders nodes by the name of their person field.
struct SPersonNameLess {
    bool operator()(const TNodeCPtr& lhs, const TNodeCPtr& rhs) const {
        return maths::COrderings::lexicographical_compare(*lhs->s_Spec.s_PartitionFieldName,
                                                          *lhs->s_Spec.s_PartitionFieldValue,
                                                          *lhs->s_Spec.s_PersonFieldName,
                                                          *rhs->s_Spec.s_PartitionFieldName,
                                                          *rhs->s_Spec.s_PartitionFieldValue,
                                                          *rhs->s_Spec.s_PersonFieldName);
    }
};

//! Orders nodes by the value of their partition field.
struct SPartitionValueLess {
    bool operator()(const TNodeCPtr& lhs, const TNodeCPtr& rhs) const {
        return maths::COrderings::lexicographical_compare(*lhs->s_Spec.s_PartitionFieldName,
                                                          *lhs->s_Spec.s_PartitionFieldValue,
                                                          *rhs->s_Spec.s_PartitionFieldName,
                                                          *rhs->s_Spec.s_PartitionFieldValue);
    }
};

//! Orders nodes by the name of their partition field.
struct SPartitionNameLess {
    bool operator()(const TNodeCPtr& lhs, const TNodeCPtr& rhs) const {
        return *lhs->s_Spec.s_PartitionFieldName < *rhs->s_Spec.s_PartitionFieldName;
    }
};

//! Return the node pointer.
SNode* address(SNode* ptr) {
    return ptr;
}

//! Get the address of a node value.
SNode* address(SNode& value) {
    return &value;
}

//! Aggregate the nodes in a layer.
template<typename LESS, typename ITR, typename FACTORY>
void aggregateLayer(ITR beginLayer, ITR endLayer, CHierarchicalResults& results, FACTORY newNode, std::vector<SNode*>& newLayer) {
    using TNodePtrVec = std::vector<SNode*>;
    using TNodeCPtrNodePtrVecMap = std::map<TNodeCPtr, TNodePtrVec, LESS>;

    newLayer.clear();

    TNodeCPtrNodePtrVecMap aggregation;
    for (ITR i = beginLayer; i != endLayer; ++i) {
        aggregation[address(*i)].push_back(address(*i));
    }

    newLayer.reserve(aggregation.size());

    for (const auto& children : aggregation) {
        LOG_TRACE("aggregating = " << core::CContainerPrinter::print(children.second));
        if (children.second.size() > 1) {
            SNode& aggregate = (results.*newNode)();
            bool population = false;
            aggregate.s_Children.reserve(children.second.size());
            for (const auto& child : children.second) {
                aggregate.s_Children.push_back(child);
                child->s_Parent = &aggregate;
                population |= child->s_Spec.s_IsPopulation;
            }
            aggregate.s_Spec.s_IsPopulation = population;
            aggregate.propagateFields();
            newLayer.push_back(&aggregate);
        } else {
            newLayer.push_back(children.second[0]);
        }
    }
}

//! \brief Propagates influences to the appropriate point in the
//! hierarchical results.
//!
//! DESCRIPTION:\n
//! This must be applied in a bottom up breadth first traversal
//! of a collection of hierarchical results. It propagates each
//! influencing field value to the highest node in the tree such
//! that it is either the person or partition field of that node.
class CCommonInfluencePropagator : public CHierarchicalResultsVisitor {
public:
    virtual void visit(const CHierarchicalResults& /*results*/, const TNode& node, bool /*pivot*/) {
        if (this->isLeaf(node)) {
            std::sort(node.s_AnnotatedProbability.s_Influences.begin(),
                      node.s_AnnotatedProbability.s_Influences.end(),
                      maths::COrderings::SFirstLess());
        } else {
            for (const auto& child : node.s_Children) {
                for (const auto& influence : child->s_AnnotatedProbability.s_Influences) {
                    if (equal({node.s_Spec.s_PartitionFieldName, node.s_Spec.s_PartitionFieldValue}, influence.first) ||
                        equal({node.s_Spec.s_PersonFieldName, node.s_Spec.s_PersonFieldValue}, influence.first)) {
                        auto i = std::lower_bound(node.s_AnnotatedProbability.s_Influences.begin(),
                                                  node.s_AnnotatedProbability.s_Influences.end(),
                                                  influence.first,
                                                  maths::COrderings::SFirstLess());
                        if (i == node.s_AnnotatedProbability.s_Influences.end()) {
                            node.s_AnnotatedProbability.s_Influences.push_back(influence);
                        } else if (!equal(i->first, influence.first)) {
                            node.s_AnnotatedProbability.s_Influences.insert(i, influence);
                        }
                    }
                }
            }
        }
    }
};

} // unnamed::

SResultSpec::SResultSpec()
    : s_Detector(0),
      s_IsSimpleCount(false),
      s_IsPopulation(false),
      s_UseNull(false),
      s_PartitionFieldName(UNSET_STRING),
      s_PartitionFieldValue(UNSET_STRING),
      s_PersonFieldName(UNSET_STRING),
      s_PersonFieldValue(UNSET_STRING),
      s_ValueFieldName(UNSET_STRING),
      s_FunctionName(UNSET_STRING),
      s_ByFieldName(UNSET_STRING),
      s_Function(function_t::E_IndividualCount) {
}

std::string SResultSpec::print() const {
    return '\'' + core::CStringUtils::typeToStringPretty(s_IsSimpleCount) + '/' + core::CStringUtils::typeToStringPretty(s_IsPopulation) +
           '/' + *s_FunctionName + '/' + *s_PartitionFieldName + '/' + *s_PartitionFieldValue + '/' + *s_PersonFieldName + '/' +
           *s_PersonFieldValue + '/' + *s_ValueFieldName + '\'';
}

void SResultSpec::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(DETECTOR_ID_TAG, s_Detector);
    inserter.insertValue(SIMPLE_COUNT_TAG, s_IsSimpleCount);
    inserter.insertValue(POPULATION_TAG, s_IsPopulation);
    inserter.insertValue(USE_NULL_TAG, s_UseNull);
    core::CPersistUtils::persist(FUNCTION_TAG, s_Function, inserter);
    if (!unset(s_PartitionFieldName)) {
        inserter.insertValue(PARTITION_FIELD_NAME_TAG, *s_PartitionFieldName);
    }
    if (!unset(s_PartitionFieldValue)) {
        inserter.insertValue(PARTITION_FIELD_VALUE_TAG, *s_PartitionFieldValue);
    }
    if (!unset(s_PersonFieldName)) {
        inserter.insertValue(PERSON_FIELD_NAME_TAG, *s_PersonFieldName);
    }
    if (!unset(s_PersonFieldValue)) {
        inserter.insertValue(PERSON_FIELD_VALUE_TAG, *s_PersonFieldValue);
    }
    if (!unset(s_ValueFieldName)) {
        inserter.insertValue(VALUE_FIELD_NAME_TAG, *s_ValueFieldName);
    }
    if (!unset(s_FunctionName)) {
        inserter.insertValue(FUNCTION_NAME_TAG, *s_FunctionName);
    }
    if (!unset(s_ByFieldName)) {
        inserter.insertValue(BY_FIELD_NAME_TAG, *s_ByFieldName);
    }
}

bool SResultSpec::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE_BUILT_IN(DETECTOR_ID_TAG, s_Detector)
        RESTORE_BUILT_IN(SIMPLE_COUNT_TAG, s_IsSimpleCount)
        RESTORE_BUILT_IN(POPULATION_TAG, s_IsPopulation)
        RESTORE_BUILT_IN(USE_NULL_TAG, s_UseNull)
        RESTORE_SETUP_TEARDOWN(
            FUNCTION_TAG, int f = 0, core::CPersistUtils::restore(FUNCTION_TAG, f, traverser), s_Function = function_t::EFunction(f))
        RESTORE_NO_ERROR(PARTITION_FIELD_NAME_TAG, s_PartitionFieldName = CStringStore::names().get(traverser.value()))
        RESTORE_NO_ERROR(PARTITION_FIELD_VALUE_TAG, s_PartitionFieldValue = CStringStore::names().get(traverser.value()))
        RESTORE_NO_ERROR(PERSON_FIELD_NAME_TAG, s_PersonFieldName = CStringStore::names().get(traverser.value()))
        RESTORE_NO_ERROR(PERSON_FIELD_VALUE_TAG, s_PersonFieldValue = CStringStore::names().get(traverser.value()))
        RESTORE_NO_ERROR(VALUE_FIELD_NAME_TAG, s_ValueFieldName = CStringStore::names().get(traverser.value()))
        RESTORE_NO_ERROR(FUNCTION_NAME_TAG, s_FunctionName = CStringStore::names().get(traverser.value()))
        RESTORE_NO_ERROR(BY_FIELD_NAME_TAG, s_ByFieldName = CStringStore::names().get(traverser.value()))
    } while (traverser.next());
    return true;
}

SNode::SNode()
    : s_Parent(nullptr),
      s_AnnotatedProbability(1.0),
      s_Detector(-3),
      s_AggregationStyle(-1),
      s_SmallestChildProbability(1.0),
      s_SmallestDescendantProbability(1.0),
      s_RawAnomalyScore(0.0),
      s_NormalizedAnomalyScore(0.0),
      s_Model(nullptr),
      s_BucketStartTime(0),
      s_BucketLength(0) {
}

SNode::SNode(const SResultSpec& simpleSearch, SAnnotatedProbability& annotatedProbability)
    : s_Parent(nullptr),
      s_Spec(simpleSearch),
      s_Detector(simpleSearch.s_Detector),
      s_AggregationStyle(-1),
      s_SmallestChildProbability(annotatedProbability.s_Probability),
      s_SmallestDescendantProbability(1.0),
      s_RawAnomalyScore(0.0),
      s_NormalizedAnomalyScore(0.0),
      s_Model(nullptr),
      s_BucketStartTime(0),
      s_BucketLength(0) {
    s_AnnotatedProbability.swap(annotatedProbability);
}

double SNode::probability() const {
    return s_AnnotatedProbability.s_Probability;
}

void SNode::propagateFields() {
    if (s_Children.empty()) {
        return;
    }

    s_Spec.s_PartitionFieldName = s_Children[0]->s_Spec.s_PartitionFieldName;
    s_Spec.s_PartitionFieldValue = s_Children[0]->s_Spec.s_PartitionFieldValue;
    s_Spec.s_PersonFieldName = s_Children[0]->s_Spec.s_PersonFieldName;
    s_Spec.s_PersonFieldValue = s_Children[0]->s_Spec.s_PersonFieldValue;
    s_BucketStartTime = s_Children[0]->s_BucketStartTime;
    for (std::size_t i = 1u; i < s_Children.size(); ++i) {
        if (!unset(s_Spec.s_PartitionFieldName) && !equal(s_Spec.s_PartitionFieldName, s_Children[i]->s_Spec.s_PartitionFieldName)) {
            s_Spec.s_PartitionFieldName = UNSET_STRING;
            s_Spec.s_PartitionFieldValue = UNSET_STRING;
            s_Spec.s_PersonFieldName = UNSET_STRING;
            s_Spec.s_PersonFieldValue = UNSET_STRING;
        }
        if (!unset(s_Spec.s_PartitionFieldValue) && !equal(s_Spec.s_PartitionFieldValue, s_Children[i]->s_Spec.s_PartitionFieldValue)) {
            s_Spec.s_PartitionFieldValue = UNSET_STRING;
            s_Spec.s_PersonFieldName = UNSET_STRING;
            s_Spec.s_PersonFieldValue = UNSET_STRING;
        }
        if (!unset(s_Spec.s_PersonFieldName) && !equal(s_Spec.s_PersonFieldName, s_Children[i]->s_Spec.s_PersonFieldName)) {
            s_Spec.s_PersonFieldName = UNSET_STRING;
        }
        if (!unset(s_Spec.s_PersonFieldValue) && !equal(s_Spec.s_PersonFieldValue, s_Children[i]->s_Spec.s_PersonFieldValue)) {
            s_Spec.s_PersonFieldValue = UNSET_STRING;
        }
    }
}

std::string SNode::print() const {
    return s_Spec.print() + ": " + core::CStringUtils::typeToStringPretty(this->probability()) + ", " +
           core::CStringUtils::typeToStringPretty(s_RawAnomalyScore) +
           (s_AnnotatedProbability.s_Influences.empty() ? "" : ", " + core::CContainerPrinter::print(s_AnnotatedProbability.s_Influences));
}

void SNode::swap(SNode& other) {
    std::swap(s_Parent, other.s_Parent);
    s_Children.swap(other.s_Children);
    std::swap(s_Spec, other.s_Spec);
    s_AnnotatedProbability.swap(other.s_AnnotatedProbability);
    std::swap(s_Detector, other.s_Detector);
    std::swap(s_AggregationStyle, other.s_AggregationStyle);
    std::swap(s_SmallestChildProbability, other.s_SmallestChildProbability);
    std::swap(s_SmallestDescendantProbability, other.s_SmallestDescendantProbability);
    std::swap(s_RawAnomalyScore, other.s_RawAnomalyScore);
    std::swap(s_NormalizedAnomalyScore, other.s_NormalizedAnomalyScore);
    std::swap(s_Model, other.s_Model);
    std::swap(s_BucketStartTime, other.s_BucketStartTime);
    std::swap(s_BucketLength, other.s_BucketLength);
}

void SNode::acceptPersistInserter1(core::CStatePersistInserter& inserter, TNodePtrSizeUMap& nodePointers) const {
    std::size_t index = nodePointers.emplace(this, nodePointers.size()).first->second;
    inserter.insertValue(SELF_TAG, index);
    core::CPersistUtils::persist(SPEC_TAG, s_Spec, inserter);
    core::CPersistUtils::persist(ANNOTATED_PROBABILITY_TAG, s_AnnotatedProbability, inserter);
    inserter.insertValue(DETECTOR_TAG, s_Detector);
    inserter.insertValue(AGGREGATION_STYLE_TAG, s_AggregationStyle);
    inserter.insertValue(SMALLEST_CHILD_TAG, s_SmallestChildProbability);
    inserter.insertValue(SMALLEST_DESCENDANT_TAG, s_SmallestDescendantProbability);
    inserter.insertValue(RAW_ANOMALY_SCORE_TAG, s_RawAnomalyScore);
    inserter.insertValue(NORMALIZED_ANOMALY_SCORE_TAG, s_NormalizedAnomalyScore);
    inserter.insertValue(BUCKET_START_TAG, s_BucketStartTime);
    inserter.insertValue(BUCKET_LENGTH_TAG, s_BucketLength);
}

void SNode::acceptPersistInserter2(core::CStatePersistInserter& inserter, const TNodePtrSizeUMap& nodePointers) const {
    if (s_Parent != nullptr) {
        auto found = nodePointers.find(s_Parent);
        if (found == nodePointers.end()) {
            LOG_ERROR("Parent not in persistence hierarchy!");
            return;
        }
        core::CPersistUtils::persist(PARENT_TAG, found->second, inserter);
    }

    for (const auto& child : s_Children) {
        auto found = nodePointers.find(child);
        if (found == nodePointers.end()) {
            LOG_ERROR("Child not in persistence hierarchy!");
            return;
        }
        core::CPersistUtils::persist(CHILD_TAG, found->second, inserter);
    }
}

bool SNode::acceptRestoreTraverser1(core::CStateRestoreTraverser& traverser, TSizeNodePtrUMap& nodePointers) {
    do {
        const std::string& name = traverser.name();
        RESTORE_SETUP_TEARDOWN(SELF_TAG,
                               std::size_t index = 0,
                               core::CStringUtils::stringToType(traverser.value(), index),
                               nodePointers.insert(std::make_pair(index, this)))
        RESTORE(SPEC_TAG, core::CPersistUtils::restore(SPEC_TAG, s_Spec, traverser))
        RESTORE(ANNOTATED_PROBABILITY_TAG, core::CPersistUtils::restore(ANNOTATED_PROBABILITY_TAG, s_AnnotatedProbability, traverser))
        RESTORE_BUILT_IN(DETECTOR_TAG, s_Detector);
        RESTORE_BUILT_IN(AGGREGATION_STYLE_TAG, s_AggregationStyle);
        RESTORE_BUILT_IN(SMALLEST_CHILD_TAG, s_SmallestChildProbability)
        RESTORE_BUILT_IN(SMALLEST_DESCENDANT_TAG, s_SmallestDescendantProbability)
        RESTORE_BUILT_IN(RAW_ANOMALY_SCORE_TAG, s_RawAnomalyScore)
        RESTORE_BUILT_IN(NORMALIZED_ANOMALY_SCORE_TAG, s_NormalizedAnomalyScore)
        RESTORE_BUILT_IN(BUCKET_START_TAG, s_BucketStartTime)
        RESTORE_BUILT_IN(BUCKET_LENGTH_TAG, s_BucketLength)
    } while (traverser.next());
    return true;
}

bool SNode::acceptRestoreTraverser2(core::CStateRestoreTraverser& traverser, const TSizeNodePtrUMap& nodePointers) {
    do {
        const std::string& name = traverser.name();
        std::size_t index = 0;
        if (name == PARENT_TAG) {
            if (!core::CPersistUtils::restore(PARENT_TAG, index, traverser)) {
                LOG_ERROR("Restore error for " << traverser.name() << " / " << traverser.value());
                return false;
            }
            auto found = nodePointers.find(index);
            if (found == nodePointers.end()) {
                LOG_ERROR("Parent not in persistence hierarchy!");
                return false;
            }
            s_Parent = found->second;
        } else if (name == CHILD_TAG) {
            if (!core::CPersistUtils::restore(CHILD_TAG, index, traverser)) {
                LOG_ERROR("Restore error for " << traverser.name() << " / " << traverser.value());
                return false;
            }
            auto found = nodePointers.find(index);
            if (found == nodePointers.end()) {
                LOG_ERROR("Parent not in persistence hierarchy!");
                return false;
            }
            s_Children.push_back(found->second);
        }
    } while (traverser.next());
    return true;
}

void swap(SNode& node1, SNode& node2) {
    node1.swap(node2);
}

} // hierarchical_results_detail::

using namespace hierarchical_results_detail;

CHierarchicalResults::CHierarchicalResults() : m_ResultType(model_t::CResultType::E_Final) {
}

void CHierarchicalResults::addSimpleCountResult(SAnnotatedProbability& annotatedProbability,
                                                const CAnomalyDetectorModel* model,
                                                core_t::TTime bucketStartTime) {
    TResultSpec search;
    search.s_IsSimpleCount = true;
    search.s_IsPopulation = false;
    search.s_FunctionName = CStringStore::names().get(COUNT);
    search.s_Function = function_t::E_IndividualCount;
    search.s_PersonFieldName = CStringStore::names().get(COUNT);
    search.s_PersonFieldValue = CStringStore::names().get(COUNT);
    search.s_UseNull = (model ? model->dataGatherer().useNull() : false);
    search.s_ByFieldName = CStringStore::names().get(COUNT);
    search.s_ScheduledEventDescriptions = model->scheduledEventDescriptions(bucketStartTime);

    // For simple counts we set all the anomaly scores to 0
    // and all the probabilities to 100%.
    TNode& leaf = this->newLeaf(search, annotatedProbability);
    leaf.s_Model = model;
    leaf.s_BucketStartTime = bucketStartTime;
    leaf.s_BucketLength = (model ? model->bucketLength() : 0);
}

void CHierarchicalResults::addModelResult(int detector,
                                          bool isPopulation,
                                          const std::string& functionName,
                                          function_t::EFunction function,
                                          const std::string& partitionFieldName,
                                          const std::string& partitionFieldValue,
                                          const std::string& personFieldName,
                                          const std::string& personFieldValue,
                                          const std::string& valueFieldName,
                                          SAnnotatedProbability& annotatedProbability,
                                          const CAnomalyDetectorModel* model,
                                          core_t::TTime bucketStartTime) {
    TResultSpec spec;
    spec.s_Detector = detector;
    spec.s_IsSimpleCount = false;
    spec.s_FunctionName = CStringStore::names().get(functionName);
    spec.s_Function = function;
    spec.s_IsPopulation = isPopulation;
    spec.s_UseNull = (model ? model->dataGatherer().useNull() : false);
    spec.s_PartitionFieldName = CStringStore::names().get(partitionFieldName);
    spec.s_PartitionFieldValue = CStringStore::names().get(partitionFieldValue);
    spec.s_PersonFieldName = CStringStore::names().get(personFieldName);
    spec.s_PersonFieldValue = CStringStore::names().get(personFieldValue);
    spec.s_ValueFieldName = CStringStore::names().get(valueFieldName);
    spec.s_ByFieldName = (model ? CStringStore::names().get(model->dataGatherer().searchKey().byFieldName()) : UNSET_STRING);
    TNode& leaf = this->newLeaf(spec, annotatedProbability);
    leaf.s_Model = model;
    leaf.s_BucketStartTime = bucketStartTime;
    leaf.s_BucketLength = (model ? model->bucketLength() : 0);
}

void CHierarchicalResults::addInfluencer(const std::string& name) {
    this->newPivotRoot(CStringStore::influencers().get(name));
}

void CHierarchicalResults::buildHierarchy() {
    using TNodePtrVec = std::vector<SNode*>;

    m_Nodes.erase(std::remove_if(m_Nodes.begin(), m_Nodes.end(), isAggregate), m_Nodes.end());

    // To make life easier for downstream code, bring a simple count node
    // to the front of the deque (if there is one).
    auto simpleCountItr = m_Nodes.end();
    for (auto i = m_Nodes.begin(); i != m_Nodes.end(); ++i) {
        i->s_Parent = nullptr;
        if (i->s_Spec.s_IsSimpleCount) {
            simpleCountItr = i;
        }
    }
    if (simpleCountItr != m_Nodes.end()) {
        while (simpleCountItr != m_Nodes.begin()) {
            auto next = simpleCountItr;
            std::iter_swap(--simpleCountItr, next);
        }
    }

    TNodePtrVec layer;
    TNodePtrVec newLayer;

    LOG_TRACE("Distinct values of the person field");
    {
        aggregateLayer<SPersonValueLess>(m_Nodes.begin(), m_Nodes.end(), *this, &CHierarchicalResults::newNode, layer);
        LOG_TRACE("layer = " << core::CContainerPrinter::print(layer));
    }

    LOG_TRACE("Distinct person field names");
    {
        newLayer.reserve(layer.size());
        aggregateLayer<SPersonNameLess>(layer.begin(), layer.end(), *this, &CHierarchicalResults::newNode, newLayer);
        newLayer.swap(layer);
        LOG_TRACE("layer = " << core::CContainerPrinter::print(layer));
    }

    LOG_TRACE("Distinct partition field values");
    {
        newLayer.reserve(layer.size());
        aggregateLayer<SPartitionValueLess>(layer.begin(), layer.end(), *this, &CHierarchicalResults::newNode, newLayer);
        newLayer.swap(layer);
        LOG_TRACE("layer = " << core::CContainerPrinter::print(layer));
    }

    LOG_TRACE("Distinct partition field names");
    {
        newLayer.reserve(layer.size());
        aggregateLayer<SPartitionNameLess>(layer.begin(), layer.end(), *this, &CHierarchicalResults::newNode, newLayer);
        newLayer.swap(layer);
        LOG_TRACE("layer = " << core::CContainerPrinter::print(layer));
    }

    if (layer.size() > 1) {
        TNode& root = this->newNode();
        bool population = false;
        for (std::size_t i = 0u; i < layer.size(); ++i) {
            root.s_Children.push_back(layer[i]);
            layer[i]->s_Parent = &root;
            population |= layer[i]->s_Spec.s_IsPopulation;
        }
        root.s_Spec.s_IsPopulation = population;
        LOG_TRACE("root = " << root.print());
    }

    LOG_TRACE("Propagating influences");

    CCommonInfluencePropagator influencePropagator;
    this->bottomUpBreadthFirst(influencePropagator);
}

void CHierarchicalResults::createPivots() {
    LOG_TRACE("Creating pivots");

    for (const auto& node : m_Nodes) {
        const auto& parentInfluences = node.s_Parent->s_AnnotatedProbability.s_Influences;
        for (const auto& influence : node.s_AnnotatedProbability.s_Influences) {
            if (node.s_Parent &&
                std::binary_search(parentInfluences.begin(), parentInfluences.end(), influence, maths::COrderings::SFirstLess())) {
                continue;
            }
            this->newPivot(influence.first).s_Children.push_back(&node);
        }
    }

    for (auto& pivot : m_PivotNodes) {
        TNode& root = this->newPivotRoot(pivot.second.s_Spec.s_PersonFieldName);
        root.s_Children.push_back(&pivot.second);
        pivot.second.s_Parent = &root;
    }
}

const CHierarchicalResults::TNode* CHierarchicalResults::root() const {
    if (m_Nodes.empty()) {
        return nullptr;
    }
    if (m_Nodes.size() == 1) {
        return &m_Nodes.front();
    }
    const TNode& result = m_Nodes.back();
    if (isLeaf(result)) {
        return nullptr;
    }
    return &result;
}

const CHierarchicalResults::TNode* CHierarchicalResults::influencer(const TStoredStringPtr& influencerName,
                                                                    const TStoredStringPtr& influencerValue) const {
    auto i = m_PivotNodes.find({influencerName, influencerValue});
    return i != m_PivotNodes.end() ? &i->second : nullptr;
}

void CHierarchicalResults::bottomUpBreadthFirst(CHierarchicalResultsVisitor& visitor) const {
    for (const auto& node : m_Nodes) {
        visitor.visit(*this, node, /*pivot =*/false);
    }
}

void CHierarchicalResults::topDownBreadthFirst(CHierarchicalResultsVisitor& visitor) const {
    for (auto i = m_Nodes.rbegin(); i != m_Nodes.rend(); ++i) {
        visitor.visit(*this, *i, /*pivot =*/false);
    }
}

void CHierarchicalResults::postorderDepthFirst(CHierarchicalResultsVisitor& visitor) const {
    if (const TNode* root = this->root()) {
        this->postorderDepthFirst(root, visitor);
    }
}

void CHierarchicalResults::pivotsBottomUpBreadthFirst(CHierarchicalResultsVisitor& visitor) const {
    for (const auto& pivot : m_PivotNodes) {
        visitor.visit(*this, pivot.second, /*pivot =*/true);
    }
    for (const auto& root : m_PivotRootNodes) {
        visitor.visit(*this, root.second, /*pivot =*/true);
    }
}

void CHierarchicalResults::pivotsTopDownBreadthFirst(CHierarchicalResultsVisitor& visitor) const {
    for (const auto& root : m_PivotRootNodes) {
        visitor.visit(*this, root.second, /*pivot =*/true);
    }
    for (const auto& pivot : m_PivotNodes) {
        visitor.visit(*this, pivot.second, /*pivot =*/true);
    }
}

bool CHierarchicalResults::empty() const {
    return m_Nodes.empty();
}

std::size_t CHierarchicalResults::resultCount() const {
    std::size_t result = 0u;
    for (const auto& node : m_Nodes) {
        if (isLeaf(node) && !node.s_Spec.s_IsSimpleCount) {
            ++result;
        }
    }
    return result;
}

void CHierarchicalResults::setInterim() {
    m_ResultType.set(model_t::CResultType::E_Interim);
}

model_t::CResultType CHierarchicalResults::resultType() const {
    return m_ResultType;
}

void CHierarchicalResults::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    using TStoredStringPtrNodeMapCItr = TStoredStringPtrNodeMap::const_iterator;
    using TStoredStringPtrNodeMapCItrVec = std::vector<TStoredStringPtrNodeMapCItr>;
    using TStoredStringPtrStoredStringPtrPrNodeMapCItr = TStoredStringPtrStoredStringPtrPrNodeMap::const_iterator;
    using TStoredStringPtrStoredStringPtrPrNodeMapCItrVec = std::vector<TStoredStringPtrStoredStringPtrPrNodeMapCItr>;

    TNodePtrSizeUMap nodePointers;

    for (const auto& node : m_Nodes) {
        inserter.insertLevel(NODES_1_TAG, boost::bind(&SNode::acceptPersistInserter1, boost::cref(node), _1, boost::ref(nodePointers)));
    }

    // Sort the keys by *value* order to ensure consistent persist state.
    TStoredStringPtrStoredStringPtrPrNodeMapCItrVec pivotIterators;
    pivotIterators.reserve(m_PivotNodes.size());
    for (auto i = m_PivotNodes.begin(); i != m_PivotNodes.end(); ++i) {
        pivotIterators.push_back(i);
    }
    std::sort(pivotIterators.begin(), pivotIterators.end(), core::CFunctional::SDereference<maths::COrderings::SFirstLess>());
    for (auto i : pivotIterators) {
        core::CPersistUtils::persist(PIVOT_NAME_TAG, *i->first.first, inserter);
        core::CPersistUtils::persist(PIVOT_VALUE_TAG, *i->first.second, inserter);
        inserter.insertLevel(PIVOT_NODES_1_TAG,
                             boost::bind(&SNode::acceptPersistInserter1, boost::cref(i->second), _1, boost::ref(nodePointers)));
    }

    // Sort the keys by *value* order to ensure consistent persist state.
    TStoredStringPtrNodeMapCItrVec pivotRootIterators;
    pivotRootIterators.reserve(m_PivotRootNodes.size());
    for (auto i = m_PivotRootNodes.begin(); i != m_PivotRootNodes.end(); ++i) {
        pivotRootIterators.push_back(i);
    }
    std::sort(pivotRootIterators.begin(), pivotRootIterators.end(), core::CFunctional::SDereference<maths::COrderings::SFirstLess>());
    for (auto i : pivotRootIterators) {
        core::CPersistUtils::persist(PIVOT_NAME_TAG, *i->first, inserter);
        inserter.insertLevel(PIVOT_ROOT_NODES_1_TAG,
                             boost::bind(&SNode::acceptPersistInserter1, boost::cref(i->second), _1, boost::ref(nodePointers)));
    }

    for (const auto& node : m_Nodes) {
        inserter.insertLevel(NODES_2_TAG, boost::bind(&SNode::acceptPersistInserter2, boost::cref(node), _1, boost::cref(nodePointers)));
    }

    for (auto i : pivotIterators) {
        core::CPersistUtils::persist(PIVOT_NAME_TAG, *i->first.first, inserter);
        core::CPersistUtils::persist(PIVOT_VALUE_TAG, *i->first.second, inserter);
        inserter.insertLevel(PIVOT_NODES_2_TAG,
                             boost::bind(&SNode::acceptPersistInserter2, boost::cref(i->second), _1, boost::cref(nodePointers)));
    }

    for (auto i : pivotRootIterators) {
        core::CPersistUtils::persist(PIVOT_NAME_TAG, *i->first, inserter);
        inserter.insertLevel(PIVOT_ROOT_NODES_2_TAG,
                             boost::bind(&SNode::acceptPersistInserter2, boost::cref(i->second), _1, boost::cref(nodePointers)));
    }
}

bool CHierarchicalResults::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    TSizeNodePtrUMap nodePointers;
    core::CStoredStringPtr influencerName;
    core::CStoredStringPtr influencerValue;
    std::size_t nodesFullyRestored = 0;

    do {
        const std::string& name = traverser.name();
        RESTORE_SETUP_TEARDOWN(NODES_1_TAG,
                               m_Nodes.push_back(SNode()),
                               traverser.traverseSubLevel(
                                   boost::bind(&SNode::acceptRestoreTraverser1, boost::ref(m_Nodes.back()), _1, boost::ref(nodePointers))),
                               /**/)
        if (name == NODES_2_TAG) {
            if (nodesFullyRestored > m_Nodes.size()) {
                LOG_ERROR("Invalid restore index for node: " << nodesFullyRestored);
            }
            if (traverser.traverseSubLevel(boost::bind(
                    &SNode::acceptRestoreTraverser2, boost::ref(m_Nodes[nodesFullyRestored]), _1, boost::cref(nodePointers))) == false) {
                LOG_ERROR("Failed to restore node");
                return false;
            }
            ++nodesFullyRestored;
            continue;
        }
        RESTORE_NO_ERROR(PIVOT_NAME_TAG, influencerName = CStringStore::influencers().get(traverser.value()))
        RESTORE_NO_ERROR(PIVOT_VALUE_TAG, influencerValue = CStringStore::influencers().get(traverser.value()))
        if (name == PIVOT_NODES_1_TAG) {
            if (!influencerName || !influencerValue) {
                LOG_ERROR("Invalid influencers for node");
                return false;
            }
            SNode& node = m_PivotNodes[TStoredStringPtrStoredStringPtrPr(influencerName, influencerValue)];
            if (traverser.traverseSubLevel(boost::bind(&SNode::acceptRestoreTraverser1, boost::ref(node), _1, boost::ref(nodePointers))) ==
                false) {
                LOG_ERROR("Failed to restore pivot node");
                return false;
            }
            influencerName = core::CStoredStringPtr();
            influencerValue = core::CStoredStringPtr();
            continue;
        } else if (name == PIVOT_NODES_2_TAG) {
            if (!influencerName || !influencerValue) {
                LOG_ERROR("Invalid influencers for node");
                return false;
            }
            SNode& node = m_PivotNodes[TStoredStringPtrStoredStringPtrPr(influencerName, influencerValue)];
            if (traverser.traverseSubLevel(boost::bind(&SNode::acceptRestoreTraverser2, boost::ref(node), _1, boost::cref(nodePointers))) ==
                false) {
                LOG_ERROR("Failed to restore pivot node");
                return false;
            }
            influencerName = core::CStoredStringPtr();
            influencerValue = core::CStoredStringPtr();
            continue;
        }
        if (name == PIVOT_ROOT_NODES_1_TAG) {
            if (!influencerName) {
                LOG_ERROR("Invalid influencer for node");
                return false;
            }
            SNode& node = m_PivotRootNodes[influencerName];
            if (traverser.traverseSubLevel(boost::bind(&SNode::acceptRestoreTraverser1, boost::ref(node), _1, boost::ref(nodePointers))) ==
                false) {
                LOG_ERROR("Failed to restore pivot node");
                return false;
            }
            influencerName = core::CStoredStringPtr();
            continue;
        }
        if (name == PIVOT_ROOT_NODES_2_TAG) {
            if (!influencerName) {
                LOG_ERROR("Invalid influencer for node");
                return false;
            }
            SNode& node = m_PivotRootNodes[influencerName];
            if (traverser.traverseSubLevel(boost::bind(&SNode::acceptRestoreTraverser2, boost::ref(node), _1, boost::cref(nodePointers))) ==
                false) {
                LOG_ERROR("Failed to restore pivot node");
                return false;
            }
            influencerName = core::CStoredStringPtr();
            continue;
        }
    } while (traverser.next());
    return true;
}

std::string CHierarchicalResults::print() const {
    std::ostringstream ss;
    for (const auto& node : m_Nodes) {
        ss << "\t" << node.print() << core_t::LINE_ENDING;
    }
    return ss.str();
}

CHierarchicalResults::TNode& CHierarchicalResults::newNode() {
    m_Nodes.push_back(TNode());
    return m_Nodes.back();
}

CHierarchicalResults::TNode& CHierarchicalResults::newLeaf(const TResultSpec& simpleSearch, SAnnotatedProbability& annotatedProbability) {
    m_Nodes.emplace_back(simpleSearch, annotatedProbability);
    return m_Nodes.back();
}

CHierarchicalResults::TNode& CHierarchicalResults::newPivot(TStoredStringPtrStoredStringPtrPr key) {
    TNode& result = m_PivotNodes[key];
    result.s_Spec.s_PersonFieldName = key.first;
    result.s_Spec.s_PersonFieldValue = key.second;
    return result;
}

CHierarchicalResults::TNode& CHierarchicalResults::newPivotRoot(const TStoredStringPtr& key) {
    TNode& result = m_PivotRootNodes[key];
    result.s_Spec.s_PersonFieldName = key;
    result.s_Spec.s_PersonFieldValue = UNSET_STRING;
    return result;
}

void CHierarchicalResults::postorderDepthFirst(const TNode* node, CHierarchicalResultsVisitor& visitor) const {
    for (const auto& child : node->s_Children) {
        this->postorderDepthFirst(child, visitor);
    }
    visitor.visit(*this, *node, /*pivot =*/false);
}

CHierarchicalResultsVisitor::~CHierarchicalResultsVisitor() {
}

bool CHierarchicalResultsVisitor::isRoot(const TNode& node) {
    return !node.s_Parent;
}

bool CHierarchicalResultsVisitor::isLeaf(const TNode& node) {
    return node.s_Children.empty();
}

bool CHierarchicalResultsVisitor::isPartitioned(const TNode& node) {
    return !((*node.s_Spec.s_PartitionFieldName).empty()) && unset(node.s_Spec.s_PartitionFieldValue);
}

bool CHierarchicalResultsVisitor::isPartition(const TNode& node) {
    return !((*node.s_Spec.s_PartitionFieldName).empty()) && !unset(node.s_Spec.s_PartitionFieldValue) &&
           (CHierarchicalResultsVisitor::isRoot(node) || unset(node.s_Parent->s_Spec.s_PartitionFieldValue));
}

bool CHierarchicalResultsVisitor::isPerson(const TNode& node) {
    if ((*node.s_Spec.s_PersonFieldName).empty() || isPartitioned(node)) {
        return false;
    }
    if (!isPopulation(node)) {
        return unset(node.s_Spec.s_PersonFieldValue) || CHierarchicalResultsVisitor::isRoot(node) ||
               unset(node.s_Parent->s_Spec.s_PersonFieldName);
    }
    return !unset(node.s_Spec.s_PersonFieldValue) &&
           (CHierarchicalResultsVisitor::isRoot(node) || (unset(node.s_Parent->s_Spec.s_PersonFieldValue)));
}

bool CHierarchicalResultsVisitor::isAttribute(const TNode& node) {
    if (!isLeaf(node) || isPartition(node) || isRoot(node)) {
        return false;
    }
    if (isPerson(*node.s_Parent)) {
        return true;
    }
    return !isPopulation(node);
}

bool CHierarchicalResultsVisitor::isSimpleCount(const TNode& node) {
    return node.s_Spec.s_IsSimpleCount;
}

bool CHierarchicalResultsVisitor::isPopulation(const TNode& node) {
    return node.s_Spec.s_IsPopulation;
}

const CHierarchicalResultsVisitor::TNode* CHierarchicalResultsVisitor::nearestAncestorForWhichWeWriteResults(const TNode& node) {
    const TNode* result = &node;
    for (result = result->s_Parent; result && !isTypeForWhichWeWriteResults(*result, false); result = result->s_Parent) {}
    return result;
}

bool CHierarchicalResultsVisitor::isTypeForWhichWeWriteResults(const TNode& node, bool pivot) {
    return pivot || isLeaf(node) || isRoot(node) || isPartition(node);
}

bool CHierarchicalResultsVisitor::shouldWriteResult(const CLimits& limits,
                                                    const CHierarchicalResults& results,
                                                    const TNode& node,
                                                    bool pivot) {
    double p = std::min(node.probability(), node.s_SmallestDescendantProbability);

    // This test ensures that we output results at aggregated levels in the
    // hierarchy if we've output results at lower levels.  Without this
    // condition the UI can be very confusing, as it's not necessarily possible
    // to find anything when searching upwards from lowest level anomalies to
    // the aggregated levels above.
    if (p < limits.unusualProbabilityThreshold() && isTypeForWhichWeWriteResults(node, pivot)) {
        return true;
    }

    // This condition is historical - in reality we always write bucket level
    // results regardless of this condition.
    // (However, if this is removed in the future another test must be added to
    // prevent the root node being allowed to permeate to the last test in this
    // method.)
    if (CHierarchicalResultsVisitor::isRoot(node)) {
        return false;
    }

    // This test ensures that if we write a result at a level of the hierarchy
    // below the bucket level we'll also write at least one result at each
    // of the levels beneath this.  Results written as a result of this test
    // will potentially have high probabilities, but should either have a low
    // probability themselves or be in a branch of the results tree which contains
    // low probability results. Again, the purpose is to avoid inconsistencies
    // in the UI where a user drills down from an aggregated result and sees
    // nothing.
    static const double OUTPUT_TOLERANCE(1.2);
    const TNode* ancestor = nearestAncestorForWhichWeWriteResults(node);
    if (ancestor && p <= OUTPUT_TOLERANCE * ancestor->s_SmallestDescendantProbability &&
        shouldWriteResult(limits, results, *ancestor, pivot)) {
        return true;
    }

    // This test ensures that if we are going to write an influencer result
    // we will write at least one of the results it influences. As with the
    // the test above nodes written as a result of this test must either have
    // a low probability themselves or be in branch of the results tree which
    // contains low probability results.
    for (const auto& influence : node.s_AnnotatedProbability.s_Influences) {
        const TNode* influencer = results.influencer(influence.first.first, influence.first.second);
        if (influencer && p <= OUTPUT_TOLERANCE * influencer->s_SmallestDescendantProbability &&
            shouldWriteResult(limits, results, *influencer, /*pivot = */ true)) {
            return true;
        }
    }

    return false;
}
}
}
