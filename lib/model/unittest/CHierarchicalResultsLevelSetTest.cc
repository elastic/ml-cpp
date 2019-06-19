/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CHierarchicalResultsLevelSetTest.h"

#include <core/CLogger.h>
#include <core/CStoredStringPtr.h>

#include <model/CAnnotatedProbability.h>
#include <model/CHierarchicalResults.h>
#include <model/CHierarchicalResultsLevelSet.h>
#include <model/CStringStore.h>

#include <memory>

namespace {
struct STestNode {
    STestNode(const std::string& name) : s_Name(name) {}
    std::string print() const { return s_Name; }
    std::string s_Name;
};

class CConcreteHierarchicalResultsLevelSet
    : public ml::model::CHierarchicalResultsLevelSet<STestNode> {
public:
    class CFactory {
    public:
        CFactory() {}

        STestNode make(const ml::model::CHierarchicalResults::TNode& node, bool) const {
            return STestNode("\"" + *node.s_Spec.s_PartitionFieldName + " " +
                             *node.s_Spec.s_PartitionFieldValue + " " +
                             *node.s_Spec.s_PersonFieldName + " " +
                             *node.s_Spec.s_PersonFieldValue + "\"");
        }
    };

public:
    CConcreteHierarchicalResultsLevelSet(const STestNode& root)
        : ml::model::CHierarchicalResultsLevelSet<STestNode>(root) {}

    //! Visit a node.
    virtual void visit(const ml::model::CHierarchicalResults& /*results*/,
                       const TNode& /*node*/,
                       bool /*pivot*/) {}

    // make public
    using ml::model::CHierarchicalResultsLevelSet<STestNode>::elements;
};

auto makeRoot() {
    ml::model::hierarchical_results_detail::SResultSpec spec;
    ml::model::SAnnotatedProbability prob;
    return CConcreteHierarchicalResultsLevelSet::TNode{spec, prob};
}

auto makeNode(CConcreteHierarchicalResultsLevelSet::TNode& parent,
              ml::core::CStoredStringPtr partitionName,
              ml::core::CStoredStringPtr partitionValue,
              ml::core::CStoredStringPtr personName,
              ml::core::CStoredStringPtr personValue) {
    ml::model::hierarchical_results_detail::SResultSpec spec;
    spec.s_PartitionFieldName = partitionName;
    spec.s_PartitionFieldValue = partitionValue;
    spec.s_PersonFieldName = personName;
    spec.s_PersonFieldValue = personValue;
    ml::model::SAnnotatedProbability prob;
    auto node = std::make_unique<CConcreteHierarchicalResultsLevelSet::TNode>(spec, prob);
    node->s_Parent = &parent;
    parent.s_Children.push_back(node.get());
    return node;
}

auto makeNode(CConcreteHierarchicalResultsLevelSet::TNode& parent,
              ml::core::CStoredStringPtr partitionName,
              ml::core::CStoredStringPtr partitionValue) {
    return makeNode(parent, partitionName, partitionValue,
                    ml::model::CStringStore::names().getEmpty(),
                    ml::model::CStringStore::names().getEmpty());
}
}

CppUnit::Test* CHierarchicalResultsLevelSetTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CHierarchicalResultsLevelSetTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CHierarchicalResultsLevelSetTest>(
        "CHierarchicalResultsLevelSetTest::testElements",
        &CHierarchicalResultsLevelSetTest::testElements));

    return suiteOfTests;
}

void CHierarchicalResultsLevelSetTest::testElements() {

    using TNodePtr = std::unique_ptr<CConcreteHierarchicalResultsLevelSet::TNode>;

    ml::core::CStoredStringPtr pa = ml::model::CStringStore::names().get("PA");
    ml::core::CStoredStringPtr pb = ml::model::CStringStore::names().get("PB");
    ml::core::CStoredStringPtr pa1 = ml::model::CStringStore::names().get("pa1");
    ml::core::CStoredStringPtr pa2 = ml::model::CStringStore::names().get("pa2");
    ml::core::CStoredStringPtr pb1 = ml::model::CStringStore::names().get("pb1");
    ml::core::CStoredStringPtr pb2 = ml::model::CStringStore::names().get("pb2");

    CConcreteHierarchicalResultsLevelSet::TNode root{makeRoot()};
    TNodePtr partitions[]{makeNode(root, pa, pa1), makeNode(root, pa, pa2)};
    TNodePtr leaves[]{makeNode(*partitions[0], pa, pa1, pb, pb1),
                      makeNode(*partitions[0], pa, pa1, pb, pb2),
                      makeNode(*partitions[1], pa, pa2, pb, pb1),
                      makeNode(*partitions[1], pa, pa2, pb, pb2)};

    CConcreteHierarchicalResultsLevelSet levelSet(STestNode("root"));

    std::vector<STestNode*> result;

    // We should get the same level set corresponding to "pa1" (the first
    // partition added) for all partition level nodes.

    for (const auto& partition : partitions) {
        levelSet.elements(*partition, false,
                          CConcreteHierarchicalResultsLevelSet::CFactory(), result);
        LOG_DEBUG(<< "partition level = " << ml::core::CContainerPrinter::print(result));
        CPPUNIT_ASSERT_EQUAL(std::string{"[\"PA pa1  \"]"},
                             ml::core::CContainerPrinter::print(result));
    }

    // We should get the same level set corresponding to ("pa1", "pb1")
    // (the first leaf added) for all leaf level nodes.

    for (const auto& leaf : leaves) {
        levelSet.elements(*leaf, false,
                          CConcreteHierarchicalResultsLevelSet::CFactory(), result);
        LOG_DEBUG(<< "leaf level = " << ml::core::CContainerPrinter::print(result));
        CPPUNIT_ASSERT_EQUAL(std::string{"[\"PA pa1 PB pb1\"]"},
                             ml::core::CContainerPrinter::print(result));
    }
}
