/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <core/CLogger.h>
#include <core/CMemoryDef.h>

#include <model/CAnnotatedProbability.h>
#include <model/CHierarchicalResults.h>
#include <model/CHierarchicalResultsLevelSet.h>

#include <boost/test/unit_test.hpp>

#include <memory>

BOOST_AUTO_TEST_SUITE(CHierarchicalResultsLevelSetTest)

namespace {
struct STestNode {
    STestNode(const std::string& name) : s_Name(name) {}
    std::string print() const { return s_Name; }
    std::string s_Name;

    std::size_t memoryUsage() const { return sizeof(s_Name); }
};

class CConcreteHierarchicalResultsLevelSet
    : public ml::model::CHierarchicalResultsLevelSet<STestNode> {
public:
    class CFactory {
    public:
        CFactory() {}

        STestNode make(const ml::model::CHierarchicalResults::TNode& node, bool) const {
            return STestNode("\"" + node.s_Spec.s_PartitionFieldName.value_or("") +
                             " " + node.s_Spec.s_PartitionFieldValue.value_or("") +
                             " " + node.s_Spec.s_PersonFieldName.value_or("") + " " +
                             node.s_Spec.s_PersonFieldValue.value_or("") + "\"");
        }
    };

public:
    CConcreteHierarchicalResultsLevelSet(const STestNode& root)
        : ml::model::CHierarchicalResultsLevelSet<STestNode>(root) {}

    //! Visit a node.
    void visit(const ml::model::CHierarchicalResults& /*results*/,
               const TNode& /*node*/,
               bool /*pivot*/) override {}

    // make public
    using ml::model::CHierarchicalResultsLevelSet<STestNode>::elements;
};

auto makeRoot() {
    ml::model::hierarchical_results_detail::SResultSpec spec;
    ml::model::SAnnotatedProbability prob;
    return CConcreteHierarchicalResultsLevelSet::TNode{spec, prob};
}

auto makeNode(CConcreteHierarchicalResultsLevelSet::TNode& parent,
              const std::string& partitionName,
              const std::string& partitionValue,
              const std::string& personName,
              const std::string& personValue) {
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
              const std::string& partitionName,
              const std::string& partitionValue) {
    return makeNode(parent, partitionName, partitionValue, "", "");
}
}

BOOST_AUTO_TEST_CASE(testElements) {

    using TNodePtr = std::unique_ptr<CConcreteHierarchicalResultsLevelSet::TNode>;

    std::string pa("PA");
    std::string pb("PB");
    std::string pa1("pa1");
    std::string pa2("pa2");
    std::string pb1("pb1");
    std::string pb2("pb2");

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
        BOOST_REQUIRE_EQUAL(std::string{"[\"PA pa1  \"]"},
                            ml::core::CContainerPrinter::print(result));
    }

    // We should get the same level set corresponding to ("pa1", "pb1")
    // (the first leaf added) for all leaf level nodes.

    for (const auto& leaf : leaves) {
        levelSet.elements(*leaf, false,
                          CConcreteHierarchicalResultsLevelSet::CFactory(), result);
        LOG_DEBUG(<< "leaf level = " << ml::core::CContainerPrinter::print(result));
        BOOST_REQUIRE_EQUAL(std::string{"[\"PA pa1 PB pb1\"]"},
                            ml::core::CContainerPrinter::print(result));
    }
}

BOOST_AUTO_TEST_CASE(testMemoryUsage) {
    CConcreteHierarchicalResultsLevelSet levelSet(STestNode("root"));
    std::size_t memoryUsage = levelSet.memoryUsage();
    BOOST_REQUIRE(memoryUsage > 0);

    auto addAndCheckMemoryUsage = [&memoryUsage, &levelSet](auto& container,
                                                            const std::string& name) {
        container.emplace_back(ml::core::CCompressedDictionary<1>::CWord(),
                               STestNode(name));
        std::size_t newMemoryUsage = levelSet.memoryUsage();
        BOOST_REQUIRE(newMemoryUsage > memoryUsage);
        memoryUsage = newMemoryUsage;
    };

    addAndCheckMemoryUsage(levelSet.m_InfluencerBucketSet, "influencer bucket 1");
    addAndCheckMemoryUsage(levelSet.m_InfluencerSet, "influencer 1");
    addAndCheckMemoryUsage(levelSet.m_PartitionSet, "partition 1");
    addAndCheckMemoryUsage(levelSet.m_PersonSet, "person 1");
    addAndCheckMemoryUsage(levelSet.m_LeafSet, "leaf 1");

    auto debugMemoryUsage = std::make_shared<ml::core::CMemoryUsage>();
    levelSet.debugMemoryUsage(debugMemoryUsage);
    BOOST_REQUIRE(debugMemoryUsage->usage() == memoryUsage);
}

BOOST_AUTO_TEST_SUITE_END()
