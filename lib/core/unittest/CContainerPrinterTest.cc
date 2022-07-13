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

#include <boost/test/unit_test.hpp>

#include <list>
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

BOOST_AUTO_TEST_SUITE(CContainerPrinterTest)

using namespace ml;

namespace {
class CPrintable {
public:
    std::string print() const { return "printable"; }
};
}

BOOST_AUTO_TEST_CASE(testContainerPrinter) {

    // Test various containers and containers of containers.

    std::vector<double> vec;
    LOG_DEBUG(<< "vec = " << vec);
    BOOST_TEST_REQUIRE(core::CContainerPrinter::print(vec) == "[]");
    vec.push_back(1.1);
    vec.push_back(3.2);
    LOG_DEBUG(<< "vec = " << vec);
    BOOST_REQUIRE_EQUAL("[1.1, 3.2]", core::CContainerPrinter::print(vec));

    std::list<std::pair<int, int>> list{{1, 2}, {2, 2}, {3, 2}};
    LOG_DEBUG(<< "list = " << list);
    BOOST_REQUIRE_EQUAL("[(1, 2), (2, 2), (3, 2)]", core::CContainerPrinter::print(list));

    std::vector<std::tuple<double, double, double>> tuples{{1.1, 1.2, 1.3},
                                                           {2.2, 2.3, 2.4}};
    LOG_DEBUG(<< "tuples = " << tuples);
    BOOST_REQUIRE_EQUAL("[(1.1, 1.2, 1.3), (2.2, 2.3, 2.4)]",
                        core::CContainerPrinter::print(tuples));

    std::list<std::shared_ptr<double>> plist;
    plist.push_back(std::shared_ptr<double>());
    plist.push_back(std::shared_ptr<double>(new double(3.0)));
    plist.push_back(std::shared_ptr<double>(new double(1.1)));
    LOG_DEBUG(<< "plist = " << plist);
    BOOST_REQUIRE_EQUAL("[\"null\", 3, 1.1]", core::CContainerPrinter::print(plist));

    double three = 3.0;
    double fivePointOne = 5.1;
    std::map<double, double*> map;
    map.emplace(1.1, &three);
    map.emplace(3.3, &fivePointOne);
    map.emplace(1.0, static_cast<double*>(nullptr));
    LOG_DEBUG(<< "map = " << map);
    BOOST_REQUIRE_EQUAL("[(1, \"null\"), (1.1, 3), (3.3, 5.1)]",
                        core::CContainerPrinter::print(map));

    std::unique_ptr<int> pints[]{std::make_unique<int>(2), std::make_unique<int>(3),
                                 std::make_unique<int>(2)};
    LOG_DEBUG(<< "pints = " << pints);
    BOOST_REQUIRE_EQUAL("[2, 3, 2]", core::CContainerPrinter::print(pints));

    std::vector<std::optional<double>> ovec(2, std::optional<double>());
    LOG_DEBUG(<< "ovec = " << ovec);
    BOOST_REQUIRE_EQUAL(std::string("[\"null\", \"null\"]"),
                        core::CContainerPrinter::print(ovec));

    std::vector<std::pair<std::list<std::pair<int, int>>, double>> aggregate;
    aggregate.emplace_back(list, 1.3);
    aggregate.emplace_back(std::list<std::pair<int, int>>(), 0.0);
    aggregate.emplace_back(list, 5.1);
    LOG_DEBUG(<< "aggregate = " << aggregate);
    BOOST_REQUIRE_EQUAL("[([(1, 2), (2, 2), (3, 2)], 1.3), ([], 0), ([(1, 2), (2, 2), (3, 2)], 5.1)]",
                        core::CContainerPrinter::print(aggregate));

    std::vector<CPrintable> printables(5);
    LOG_DEBUG(<< printables);
    BOOST_REQUIRE_EQUAL("[printable, printable, printable, printable, printable]",
                        core::CContainerPrinter::print(printables));
}

BOOST_AUTO_TEST_CASE(testPrintContainers) {

    // Test manipulating a stream to print containers.

    std::ostringstream o;

    int array[]{1, 2, 3};
    o << core::CPrintContainers{} // activates container printing
      << "a vec = " << std::vector<double>{1.0, 2.0, 3.0}
      << ", a map = " << std::map<int, double>{{1, 1.5}, {2, 2.5}}
      << ", a pair = " << std::make_pair(3, true) << ", an array = " << array;

    LOG_DEBUG(<< o.str());
    BOOST_REQUIRE_EQUAL("a vec = [1, 2, 3], a map = [(1, 1.5), (2, 2.5)], a pair = (3, true), an array = [1, 2, 3]",
                        o.str());
}

BOOST_AUTO_TEST_SUITE_END()
