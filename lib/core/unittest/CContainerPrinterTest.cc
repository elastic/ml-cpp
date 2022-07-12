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

#include <core/CContainerPrinter.h>
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
using namespace core;

BOOST_AUTO_TEST_CASE(testAll) {
    std::vector<double> vec;
    LOG_DEBUG(<< "vec = " << CContainerPrinter::print(vec));
    BOOST_TEST_REQUIRE(CContainerPrinter::print(vec) == "[]");
    vec.push_back(1.1);
    vec.push_back(3.2);
    LOG_DEBUG(<< "vec = " << CContainerPrinter::print(vec));
    BOOST_REQUIRE_EQUAL(std::string("[1.1, 3.2]"), CContainerPrinter::print(vec));

    std::list<std::pair<int, int>> list{{1, 2}, {2, 2}, {3, 2}};
    LOG_DEBUG(<< "list = " << CContainerPrinter::print(list));
    BOOST_REQUIRE_EQUAL(std::string("[(1, 2), (2, 2), (3, 2)]"),
                        CContainerPrinter::print(list));

    std::vector<std::tuple<double, double, double>> tuples{{1.1, 1.2, 1.3},
                                                           {2.2, 2.3, 2.4}};
    LOG_DEBUG(<< "tuples = " << core::CContainerPrinter::print(tuples));
    BOOST_REQUIRE_EQUAL(std::string("[(1.1, 1.2, 1.3), (2.2, 2.3, 2.4)]"),
                        core::CContainerPrinter::print(tuples));

    std::list<std::shared_ptr<double>> plist;
    plist.push_back(std::shared_ptr<double>());
    plist.push_back(std::shared_ptr<double>(new double(3.0)));
    plist.push_back(std::shared_ptr<double>(new double(1.1)));
    LOG_DEBUG(<< "plist = " << CContainerPrinter::print(plist));
    BOOST_REQUIRE_EQUAL(std::string("[\"null\", 3, 1.1]"), CContainerPrinter::print(plist));

    double three = 3.0;
    double fivePointOne = 5.1;
    std::map<double, double*> map;
    map.emplace(1.1, &three);
    map.emplace(3.3, &fivePointOne);
    map.emplace(1.0, static_cast<double*>(nullptr));
    LOG_DEBUG(<< "map = " << CContainerPrinter::print(map));
    BOOST_REQUIRE_EQUAL(std::string("[(1, \"null\"), (1.1, 3), (3.3, 5.1)]"),
                        CContainerPrinter::print(map));

    std::unique_ptr<int> pints[]{std::make_unique<int>(2), std::make_unique<int>(3),
                                 std::make_unique<int>(2)};
    LOG_DEBUG(<< "pints = "
              << CContainerPrinter::print(std::begin(pints), std::end(pints)));
    BOOST_REQUIRE_EQUAL(std::string("[2, 3, 2]"),
                        CContainerPrinter::print(std::begin(pints), std::end(pints)));

    std::vector<std::optional<double>> ovec(2, std::optional<double>());
    LOG_DEBUG(<< "ovec = " << CContainerPrinter::print(ovec));
    BOOST_REQUIRE_EQUAL(std::string("[\"null\", \"null\"]"),
                        CContainerPrinter::print(ovec));

    std::vector<std::pair<std::list<std::pair<int, int>>, double>> aggregate;
    aggregate.emplace_back(list, 1.3);
    aggregate.emplace_back(std::list<std::pair<int, int>>(), 0.0);
    aggregate.emplace_back(list, 5.1);
    LOG_DEBUG(<< "aggregate = " << CContainerPrinter::print(aggregate));
    BOOST_REQUIRE_EQUAL(std::string("[([(1, 2), (2, 2), (3, 2)], 1.3), ([], 0), ([(1, 2), (2, 2), (3, 2)], 5.1)]"),
                        CContainerPrinter::print(aggregate));
}

BOOST_AUTO_TEST_SUITE_END()
