/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <boost/test/unit_test.hpp>
#include <boost/optional.hpp>

#include <list>
#include <map>
#include <memory>
#include <utility>
#include <vector>

BOOST_AUTO_TEST_SUITE(CContainerPrinterTest)

using namespace ml;
using namespace core;

BOOST_AUTO_TEST_CASE(testAll) {
    std::vector<double> vec;
    LOG_DEBUG(<< "vec = " << CContainerPrinter::print(vec));
    BOOST_TEST(CContainerPrinter::print(vec) == "[]");
    vec.push_back(1.1);
    vec.push_back(3.2);
    LOG_DEBUG(<< "vec = " << CContainerPrinter::print(vec));
    BOOST_CHECK_EQUAL(std::string("[1.1, 3.2]"), CContainerPrinter::print(vec));

    std::list<std::pair<int, int>> list;
    list.push_back(std::make_pair(1, 2));
    list.push_back(std::make_pair(2, 2));
    list.push_back(std::make_pair(3, 2));
    LOG_DEBUG(<< "list = " << CContainerPrinter::print(list));
    BOOST_CHECK_EQUAL(std::string("[(1, 2), (2, 2), (3, 2)]"),
                         CContainerPrinter::print(list));

    std::list<std::shared_ptr<double>> plist;
    plist.push_back(std::shared_ptr<double>());
    plist.push_back(std::shared_ptr<double>(new double(3.0)));
    plist.push_back(std::shared_ptr<double>(new double(1.1)));
    LOG_DEBUG(<< "plist = " << CContainerPrinter::print(plist));
    BOOST_CHECK_EQUAL(std::string("[\"null\", 3, 1.1]"), CContainerPrinter::print(plist));

    double three = 3.0;
    double fivePointOne = 5.1;
    std::map<double, double*> map;
    map.insert(std::make_pair(1.1, &three));
    map.insert(std::make_pair(3.3, &fivePointOne));
    map.insert(std::make_pair(1.0, static_cast<double*>(nullptr)));
    LOG_DEBUG(<< "map = " << CContainerPrinter::print(map));
    BOOST_CHECK_EQUAL(std::string("[(1, \"null\"), (1.1, 3), (3.3, 5.1)]"),
                         CContainerPrinter::print(map));

    std::unique_ptr<int> pints[] = {std::unique_ptr<int>(new int(2)),
                                    std::unique_ptr<int>(new int(3)),
                                    std::unique_ptr<int>(new int(2))};
    LOG_DEBUG(<< "pints = "
              << CContainerPrinter::print(std::begin(pints), std::end(pints)));
    BOOST_CHECK_EQUAL(std::string("[2, 3, 2]"),
                         CContainerPrinter::print(std::begin(pints), std::end(pints)));

    std::vector<boost::optional<double>> ovec(2, boost::optional<double>());
    LOG_DEBUG(<< "ovec = " << CContainerPrinter::print(ovec));
    BOOST_CHECK_EQUAL(std::string("[\"null\", \"null\"]"),
                         CContainerPrinter::print(ovec));

    std::vector<std::pair<std::list<std::pair<int, int>>, double>> aggregate;
    aggregate.push_back(std::make_pair(list, 1.3));
    aggregate.push_back(std::make_pair(std::list<std::pair<int, int>>(), 0.0));
    aggregate.push_back(std::make_pair(list, 5.1));
    LOG_DEBUG(<< "aggregate = " << CContainerPrinter::print(aggregate));
    BOOST_CHECK_EQUAL(std::string("[([(1, 2), (2, 2), (3, 2)], 1.3), ([], 0), ([(1, 2), (2, 2), (3, 2)], 5.1)]"),
                         CContainerPrinter::print(aggregate));
}


BOOST_AUTO_TEST_SUITE_END()
