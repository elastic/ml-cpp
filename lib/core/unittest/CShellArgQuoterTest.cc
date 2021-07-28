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
#include <core/CShellArgQuoter.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CShellArgQuoterTest)

BOOST_AUTO_TEST_CASE(testQuote, *boost::unit_test::disabled()) {
    LOG_DEBUG(<< "\n"
                 "echo "
              << ml::core::CShellArgQuoter::quote("")
              << "\n"
                 "echo "
              << ml::core::CShellArgQuoter::quote("hello")
              << "\n"
                 "echo "
              << ml::core::CShellArgQuoter::quote("\"hello\" there")
              << "\n"
                 "echo "
              << ml::core::CShellArgQuoter::quote("'hello' there")
              << "\n"
                 "echo "
              << ml::core::CShellArgQuoter::quote("hello! there")
              << "\n"
                 "echo "
              << ml::core::CShellArgQuoter::quote("don't want this to fail!")
              << "\n"
                 "echo "
              << ml::core::CShellArgQuoter::quote("don't want this expanded: $HOME")
              << "\n"
                 "echo "
              << ml::core::CShellArgQuoter::quote("don't want this expanded: %windir%")
              << "\n"
                 "echo "
              << ml::core::CShellArgQuoter::quote("don't want this expanded: \"$HOME\"")
              << "\n"
                 "echo "
              << ml::core::CShellArgQuoter::quote("don't want this expanded: \"%windir%\"")
              << "\n"
                 "echo "
              << ml::core::CShellArgQuoter::quote("don't want this expanded: '$HOME'")
              << "\n"
                 "echo "
              << ml::core::CShellArgQuoter::quote("don't want this expanded: '%windir%'")
              << "\n"
                 "echo "
              << ml::core::CShellArgQuoter::quote("top ^ hat!"));

    // Paste the output of the above into a command prompt and check what
    // happens...

    // TODO: could run these using system() and assert on the output
    // (but it's not high priority to do this as we don't use this
    // functionality currently)
}

BOOST_AUTO_TEST_SUITE_END()
