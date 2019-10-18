/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CStopWatch.h>
#include <core/CWordDictionary.h>

#include <api/CTokenListDataTyper.h>
#include <api/CTokenListReverseSearchCreator.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CTokenListDataTyperTest)

namespace {

using TTokenListDataTyperKeepsFields =
    ml::api::CTokenListDataTyper<true,  // Warping
                                 true,  // Underscores
                                 true,  // Dots
                                 true,  // Dashes
                                 true,  // Ignore leading digit
                                 true,  // Ignore hex
                                 true,  // Ignore date words
                                 false, // Ignore field names
                                 2,     // Min dictionary word length
                                 ml::core::CWordDictionary::TWeightVerbs5Other2>;

const TTokenListDataTyperKeepsFields::TTokenListReverseSearchCreatorIntfCPtr NO_REVERSE_SEARCH_CREATOR;
}


void CTokenListDataTyperTest::setUp() {
    // Enable trace level logging for these unit tests
    ml::core::CLogger::instance().setLoggingLevel(ml::core::CLogger::E_Trace);
}

void CTokenListDataTyperTest::tearDown() {
    // Revert to debug level logging for any subsequent unit tests
    ml::core::CLogger::instance().setLoggingLevel(ml::core::CLogger::E_Debug);
}

BOOST_AUTO_TEST_CASE(testHexData) {
    TTokenListDataTyperKeepsFields typer(NO_REVERSE_SEARCH_CREATOR, 0.7, "whatever");

    BOOST_CHECK_EQUAL(1, typer.computeType(false, "[0x0000000800000000 ", 500));
    BOOST_CHECK_EQUAL(1, typer.computeType(false, "0x0000000800000000", 500));
    BOOST_CHECK_EQUAL(1, typer.computeType(false, " 0x0000000800000000,", 500));
    BOOST_CHECK_EQUAL(1, typer.computeType(false, "0x0000000800000000)", 500));
    BOOST_CHECK_EQUAL(1, typer.computeType(false, " 0x0000000800000000,", 500));
}

BOOST_AUTO_TEST_CASE(testRmdsData) {
    TTokenListDataTyperKeepsFields typer(NO_REVERSE_SEARCH_CREATOR, 0.7, "whatever");

    BOOST_CHECK_EQUAL(1, typer.computeType(false, "<ml13-4608.1.p2ps: Info: > Source ML_SERVICE2 on 13122:867 has shut down.",
                                              500));
    BOOST_CHECK_EQUAL(1, typer.computeType(false, "<ml13-4602.1.p2ps: Info: > Source MONEYBROKER on 13112:736 has shut down.",
                                              500));
    BOOST_CHECK_EQUAL(1, typer.computeType(false, "<ml13-4606.1.p2ps: Info: > Source CUBE_LIQUID on 13188:2010 has shut down.",
                                              500));
    BOOST_CHECK_EQUAL(1, typer.computeType(false, "<ml13-4608.1.p2ps: Info: > Source ML SERVICE2 on 13122:867 has shut down.",
                                              500));
    BOOST_CHECK_EQUAL(2, typer.computeType(false, "<ml13-4602.1.p2ps: Info: > Source MONEYBROKER on 13112:736 has started.",
                                              500));
    BOOST_CHECK_EQUAL(2, typer.computeType(false, "<ml13-4608.1.p2ps: Info: > Source ML_SERVICE2 on 13122:867 has started.",
                                              500));
    BOOST_CHECK_EQUAL(3, typer.computeType(false, "<ml00-4201.1.p2ps: Info: > Service CUBE_CHIX, id of 132, has started.",
                                              500));
    BOOST_CHECK_EQUAL(3, typer.computeType(false, "<ml00-4601.1.p2ps: Info: > Service CUBE_IDEM, id of 232, has started.",
                                              500));
    BOOST_CHECK_EQUAL(3, typer.computeType(false, "<ml00-4601.1.p2ps: Info: > Service CUBE_IDEM, id of 232, has started.",
                                              500));
    BOOST_CHECK_EQUAL(4, typer.computeType(false, "<ml00-4201.1.p2ps: Info: > Service CUBE_CHIX has shut down.",
                                              500));
}

BOOST_AUTO_TEST_CASE(testProxyData) {
    TTokenListDataTyperKeepsFields typer(NO_REVERSE_SEARCH_CREATOR, 0.7, "whatever");

    BOOST_CHECK_EQUAL(1, typer.computeType(false,
                                              " [1094662464] INFO  transaction <3c26701d3140-kn8n1c8f5d2o> - Transaction TID: "
                                              "z9hG4bKy6aEy6aEy6aEaUgi!UmU-Ma.9-6bf50ea0192.168.251.8SUBSCRIBE deleted",
                                              500));
    BOOST_CHECK_EQUAL(1, typer.computeType(false,
                                              " [1091504448] INFO  transaction <3c26701ad775-1cref2zy3w9e> - Transaction TID: "
                                              "z9hG4bK_UQA_UQA_UQAsO0i!OG!yYK.25-5bee09e0192.168.251.8SUBSCRIBE deleted",
                                              500));
    BOOST_CHECK_EQUAL(2, typer.computeType(false,
                                              " [1094662464] INFO  transactionuser <6508700927200972648@10.10.18.82> - ---------------- "
                                              "DESTROYING RegistrationServer ---------------",
                                              500));
    BOOST_CHECK_EQUAL(3, typer.computeType(false, " [1111529792] INFO  proxy <45409105041220090733@192.168.251.123> - +++++++++++++++ CREATING ProxyCore ++++++++++++++++",
                                              500));
    BOOST_CHECK_EQUAL(4, typer.computeType(false, " [1091504448] INFO  transactionuser <3c26709ab9f0-iih26eh8pxxa> - +++++++++++++++ CREATING PresenceAgent ++++++++++++++++",
                                              500));
    BOOST_CHECK_EQUAL(5, typer.computeType(false,
                                              " [1111529792] INFO  session <45409105041220090733@192.168.251.123> - ----------------- PROXY "
                                              "Session DESTROYED --------------------",
                                              500));
    BOOST_CHECK_EQUAL(5, typer.computeType(false,
                                              " [1094662464] INFO  session <ch6z1bho8xeprb3z4ty604iktl6c@dave.proxy.uk> - ----------------- "
                                              "PROXY Session DESTROYED --------------------",
                                              500));
}

BOOST_AUTO_TEST_CASE(testFxData) {
    TTokenListDataTyperKeepsFields typer(NO_REVERSE_SEARCH_CREATOR, 0.7, "whatever");

    BOOST_CHECK_EQUAL(1, typer.computeType(false,
                                              "<L_MSG MN=\"ml12220\" PID=\"ml010_managed4\" TID=\"asyncDelivery41\" DT=\"\" PT=\"ERROR\" AP=\"wts\" DN=\"\" "
                                              "SN=\"\" SR=\"co.elastic.session.ejb.FxCoverSessionBean\">javax.ejb.FinderException - findFxCover([]): "
                                              "null</L_MSG>",
                                              500));
    BOOST_CHECK_EQUAL(1, typer.computeType(false,
                                              "<L_MSG MN=\"ml12213\" PID=\"ml010_managed2\" TID=\"asyncDelivery44\" DT=\"\" PT=\"ERROR\" AP=\"wts\" DN=\"\" "
                                              "SN=\"\" SR=\"co.elastic.session.ejb.FxCoverSessionBean\">javax.ejb.FinderException - findFxCover([]): "
                                              "null</L_MSG>",
                                              500));
}

BOOST_AUTO_TEST_CASE(testApacheData) {
    TTokenListDataTyperKeepsFields typer(NO_REVERSE_SEARCH_CREATOR, 0.7, "whatever");

    BOOST_CHECK_EQUAL(1, typer.computeType(false, " org.apache.coyote.http11.Http11BaseProtocol destroy",
                                              500));
    BOOST_CHECK_EQUAL(2, typer.computeType(false, " org.apache.coyote.http11.Http11BaseProtocol init",
                                              500));
    BOOST_CHECK_EQUAL(3, typer.computeType(false, " org.apache.coyote.http11.Http11BaseProtocol start",
                                              500));
    BOOST_CHECK_EQUAL(4, typer.computeType(false, " org.apache.coyote.http11.Http11BaseProtocol stop",
                                              500));
}

BOOST_AUTO_TEST_CASE(testBrokerageData) {
    TTokenListDataTyperKeepsFields typer(NO_REVERSE_SEARCH_CREATOR, 0.7, "whatever");

    BOOST_CHECK_EQUAL(
        1, typer.computeType(false,
                             "AUDIT  ; tomcat-http--16; ee96c0c4567c0c11d6b90f9bc8b54aaa77; REQ4e42023e0a0328d020003e460005aa33; "
                             "applnx911.elastic.co; ; Request Complete: /mlgw/mlb/ofsummary/summary "
                             "[T=283ms,CUSTPREF-WEB_ACCOUNT_PREFERENCES=95,MAUI-ETSPROF2=155,NBMSG-NB_MESSAGING_SERVICE=164,CustAcctProfile="
                             "BRK=2;NB=0;FILI=0;CESG=0;CC=0;AcctTotal=2,migrated=2]",
                             500));
    BOOST_CHECK_EQUAL(2, typer.computeType(false,
                                              "AUDIT  ; tomcat-http--39; ee763e95747c0b11d6b90f9bc8b54aaa77; REQ4e42023e0a0429a020000c6f0002aa33; "
                                              "applnx811.elastic.co; ; Request Complete: /mlgw/mlb/ofaccounts/brokerageAccountHistory "
                                              "[T=414ms,CUSTPREF-INS_PERSON_WEB_ACCT_PREFERENCES=298,MAUI-PSL04XD=108]",
                                              500));
    BOOST_CHECK_EQUAL(3, typer.computeType(false,
                                              "AUDIT  ; tomcat-http--39; ee256201da7c0c11d6b90f9bc8b54aaa77; REQ4e42023b0a022925200027180002aa33; "
                                              "applnx711.elastic.co; ; Request Complete: /mlgw/mlb/ofpositions/brokerageAccountPositionsIframe "
                                              "[T=90ms,CacheStore-GetAttribute=5,MAUI-ECAPPOS=50,RR-QUOTE_TRANSACTION=11]",
                                              500));
}

BOOST_AUTO_TEST_CASE(testVmwareData) {
    TTokenListDataTyperKeepsFields typer(NO_REVERSE_SEARCH_CREATOR, 0.7, "whatever");

    BOOST_CHECK_EQUAL(1, typer.computeType(false, "Vpxa: [49EC0B90 verbose 'VpxaHalCnxHostagent' opID=WFU-ddeadb59] [WaitForUpdatesDone] Received callback",
                                              103));
    BOOST_CHECK_EQUAL(2, typer.computeType(false, "Vpxa: [49EC0B90 verbose 'Default' opID=WFU-ddeadb59] [VpxaHalVmHostagent] 11: GuestInfo changed 'guest.disk",
                                              107));
    BOOST_CHECK_EQUAL(3, typer.computeType(false, "Vpxa: [49EC0B90 verbose 'VpxaHalCnxHostagent' opID=WFU-ddeadb59] [WaitForUpdatesDone] Completed callback",
                                              104));
    BOOST_CHECK_EQUAL(1, typer.computeType(false, "Vpxa: [49EC0B90 verbose 'VpxaHalCnxHostagent' opID=WFU-35689729] [WaitForUpdatesDone] Received callback",
                                              103));
    BOOST_CHECK_EQUAL(2, typer.computeType(false, "Vpxa: [49EC0B90 verbose 'Default' opID=WFU-35689729] [VpxaHalVmHostagent] 15: GuestInfo changed 'guest.disk",
                                              107));
    BOOST_CHECK_EQUAL(3, typer.computeType(false, "Vpxa: [49EC0B90 verbose 'VpxaHalCnxHostagent' opID=WFU-35689729] [WaitForUpdatesDone] Completed callback",
                                              104));
}

BOOST_AUTO_TEST_CASE(testBankData) {
    TTokenListDataTyperKeepsFields typer(NO_REVERSE_SEARCH_CREATOR, 0.7, "whatever");

    BOOST_CHECK_EQUAL(1, typer.computeType(false,
                                              "INFO  [co.elastic.settlement.synchronization.PaymentFlowProcessorImpl] Process payment flow "
                                              "for tradeId=80894728 and backOfficeId=9354474",
                                              500));
    BOOST_CHECK_EQUAL(2, typer.computeType(false,
                                              "INFO  [co.elastic.settlement.synchronization.PaymentFlowProcessorImpl] Synchronization of "
                                              "payment flow is complete for tradeId=80013186 and backOfficeId=265573",
                                              500));

    // This is not great, but it's tricky when only 1 word differs from the
    // first type
    BOOST_CHECK_EQUAL(1, typer.computeType(false,
                                              "INFO  [co.elastic.settlement.synchronization.PaymentFlowProcessorImpl] Synchronize payment "
                                              "flow for tradeId=80894721 and backOfficeId=9354469",
                                              500));
}

BOOST_AUTO_TEST_CASE(testJavaGcData) {
    TTokenListDataTyperKeepsFields typer(NO_REVERSE_SEARCH_CREATOR, 0.7, "whatever");

    BOOST_CHECK_EQUAL(
        1, typer.computeType(false, "2016-04-27T19:57:43.644-0700: 1922084.903: [GC", 46));
    BOOST_CHECK_EQUAL(
        1, typer.computeType(false, "2016-04-28T19:57:43.644-0700: 1922084.903: [GC", 46));
    BOOST_CHECK_EQUAL(
        1, typer.computeType(false, "2016-04-29T19:57:43.644-0700: 1922084.903: [GC", 46));
    BOOST_CHECK_EQUAL(
        1, typer.computeType(false, "2016-04-30T19:57:43.644-0700: 1922084.903: [GC", 46));
    BOOST_CHECK_EQUAL(
        1, typer.computeType(false, "2016-04-30T19:57:43.644-0700: 1922084.904: [GC", 46));
    BOOST_CHECK_EQUAL(
        1, typer.computeType(false, "2016-04-30T19:57:43.644-0700: 1922084.905: [GC", 46));
    BOOST_CHECK_EQUAL(
        1, typer.computeType(false, "2016-04-30T19:57:43.644-0700: 1922084.906: [GC", 46));
    BOOST_CHECK_EQUAL(
        1, typer.computeType(false, "2016-04-30T19:57:43.644-0700: 1922085.906: [GC", 46));
    BOOST_CHECK_EQUAL(
        1, typer.computeType(false, "2016-04-30T19:57:43.644-0700: 1922086.906: [GC", 46));
    BOOST_CHECK_EQUAL(
        1, typer.computeType(false, "2016-04-30T19:57:43.644-0700: 1922087.906: [GC", 46));
    BOOST_CHECK_EQUAL(
        1, typer.computeType(false, "2016-04-30T19:57:43.645-0700: 1922087.906: [GC", 46));
    BOOST_CHECK_EQUAL(
        1, typer.computeType(false, "2016-04-30T19:57:43.646-0700: 1922087.906: [GC", 46));
    BOOST_CHECK_EQUAL(
        1, typer.computeType(false, "2016-04-30T19:57:43.647-0700: 1922087.906: [GC", 46));

    BOOST_CHECK_EQUAL(2, typer.computeType(false, "PSYoungGen      total 2572800K, used 1759355K [0x0000000759500000, 0x0000000800000000, 0x0000000800000000)",
                                              106));
    BOOST_CHECK_EQUAL(2, typer.computeType(false, "PSYoungGen      total 2572801K, used 1759355K [0x0000000759500000, 0x0000000800000000, 0x0000000800000000)",
                                              106));
    BOOST_CHECK_EQUAL(2, typer.computeType(false, "PSYoungGen      total 2572802K, used 1759355K [0x0000000759500000, 0x0000000800000000, 0x0000000800000000)",
                                              106));
    BOOST_CHECK_EQUAL(2, typer.computeType(false, "PSYoungGen      total 2572803K, used 1759355K [0x0000000759500000, 0x0000000800000000, 0x0000000800000000)",
                                              106));
    BOOST_CHECK_EQUAL(2, typer.computeType(false, "PSYoungGen      total 2572803K, used 1759355K [0x0000000759600000, 0x0000000800000000, 0x0000000800000000)",
                                              106));
    BOOST_CHECK_EQUAL(2, typer.computeType(false, "PSYoungGen      total 2572803K, used 1759355K [0x0000000759700000, 0x0000000800000000, 0x0000000800000000)",
                                              106));
    BOOST_CHECK_EQUAL(2, typer.computeType(false, "PSYoungGen      total 2572803K, used 1759355K [0x0000000759800000, 0x0000000800000000, 0x0000000800000000)",
                                              106));
}

BOOST_AUTO_TEST_CASE(testPersist) {
    TTokenListDataTyperKeepsFields origTyper(NO_REVERSE_SEARCH_CREATOR, 0.7, "whatever");

    origTyper.computeType(false, "<ml13-4608.1.p2ps: Info: > Source ML_SERVICE2 on 13122:867 has shut down.",
                          500);
    origTyper.computeType(false, "<ml13-4602.1.p2ps: Info: > Source MONEYBROKER on 13112:736 has shut down.",
                          500);
    origTyper.computeType(false, "<ml13-4606.1.p2ps: Info: > Source CUBE_LIQUID on 13188:2010 has shut down.",
                          500);
    origTyper.computeType(false, "<ml13-4608.1.p2ps: Info: > Source ML SERVICE2 on 13122:867 has shut down.",
                          500);
    origTyper.computeType(false, "<ml13-4602.1.p2ps: Info: > Source MONEYBROKER on 13112:736 has started.",
                          500);
    origTyper.computeType(false, "<ml13-4608.1.p2ps: Info: > Source ML_SERVICE2 on 13122:867 has started.",
                          500);
    origTyper.computeType(false, "<ml00-4201.1.p2ps: Info: > Service CUBE_CHIX, id of 132, has started.",
                          500);
    origTyper.computeType(false, "<ml00-4601.1.p2ps: Info: > Service CUBE_IDEM, id of 232, has started.",
                          500);
    origTyper.computeType(false, "<ml00-4601.1.p2ps: Info: > Service CUBE_IDEM, id of 232, has started.",
                          500);
    origTyper.computeType(
        false, "<ml00-4201.1.p2ps: Info: > Service CUBE_CHIX has shut down.", 500);

    std::string origXml;
    {
        ml::core::CRapidXmlStatePersistInserter inserter("root");
        origTyper.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_DEBUG(<< "Typer XML representation:\n" << origXml);

    // Restore the XML into a new typer
    TTokenListDataTyperKeepsFields restoredTyper(NO_REVERSE_SEARCH_CREATOR, 0.7, "whatever");
    {
        ml::core::CRapidXmlParser parser;
        BOOST_TEST(parser.parseStringIgnoreCdata(origXml));
        ml::core::CRapidXmlStateRestoreTraverser traverser(parser);
        BOOST_TEST(traverser.traverseSubLevel(
            std::bind(&TTokenListDataTyperKeepsFields::acceptRestoreTraverser,
                      &restoredTyper, std::placeholders::_1)));
    }

    // The XML representation of the new typer should be the same as the original
    std::string newXml;
    {
        ml::core::CRapidXmlStatePersistInserter inserter("root");
        restoredTyper.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    BOOST_CHECK_EQUAL(origXml, newXml);
}

BOOST_AUTO_TEST_CASE(testLongReverseSearch) {
    TTokenListDataTyperKeepsFields::TTokenListReverseSearchCreatorIntfCPtr reverseSearchCreator(
        new ml::api::CTokenListReverseSearchCreator("_raw"));
    TTokenListDataTyperKeepsFields typer(reverseSearchCreator, 0.7, "_raw");

    // Create a long message with lots of junk that will create a ridiculous
    // reverse search if not constrained
    std::string longMessage("a few dictionary words to start off");
    for (size_t i = 1; i < 26; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            longMessage += ' ';
            longMessage.append(20, char('a' + j));
        }
    }
    LOG_DEBUG(<< "Long message is: " << longMessage);

    // Only 1 message so must be type 1
    int type = typer.computeType(false, longMessage, longMessage.length());
    BOOST_CHECK_EQUAL(1, type);

    std::string terms;
    std::string regex;
    size_t maxMatchingLength(0);
    bool wasCached(false);

    // Only 1 message so the reverse search COULD include all the tokens, but
    // shouldn't because such a reverse search would be ridiculously long
    BOOST_TEST(typer.createReverseSearch(type, terms, regex, maxMatchingLength, wasCached));

    BOOST_TEST(!wasCached);
    LOG_DEBUG(<< "Terms length: " << terms.length());
    LOG_TRACE(<< "Terms: " << terms);
    LOG_DEBUG(<< "Regex length: " << regex.length());
    LOG_TRACE(<< "Regex: " << regex);
    BOOST_TEST(terms.length() + regex.length() <= 10000);
    BOOST_CHECK_EQUAL(longMessage.length() * 11 / 10, maxMatchingLength);

    // It should include the dictionary words (but note that the single letter
    // 'a' isn't currently considered a dictionary word)
    BOOST_TEST(terms.find("few") != std::string::npos);
    BOOST_TEST(terms.find("dictionary") != std::string::npos);
    BOOST_TEST(terms.find("words") != std::string::npos);
    BOOST_TEST(terms.find("to") != std::string::npos);
    BOOST_TEST(terms.find("start") != std::string::npos);
    BOOST_TEST(terms.find("off") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(testPreTokenised) {
    TTokenListDataTyperKeepsFields typer(NO_REVERSE_SEARCH_CREATOR, 0.7, "whatever");

    BOOST_CHECK_EQUAL(1, typer.computeType(false, "<ml13-4608.1.p2ps: Info: > Source ML_SERVICE2 on 13122:867 has shut down.",
                                              500));
    BOOST_CHECK_EQUAL(1, typer.computeType(false, "<ml13-4602.1.p2ps: Info: > Source MONEYBROKER on 13112:736 has shut down.",
                                              500));
    BOOST_CHECK_EQUAL(1, typer.computeType(false, "<ml13-4606.1.p2ps: Info: > Source CUBE_LIQUID on 13188:2010 has shut down.",
                                              500));
    BOOST_CHECK_EQUAL(1, typer.computeType(false, "<ml13-4608.1.p2ps: Info: > Source ML SERVICE2 on 13122:867 has shut down.",
                                              500));
    BOOST_CHECK_EQUAL(2, typer.computeType(false, "<ml13-4602.1.p2ps: Info: > Source MONEYBROKER on 13112:736 has started.",
                                              500));
    BOOST_CHECK_EQUAL(2, typer.computeType(false, "<ml13-4608.1.p2ps: Info: > Source ML_SERVICE2 on 13122:867 has started.",
                                              500));
    BOOST_CHECK_EQUAL(3, typer.computeType(false, "<ml00-4201.1.p2ps: Info: > Service CUBE_CHIX, id of 132, has started.",
                                              500));
    BOOST_CHECK_EQUAL(3, typer.computeType(false, "<ml00-4601.1.p2ps: Info: > Service CUBE_IDEM, id of 232, has started.",
                                              500));
    BOOST_CHECK_EQUAL(3, typer.computeType(false, "<ml00-4601.1.p2ps: Info: > Service CUBE_IDEM, id of 232, has started.",
                                              500));
    BOOST_CHECK_EQUAL(4, typer.computeType(false, "<ml00-4201.1.p2ps: Info: > Service CUBE_CHIX has shut down.",
                                              500));

    TTokenListDataTyperKeepsFields::TStrStrUMap fields;

    // The pre-tokenised tokens exactly match those of the other message in
    // category 4, so this should get put it category 4
    fields[TTokenListDataTyperKeepsFields::PRETOKENISED_TOKEN_FIELD] =
        "ml00-4201.1.p2ps,Info,Service,CUBE_CHIX,has,shut,down";
    BOOST_CHECK_EQUAL(4, typer.computeType(false, fields, "<ml00-4201.1.p2ps: Info: > Service CUBE_CHIX has shut down.",
                                              500));

    // Here we cheat.  The pre-tokenised tokens exactly match those of the
    // first message, so this should get put in category 1.  But the full
    // message is indentical to that of the category 4 message, so if this test
    // ever fails with the message being put in category 4 then it probably
    // means there's a bug where the pre-tokenised tokens are being ignored.
    // (Obviously in production we wouldn't get the discrepancy between the
    // pre-tokenised tokens and the full message.)
    fields[TTokenListDataTyperKeepsFields::PRETOKENISED_TOKEN_FIELD] =
        "ml13-4608.1.p2ps,Info,Source,ML_SERVICE2,on,has,shut,down";
    BOOST_CHECK_EQUAL(1, typer.computeType(false, fields, "<ml00-4201.1.p2ps: Info: > Service CUBE_CHIX has shut down.",
                                              500));

    // Similar principle, but with Chinese, Japanese and Korean tokens, so
    // should go in a new category.
    fields[TTokenListDataTyperKeepsFields::PRETOKENISED_TOKEN_FIELD] = "编码,コーディング,코딩";
    BOOST_CHECK_EQUAL(5, typer.computeType(false, fields, "<ml00-4201.1.p2ps: Info: > Service CUBE_CHIX has shut down.",
                                              500));
}

BOOST_AUTO_TEST_CASE(testPreTokenisedPerformance) {
    static const size_t TEST_SIZE(100000);
    ml::core::CStopWatch stopWatch;

    uint64_t inlineTokenisationTime(0);
    {
        TTokenListDataTyperKeepsFields typer(NO_REVERSE_SEARCH_CREATOR, 0.7, "whatever");

        LOG_DEBUG(<< "Before test with inline tokenisation");

        stopWatch.start();
        for (size_t count = 0; count < TEST_SIZE; ++count) {
            BOOST_CHECK_EQUAL(1, typer.computeType(false, "Vpxa: [49EC0B90 verbose 'VpxaHalCnxHostagent' opID=WFU-ddeadb59] [WaitForUpdatesDone] Received callback",
                                                      103));
        }
        inlineTokenisationTime = stopWatch.stop();

        LOG_DEBUG(<< "After test with inline tokenisation");
        LOG_DEBUG(<< "Inline tokenisation test took " << inlineTokenisationTime << "ms");
    }

    stopWatch.reset();

    TTokenListDataTyperKeepsFields::TStrStrUMap fields;
    fields[TTokenListDataTyperKeepsFields::PRETOKENISED_TOKEN_FIELD] =
        "Vpxa,verbose,VpxaHalCnxHostagent,opID,WFU-ddeadb59,WaitForUpdatesDone,Received,callback";

    uint64_t preTokenisationTime(0);
    {
        TTokenListDataTyperKeepsFields typer(NO_REVERSE_SEARCH_CREATOR, 0.7, "whatever");

        LOG_DEBUG(<< "Before test with pre-tokenisation");

        stopWatch.start();
        for (size_t count = 0; count < TEST_SIZE; ++count) {
            BOOST_CHECK_EQUAL(1, typer.computeType(false, fields, "Vpxa: [49EC0B90 verbose 'VpxaHalCnxHostagent' opID=WFU-ddeadb59] [WaitForUpdatesDone] Received callback",
                                                      103));
        }
        preTokenisationTime = stopWatch.stop();

        LOG_DEBUG(<< "After test with pre-tokenisation");
        LOG_DEBUG(<< "Pre-tokenisation test took " << preTokenisationTime << "ms");
    }

    const char* keepGoingEnvVar{std::getenv("ML_KEEP_GOING")};
    bool likelyInCi = (keepGoingEnvVar != nullptr && *keepGoingEnvVar != '\0');
    if (likelyInCi) {
        // CI is most likely running on a VM, and this test can fail quite often
        // due to the VM stalling or being slowed down by noisy neighbours
        LOG_INFO(<< "Skipping test pre-tokenised performance assertion");
    } else {
        BOOST_TEST(preTokenisationTime <= inlineTokenisationTime);
    }
}

BOOST_AUTO_TEST_SUITE_END()
