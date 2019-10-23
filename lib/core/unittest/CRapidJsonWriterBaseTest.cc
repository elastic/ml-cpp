/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CRapidJsonWriterBase.h>
#include <core/CStringUtils.h>

#include <rapidjson/document.h>
#include <rapidjson/ostreamwrapper.h>

#include <boost/test/unit_test.hpp>

#include <limits>
#include <sstream>

BOOST_AUTO_TEST_SUITE(CRapidJsonWriterBaseTest)

namespace {
const std::string STR_NAME("str");
const std::string EMPTY1_NAME("empty1");
const std::string EMPTY2_NAME("empty2");
const std::string DOUBLE_NAME("double");
const std::string NAN_NAME("nan");
const std::string INFINITY_NAME("infinity");
const std::string BOOL_NAME("bool");
const std::string INT_NAME("int");
const std::string TIME_NAME("time");
const std::string UINT_NAME("uint");
const std::string STR_ARRAY_NAME("str[]");
const std::string DOUBLE_ARRAY_NAME("double[]");
const std::string NAN_ARRAY_NAME("nan[]");
const std::string TTIME_ARRAY_NAME("TTime[]");
}

BOOST_AUTO_TEST_CASE(testAddFields) {
    std::ostringstream strm;
    rapidjson::OStreamWrapper writeStream(strm);
    using TGenericLineWriter =
        ml::core::CRapidJsonWriterBase<rapidjson::OStreamWrapper, rapidjson::UTF8<>,
                                       rapidjson::UTF8<>, rapidjson::CrtAllocator>;
    TGenericLineWriter writer(writeStream);

    rapidjson::Document doc = writer.makeDoc();
    ;

    writer.addStringFieldCopyToObj(STR_NAME, "hello", doc);
    writer.addStringFieldCopyToObj(EMPTY1_NAME, "", doc);
    writer.addStringFieldCopyToObj(EMPTY2_NAME, "", doc, true);
    writer.addDoubleFieldToObj(DOUBLE_NAME, 1.77e-156, doc);
    writer.addDoubleFieldToObj(NAN_NAME, std::numeric_limits<double>::quiet_NaN(), doc);
    writer.addDoubleFieldToObj(INFINITY_NAME, std::numeric_limits<double>::infinity(), doc);
    writer.addBoolFieldToObj(BOOL_NAME, false, doc);
    writer.addIntFieldToObj(INT_NAME, -9, doc);
    writer.addTimeFieldToObj(TIME_NAME, ml::core_t::TTime(1521035866), doc);
    writer.addUIntFieldToObj(UINT_NAME, 999999999999999ull, doc);
    writer.addStringArrayFieldToObj(STR_ARRAY_NAME,
                                    TGenericLineWriter::TStrVec(3, "blah"), doc);
    writer.addDoubleArrayFieldToObj(DOUBLE_ARRAY_NAME,
                                    TGenericLineWriter::TDoubleVec(10, 1.5), doc);
    writer.addDoubleArrayFieldToObj(
        NAN_ARRAY_NAME,
        TGenericLineWriter::TDoubleVec(2, std::numeric_limits<double>::quiet_NaN()), doc);
    writer.addTimeArrayFieldToObj(TTIME_ARRAY_NAME,
                                  TGenericLineWriter::TTimeVec(2, 1421421421), doc);

    writer.write(doc);
    writer.Flush();

    std::string printedDoc(strm.str());
    ml::core::CStringUtils::trimWhitespace(printedDoc);

    LOG_DEBUG(<< "Printed doc is: " << printedDoc);

    std::string expectedDoc("{"
                            "\"str\":\"hello\","
                            "\"empty2\":\"\","
                            "\"double\":1.77e-156,"
                            "\"nan\":0,"
                            "\"infinity\":0,"
                            "\"bool\":false,"
                            "\"int\":-9,"
                            "\"time\":1521035866000,"
                            "\"uint\":999999999999999,"
                            "\"str[]\":[\"blah\",\"blah\",\"blah\"],"
                            "\"double[]\":[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],"
                            "\"nan[]\":[0,0],"
                            "\"TTime[]\":[1421421421000,1421421421000]"
                            "}");

    BOOST_REQUIRE_EQUAL(expectedDoc, printedDoc);
}

BOOST_AUTO_TEST_CASE(testRemoveMemberIfPresent) {
    std::ostringstream strm;
    rapidjson::OStreamWrapper writeStream(strm);
    using TGenericLineWriter =
        ml::core::CRapidJsonWriterBase<rapidjson::OStreamWrapper, rapidjson::UTF8<>,
                                       rapidjson::UTF8<>, rapidjson::CrtAllocator>;
    TGenericLineWriter writer(writeStream);

    rapidjson::Document doc = writer.makeDoc();
    ;

    std::string foo("foo");

    writer.addStringFieldCopyToObj(foo, "42", doc);
    BOOST_TEST_REQUIRE(doc.HasMember(foo));

    writer.removeMemberIfPresent(foo, doc);
    BOOST_TEST_REQUIRE(doc.HasMember(foo) == false);

    writer.removeMemberIfPresent(foo, doc);
    BOOST_TEST_REQUIRE(doc.HasMember(foo) == false);
}

BOOST_AUTO_TEST_SUITE_END()
