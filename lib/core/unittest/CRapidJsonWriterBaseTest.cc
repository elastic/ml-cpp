/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CRapidJsonWriterBaseTest.h"


#include <core/CLogger.h>
#include <core/CRapidJsonWriterBase.h>
#include <core/CStringUtils.h>

#include <rapidjson/document.h>
#include <rapidjson/ostreamwrapper.h>

#include <limits>
#include <sstream>


CppUnit::Test *CRapidJsonWriterBaseTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CRapidJsonWriterBaseTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CRapidJsonWriterBaseTest>(
                                   "CRapidJsonWriterBaseTest::testAddFields",
                                   &CRapidJsonWriterBaseTest::testAddFields) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CRapidJsonWriterBaseTest>(
                                   "CRapidJsonWriterBaseTest::testRemoveMemberIfPresent",
                                   &CRapidJsonWriterBaseTest::testRemoveMemberIfPresent) );

    return suiteOfTests;
}

namespace
{
const std::string STR_NAME("str");
const std::string EMPTY1_NAME("empty1");
const std::string EMPTY2_NAME("empty2");
const std::string DOUBLE_NAME("double");
const std::string NAN_NAME("nan");
const std::string INFINITY_NAME("infinity");
const std::string BOOL_NAME("bool");
const std::string INT_NAME("int");
const std::string UINT_NAME("uint");
const std::string STR_ARRAY_NAME("str[]");
const std::string DOUBLE_ARRAY_NAME("double[]");
const std::string NAN_ARRAY_NAME("nan[]");
const std::string TTIME_ARRAY_NAME("TTime[]");
}

void CRapidJsonWriterBaseTest::testAddFields(void)
{
    std::ostringstream strm;
    rapidjson::OStreamWrapper writeStream(strm);
    typedef ml::core::CRapidJsonWriterBase<rapidjson::OStreamWrapper, rapidjson::UTF8<>, rapidjson::UTF8<>,
            rapidjson::CrtAllocator> TGenericLineWriter;
    TGenericLineWriter writer(writeStream);

    rapidjson::Document doc = writer.makeDoc();;

    writer.addStringFieldCopyToObj(STR_NAME, "hello", doc);
    writer.addStringFieldCopyToObj(EMPTY1_NAME, "", doc);
    writer.addStringFieldCopyToObj(EMPTY2_NAME, "", doc, true);
    writer.addDoubleFieldToObj(DOUBLE_NAME, 1.77e-156, doc); 
    writer.addDoubleFieldToObj(NAN_NAME, std::numeric_limits<double>::quiet_NaN(), doc);
    writer.addDoubleFieldToObj(INFINITY_NAME, std::numeric_limits<double>::infinity(), doc);
    writer.addBoolFieldToObj(BOOL_NAME, false, doc);
    writer.addIntFieldToObj(INT_NAME, -9, doc);
    writer.addUIntFieldToObj(UINT_NAME, 999999999999999ull, doc);
    writer.addStringArrayFieldToObj(STR_ARRAY_NAME,    TGenericLineWriter::TStrVec(3, "blah"), doc);
    writer.addDoubleArrayFieldToObj(DOUBLE_ARRAY_NAME, TGenericLineWriter::TDoubleVec(10, 1.5), doc);
    writer.addDoubleArrayFieldToObj(NAN_ARRAY_NAME,    TGenericLineWriter::TDoubleVec(2, std::numeric_limits<double>::quiet_NaN()), doc);
    writer.addTimeArrayFieldToObj(TTIME_ARRAY_NAME,    TGenericLineWriter::TTimeVec(2, 1421421421), doc);

    writer.write(doc);
    writer.Flush();

    std::string printedDoc(strm.str());
    ml::core::CStringUtils::trimWhitespace(printedDoc);

    LOG_DEBUG("Printed doc is: " << printedDoc);

    std::string expectedDoc("{"
                                "\"str\":\"hello\","
                                "\"empty2\":\"\","
                                "\"double\":1.77e-156,"
                                "\"nan\":0,"
                                "\"infinity\":0,"
                                "\"bool\":false,"
                                "\"int\":-9,"
                                "\"uint\":999999999999999,"
                                "\"str[]\":[\"blah\",\"blah\",\"blah\"],"
                                "\"double[]\":[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],"
                                "\"nan[]\":[0,0],"
                                "\"TTime[]\":[1421421421000,1421421421000]"
                            "}");

    CPPUNIT_ASSERT_EQUAL(expectedDoc, printedDoc);
}


void CRapidJsonWriterBaseTest::testRemoveMemberIfPresent(void)
{
    std::ostringstream strm;
    rapidjson::OStreamWrapper writeStream(strm);
    typedef ml::core::CRapidJsonWriterBase<rapidjson::OStreamWrapper, rapidjson::UTF8<>, rapidjson::UTF8<>,
            rapidjson::CrtAllocator> TGenericLineWriter;
    TGenericLineWriter writer(writeStream);

    rapidjson::Document doc = writer.makeDoc();;

    std::string foo("foo");

    writer.addStringFieldCopyToObj(foo, "42", doc);
    CPPUNIT_ASSERT(doc.HasMember(foo));

    writer.removeMemberIfPresent(foo, doc);
    CPPUNIT_ASSERT(doc.HasMember(foo) == false);

    writer.removeMemberIfPresent(foo, doc);
    CPPUNIT_ASSERT(doc.HasMember(foo) == false);
}
