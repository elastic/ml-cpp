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

#include <core/CBoostJsonUnbufferedIStreamWrapper.h>

#include <api/CSerializableToJson.h>

#include <test/CRandomNumbers.h>

#include <boost/test/tools/interface.hpp>
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <sstream>

BOOST_AUTO_TEST_SUITE(CSerializableToJsonTest)

using namespace ml;

namespace {
using TDoubleVec = std::vector<double>;

class CSerializableVector : public api::CSerializableToCompressedChunkedJson,
                            public api::CSerializableFromCompressedChunkedJson {
public:
    CSerializableVector() = default;
    CSerializableVector(std::string name, TDoubleVec values)
        : CSerializableToCompressedChunkedJson{50}, m_Name{std::move(name)}, m_Values{std::move(values)} {
    }

    std::string compare(const CSerializableVector& rhs) const {
        if (m_Name != rhs.m_Name) {
            return m_Name + " vs " + rhs.m_Name;
        }
        if (m_Values.size() != rhs.m_Values.size()) {
            return std::to_string(m_Values.size()) + " vs " +
                   std::to_string(rhs.m_Values.size());
        }
        for (std::size_t i = 0; i < m_Values.size(); ++i) {
            if (std::fabs(m_Values[i] - rhs.m_Values[i]) > 1e-6) {
                return std::to_string(i) + "/" + std::to_string(m_Values[i]) +
                       " vs " + std::to_string(rhs.m_Values[i]);
            }
        }
        return "";
    }

    void fromCompressedJsonStream(TIStreamPtr inputStream) {
        std::stringstream buffer;
        auto state = rawJsonStream("state_doc", "state", std::move(inputStream), buffer);
        this->readFromJsonStream(std::move(state));
    }

    void addCompressedToJsonStream(TBoostJsonWriter& writer) const override {
        this->CSerializableToCompressedChunkedJson::addCompressedToJsonStream(
            "state_doc", "state", writer);
    }

private:
    void addToJsonStream(TGenericLineWriter& writer) const override {
        writer.onObjectBegin();
        writer.onKey("name");
        writer.onString(m_Name);
        writer.onKey("values");
        writer.onArrayBegin();
        for (const auto& value : m_Values) {
            writer.onDouble(value);
        }
        writer.onArrayEnd();
        writer.onObjectEnd();
    }

    void readFromJsonStream(TIStreamPtr inputStream) {
        if (inputStream != nullptr) {
            json::error_code ec;
            json::parse_options opts;
            opts.numbers = json::number_precision::precise;
            json::value doc = json::parse(*inputStream, ec, {}, opts);
            BOOST_TEST_REQUIRE(ec.failed() == false);
            BOOST_TEST_REQUIRE(doc.is_object());

            m_Name = doc.as_object().at("name").as_string();
            m_Values.reserve(doc.as_object().at("values").as_array().size());
            for (const auto& value : doc.as_object().at("values").as_array()) {
                double d = value.to_number<double>();
                m_Values.push_back(d);
            }
        }
    }

private:
    std::string m_Name;
    TDoubleVec m_Values;
};

void arrayToNdJson(std::string array, std::ostream& ndjson) {
    array.erase(std::remove(array.begin(), array.end(), '\n'), array.end());
    json::error_code ec;
    json::value doc = json::parse(array, ec);
    BOOST_TEST_REQUIRE(ec.failed() == false);
    BOOST_TEST_REQUIRE(doc.is_array());

    for (const auto& chunk : doc.as_array()) {
        std::string string;
        core::CStringBufWriter printer{string};
        printer.write(chunk);
        ndjson << string;
    }
}
}

BOOST_AUTO_TEST_CASE(testRoundTrip) {

    test::CRandomNumbers rng;

    for (std::size_t number = 10; number < 200; number += 10) {

        TDoubleVec values;
        rng.generateUniformSamples(0.0, 100.0, number, values);

        CSerializableVector original{"test_vector", std::move(values)};

        auto ndjson = std::make_shared<std::stringstream>();
        std::ostringstream storage;
        {
            core::CJsonOutputStreamWrapper osw{storage};
            CSerializableVector::TBoostJsonWriter writer{osw};
            original.addCompressedToJsonStream(writer);
        }
        arrayToNdJson(storage.str(), *ndjson);

        CSerializableVector restored;
        restored.fromCompressedJsonStream(ndjson);

        auto result = restored.compare(original);
        BOOST_TEST_REQUIRE(result.empty(), result);
    }
}

BOOST_AUTO_TEST_SUITE_END()
