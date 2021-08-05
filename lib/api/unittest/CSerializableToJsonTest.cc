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

#include <api/CSerializableToJson.h>

#include <boost/test/tools/interface.hpp>
#include <rapidjson/istreamwrapper.h>
#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>
#include <rapidjson/document.h>

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

    void addCompressedToJsonStream(TRapidJsonWriter& writer) const override {
        this->CSerializableToCompressedChunkedJson::addCompressedToJsonStream(
            "state_doc", "state", writer);
    }

private:
    void addToJsonStream(TGenericLineWriter& writer) const override {
        writer.StartObject();
        writer.Key("name");
        writer.String(m_Name);
        writer.EndObject();
        writer.StartObject();
        writer.Key("values");
        writer.StartArray();
        for (const auto& value : m_Values) {
            writer.Double(value);
        }
        writer.EndArray();
        writer.EndObject();
    }

    void readFromJsonStream(TIStreamPtr inputStream) {
        if (inputStream != nullptr) {
            rapidjson::IStreamWrapper isw{*inputStream};
            rapidjson::Document doc;
            doc.ParseStream<rapidjson::kParseStopWhenDoneFlag>(isw);
            m_Name = doc["name"].GetString();
            doc.ParseStream<rapidjson::kParseStopWhenDoneFlag>(isw);
            m_Values.reserve(doc["values"].Size());
            for (const auto& value : doc["values"].GetArray()) {
                m_Values.push_back(value.GetDouble());
            }
        }
    }

private:
    std::string m_Name;
    TDoubleVec m_Values;
};

void arrayToNdJson(std::string array, std::ostream& ndjson) {
    array.erase(std::remove(array.begin(), array.end(), '\n'), array.end());
    rapidjson::Document doc;
    doc.Parse(array);
    for (const auto& chunk : doc.GetArray()) {
        rapidjson::StringBuffer string;
        core::CRapidJsonLineWriter<rapidjson::StringBuffer> printer{string};
        chunk.Accept(printer);
        ndjson << string.GetString();
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
            CSerializableVector::TRapidJsonWriter writer{osw};
            original.addCompressedToJsonStream(writer);
        }
        arrayToNdJson(storage.str(), *ndjson);

        CSerializableVector restored;
        restored.fromCompressedJsonStream(ndjson);

        auto result = restored.compare(original);
        if (result.empty() == false) {
            BOOST_TEST_FAIL(result);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
