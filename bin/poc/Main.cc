/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CBase64Filter.h>
#include <core/CLogger.h>
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CRapidJsonConcurrentLineWriter.h>

#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>

#include <sstream>
namespace {
using TFilteredInput = boost::iostreams::filtering_stream<boost::iostreams::input>;
using TFilteredOutput = boost::iostreams::filtering_stream<boost::iostreams::output>;
}

using namespace ml;
int main(int /*argc*/, char** /*argv*/) {

    std::stringstream compressedStream;
    {
    TFilteredOutput outFilter;
    outFilter.push(boost::iostreams::gzip_compressor());
    outFilter.push(core::CBase64Encoder());
    outFilter.push(compressedStream);
        core::CJsonOutputStreamWrapper streamWrapper{outFilter};
        core::CRapidJsonConcurrentLineWriter writer{streamWrapper};
        writer.StartObject();
        writer.Key("foo");
        writer.Int(1);
        writer.EndObject();
    }

    LOG_DEBUG(<< "Compressed: " << compressedStream.str());

    std::string decompressedString;
    {
        TFilteredInput inFilter;
        inFilter.push(boost::iostreams::gzip_decompressor());
        inFilter.push(core::CBase64Decoder());
        inFilter.push(compressedStream);
        inFilter >> decompressedString;
    }
    
    LOG_DEBUG(<< "Decompressed: " << decompressedString);

    return EXIT_SUCCESS;
}
