/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CBase64Filter.h>
#include <core/CJsonOutputStreamWrapper.h>
#include <core/CLogger.h>
#include <core/CRapidJsonConcurrentLineWriter.h>

#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>

#include <sstream>
#include <string>
namespace {
using TFilteredInput = boost::iostreams::filtering_stream<boost::iostreams::input>;
using TFilteredOutput = boost::iostreams::filtering_stream<boost::iostreams::output>;

const std::size_t CHUNK_SIZE{8};

class MyChunker {
public:
    using char_type = char;
    //! Tell boost::iostreams what this filter is capable of
    struct category : public boost::iostreams::output,
                      public boost::iostreams::filter_tag,
                      public boost::iostreams::multichar_tag,
                      public boost::iostreams::closable_tag {};

public:
    MyChunker() : m_CurrentDocNum{0}, m_BytesConsumed{0} {}
    template<typename SINK>
    std::streamsize write(SINK& snk, const char_type* s, std::streamsize n) {
        std::streamsize written = 0;
        while (n > 0) {
            if (m_BytesConsumed == 0) {
                if (m_CurrentDocNum > 0) {
                    boost::iostreams::write(snk, "}},", 3);
                }

                std::string header(1, '{');

                header += '\"';
                header += "compressed_inference_model";
                header += "\" : { ";
                header += "\"doc_num\": " + std::to_string(m_CurrentDocNum) + ", ";

                LOG_TRACE(<< "Write: " << header);
                boost::iostreams::write(snk, header.c_str(), header.size());
            } else {
                LOG_TRACE(<< "Write: ,");
                boost::iostreams::write(snk, ",", 1);
            }
            this->writeInternal(snk, s, written, n);
            if (m_BytesConsumed >= CHUNK_SIZE || n == 0) {
                LOG_TRACE(<< "Terminated stream " << m_CurrentDocNum);
                this->closeStream(snk, false);
                m_BytesConsumed = 0;
            }
        }
        LOG_TRACE(<< "Returning " << written);
        return written;
    }
    template<typename SINK>
    void close(SINK& snk) {
        this->closeStream(snk, true);
    }

private:
    template<typename SINK>
    void writeInternal(SINK& snk, const char* s, std::streamsize& written, std::streamsize& n) {
        std::size_t bytesToWrite = std::min(std::size_t(n), CHUNK_SIZE - m_BytesConsumed);
        LOG_DEBUG(<< "Writing string: " << std::string(&s[written], bytesToWrite));
        boost::iostreams::write(snk, "\"definition\": \"", 15);
        boost::iostreams::write(snk, &s[written], bytesToWrite);
        boost::iostreams::write(snk, "\"", 1);
        written += bytesToWrite;
        n -= bytesToWrite;
        m_BytesConsumed += bytesToWrite;
    }

    template<typename SINK>
    void closeStream(SINK& snk, bool isFinal) {
        std::string footer;
        if (isFinal) {
            footer += ",\"";
            footer += "eos";
            footer += "\":true}}";
        }
        LOG_DEBUG(<< "Write: " << footer);
        boost::iostreams::write(snk, footer.c_str(), footer.size());
        ++m_CurrentDocNum;
    }

private:
    //! The sequential document number currently being written to
    std::size_t m_CurrentDocNum;

    //! The number of bytes consumed from the input stream
    std::size_t m_BytesConsumed;
};
}

using namespace ml;
int main(int /*argc*/, char** /*argv*/) {

    std::stringstream compressedStream;
    {
        TFilteredOutput outFilter;
        outFilter.push(boost::iostreams::gzip_compressor());
        outFilter.push(core::CBase64Encoder());
        outFilter.push(MyChunker());
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
