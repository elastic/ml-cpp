/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
#include <core/CHexUtils.h>

#include <core/CoreTypes.h>

#include <iomanip>
#include <ios>
#include <iostream>
#include <string>

#include <ctype.h>

namespace ml {
namespace core {

CHexUtils::CHexUtils(const uint8_t* pkt, size_t pktLen, bool printHeader, bool printAscii)
    : m_Pkt(pkt), m_PktLen(pktLen), m_PrintHeader(printHeader), m_PrintAscii(printAscii) {
}

CHexUtils::CHexUtils(const TDataVec& data, bool printHeader, bool printAscii)
    : m_Pkt((data.size() > 0) ? &data[0] : 0), m_PktLen(data.size()), m_PrintHeader(printHeader), m_PrintAscii(printAscii) {
}

void CHexUtils::dump(const uint8_t* pkt, size_t pktLen) {
    CHexUtils hex(pkt, pktLen);

    std::cout << hex << std::endl;
}

std::ostream& operator<<(std::ostream& strm, const CHexUtils& hex) {
    if (hex.m_PrintHeader) {
        strm << "DataSize: " << hex.m_PktLen << " {" << core_t::LINE_ENDING;
    }

    if (hex.m_Pkt != 0) {
        strm << std::hex;

        std::string text;

        for (size_t i = 0; i < hex.m_PktLen; ++i) {
            strm << std::setfill('0') << std::setw(2) << static_cast<uint32_t>(hex.m_Pkt[i]) << ' ';

            if (::isprint(hex.m_Pkt[i])) {
                text += static_cast<char>(hex.m_Pkt[i]);
            } else {
                text += '.';
            }

            if (((i + 1) % 8) == 0) {
                strm << ' ';
            }

            if (hex.m_PrintAscii && ((i + 1) % 16) == 0) {
                strm << text << core_t::LINE_ENDING;
                text.clear();
            }
        }

        if (hex.m_PrintAscii && (hex.m_PktLen % 16) != 0) {
            // pad space
            size_t max(((hex.m_PktLen / 16) + 1) * 16);
            for (size_t i = hex.m_PktLen; i <= max; ++i) {
                if (i != max) {
                    strm << "   ";
                }

                if (((i + 1) % 8) == 0) {
                    strm << ' ';
                }
            }
            strm << text << core_t::LINE_ENDING;
        }

        strm << std::dec;
    }

    if (hex.m_PrintHeader) {
        strm << '}';
    }

    return strm;
}
}
}
