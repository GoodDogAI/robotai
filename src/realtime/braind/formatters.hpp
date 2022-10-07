#pragma once

#include "msgvec.h"
#include <fmt/format.h>


template <> struct fmt::formatter<MsgVec::TimeoutResult>: formatter<string_view> {
  // parse is inherited from formatter<string_view>.
  template <typename FormatContext>
  auto format(MsgVec::TimeoutResult r, FormatContext& ctx) const {
    string_view name = "unknown";
    switch (r) {
        case MsgVec::TimeoutResult::MESSAGES_NOT_READY:
            name = "MESSAGES_NOT_READY";
            break;
        case MsgVec::TimeoutResult::MESSAGES_PARTIALLY_READY:
            name = "MESSAGES_PARTIALLY_READY";
            break;
        case MsgVec::TimeoutResult::MESSAGES_ALL_READY:
            name = "MESSAGES_ALL_READY";
            break;
    }
    return formatter<string_view>::format(name, ctx);
  }
};