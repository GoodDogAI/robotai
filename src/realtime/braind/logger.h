#pragma once

class Logger : public nvinfer1::ILogger
{
  void log(Severity severity, const char *msg) noexcept
  {
    // Would advise using a proper logging utility such as https://github.com/gabime/spdlog
    // For the sake of this tutorial, will just log to the console.

    // Only log Warnings or more important.
    if (severity <= Severity::kWARNING)
    {
      fmt::print(stderr, "{}", msg);
    }
  }
};