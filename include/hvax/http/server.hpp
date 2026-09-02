#pragma once

#include <cstdint>
#include <vector>

#include "hvax/engine.hpp"

namespace hvax {

void run_server(Engine& engine);

// Exposed for deterministic validation of the PDF preprocessing step.
bool trim_pdf_white_margins(std::vector<uint8_t>& bytes, int64_t max_pixels);

}  // namespace hvax
