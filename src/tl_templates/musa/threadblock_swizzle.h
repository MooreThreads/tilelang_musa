#pragma once

#include "common.h"

namespace tl {

template <int panel_width> TL_DEVICE dim3 rasterization2DRow() {
  const unsigned int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
  const unsigned int panel_size = panel_width * gridDim.x;
  const unsigned int res_panel_width = gridDim.y % panel_width;
  const unsigned int total_panel = gridDim.y / panel_width * panel_width * gridDim.x;
  const unsigned int panel_idx = block_idx / panel_size;
  const unsigned int panel_offset = block_idx % panel_size;

  unsigned int mini_x;
  unsigned int row_idx, col_idx;
  if (block_idx >= total_panel) {
    col_idx = panel_offset / res_panel_width;
    mini_x = panel_offset % res_panel_width;
    mini_x = (col_idx & 1) ? res_panel_width - 1 - mini_x : mini_x;
  } else {
    col_idx = panel_offset / panel_width;
    mini_x = panel_offset % panel_width;
    mini_x = (col_idx & 1) ? panel_width - 1 - mini_x : mini_x;
  }

  if (panel_idx & 1) {
    col_idx = gridDim.x - 1 - col_idx;
  }
  row_idx = panel_idx * panel_width + mini_x;

  return {col_idx, row_idx, blockIdx.z};
}

template <int panel_width> TL_DEVICE dim3 rasterization2DColumn() {
  const unsigned int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
  const unsigned int panel_size = panel_width * gridDim.y;
  const unsigned int res_panel_width = gridDim.x % panel_width;
  const unsigned int total_panel = gridDim.x / panel_width * panel_width * gridDim.y;
  const unsigned int panel_idx = block_idx / panel_size;
  const unsigned int panel_offset = block_idx % panel_size;

  unsigned int mini_x;
  unsigned int row_idx, col_idx;
  if (block_idx >= total_panel) {
    row_idx = panel_offset / res_panel_width;
    mini_x = panel_offset % res_panel_width;
    mini_x = (row_idx & 1) ? res_panel_width - 1 - mini_x : mini_x;
  } else {
    row_idx = panel_offset / panel_width;
    mini_x = panel_offset % panel_width;
    mini_x = (row_idx & 1) ? panel_width - 1 - mini_x : mini_x;
  }

  if (panel_idx & 1) {
    row_idx = gridDim.y - 1 - row_idx;
  }
  col_idx = panel_idx * panel_width + mini_x;

  return {col_idx, row_idx, blockIdx.z};
}

} // namespace tl
