// ATTENTION SCORES (visualization-only).
//
// Computes post-softmax attention scores for one (batch, head) pair, writing
// into the layer's slice of a shared per-token scores buffer. This produces the
// data needed to draw real "what the model is looking at" beams from the current
// token back to source positions, for ALL 32 layers in one token.
//
// Output buffer layout (for the whole token):
//   scores_out[layer * 32 * 256 + head * 256 + slot] f32
//   The shader writes only into one layer's slice, picked by layer_offset_words.
//   layer_offset_words = layer * (32 * 256)
//
// Pages layout (same as attention.wgsl):
//   pages[page * 98304 + head * 1536 + slot * 96 + dim]
//   K at offset 0, V at offset 49152

enable f16;

const MAX_SCORE_SLOTS : i32 = 256;
const HEADS_PER_LAYER : i32 = 32;
const WORDS_PER_LAYER : i32 = 32 * 256; // 8192

@group(0) @binding(0) var<storage, read> Q : array<f16>;
@group(0) @binding(1) var<storage, read> page_table_indptr : array<i32>;
@group(0) @binding(2) var<storage, read> page_table_values : array<i32>;
@group(0) @binding(3) var<storage, read> pages : array<f16>;
@group(0) @binding(4) var<storage, read> length_info : array<i32>;
@group(0) @binding(5) var<storage, read_write> scores_out : array<f32>;

struct PODArgs {
  B: i32,
  max_num_pages: i32,
  nnz_pages: i32,
  pages_elem_offset: i32,
  page_indptr_elem_offset: i32,
  page_values_elem_offset: i32,
  length_info_elem_offset: i32,
  sm_scale: f32,
  packGridDimX: u32,
  layer_offset_words: i32  // = layer * 32 * 256
}
@group(0) @binding(6) var<uniform> podArgs : PODArgs;

var<workgroup> score_reduce : array<f32, 32>;
var<workgroup> raw_scores : array<f32, MAX_SCORE_SLOTS>;

@compute @workgroup_size(32, 1, 1)
fn attention_scores(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let batch : i32 = i32(blockIdx.x);
  let head : i32 = i32(blockIdx.y);
  let tid : i32 = i32(threadIdx.x);

  if (batch >= podArgs.B) { return; }

  // Each thread owns 3 elements of Q
  let q0 : f32 = f32(Q[batch * 3072 + head * 96 + tid * 3]);
  let q1 : f32 = f32(Q[batch * 3072 + head * 96 + tid * 3 + 1]);
  let q2 : f32 = f32(Q[batch * 3072 + head * 96 + tid * 3 + 2]);

  let indptr_begin : i32 = page_table_indptr[batch + podArgs.page_indptr_elem_offset];
  let indptr_end : i32 = page_table_indptr[batch + podArgs.page_indptr_elem_offset + 1];
  let kv_len : i32 = length_info[batch + podArgs.length_info_elem_offset];

  // Pass 1: compute raw dot products into raw_scores[]
  var slot_global : i32 = 0;
  for (var page_idx : i32 = indptr_begin; page_idx < indptr_end; page_idx = page_idx + 1) {
    let page_no : i32 = page_table_values[page_idx + podArgs.page_values_elem_offset];
    let page_start : i32 = (page_idx - indptr_begin) * 16;
    let slots_in_page : i32 = min(16, kv_len - page_start);

    for (var slot : i32 = 0; slot < slots_in_page; slot = slot + 1) {
      let k_base : i32 = page_no * 98304 + head * 1536 + slot * 96 + podArgs.pages_elem_offset;

      let partial : f32 = q0 * f32(pages[k_base + tid * 3])
                        + q1 * f32(pages[k_base + tid * 3 + 1])
                        + q2 * f32(pages[k_base + tid * 3 + 2]);

      score_reduce[tid] = partial;
      workgroupBarrier();
      if (tid < 16) { score_reduce[tid] = score_reduce[tid] + score_reduce[tid + 16]; }
      workgroupBarrier();
      if (tid < 8) { score_reduce[tid] = score_reduce[tid] + score_reduce[tid + 8]; }
      workgroupBarrier();
      if (tid < 4) { score_reduce[tid] = score_reduce[tid] + score_reduce[tid + 4]; }
      workgroupBarrier();
      if (tid < 2) { score_reduce[tid] = score_reduce[tid] + score_reduce[tid + 2]; }
      workgroupBarrier();
      if (tid < 1) { score_reduce[tid] = score_reduce[tid] + score_reduce[tid + 1]; }
      workgroupBarrier();

      if (tid == 0 && slot_global < MAX_SCORE_SLOTS) {
        raw_scores[slot_global] = score_reduce[0] * podArgs.sm_scale;
      }
      slot_global = slot_global + 1;
      workgroupBarrier();
    }
  }

  // Pass 2: softmax — done by tid 0
  if (tid == 0) {
    let n : i32 = min(slot_global, MAX_SCORE_SLOTS);
    var m : f32 = -50000.0;
    for (var i : i32 = 0; i < n; i = i + 1) {
      if (raw_scores[i] > m) { m = raw_scores[i]; }
    }
    var sum : f32 = 0.0;
    for (var i : i32 = 0; i < n; i = i + 1) {
      let e : f32 = exp(raw_scores[i] - m);
      raw_scores[i] = e;
      sum = sum + e;
    }
    let inv_sum : f32 = select(0.0, 1.0 / sum, sum > 0.0);
    let base : i32 = podArgs.layer_offset_words + head * MAX_SCORE_SLOTS;
    for (var i : i32 = 0; i < MAX_SCORE_SLOTS; i = i + 1) {
      if (i < n) {
        scores_out[base + i] = raw_scores[i] * inv_sum;
      } else {
        scores_out[base + i] = 0.0;
      }
    }
  }
}
