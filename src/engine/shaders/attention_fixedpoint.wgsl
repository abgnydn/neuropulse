// CONTINUOUS-ATTENTION FIXED-POINT (DECODE) — Picard iteration of attention.wgsl.
//
// SUB-STEP PROBE: this iterates Q within a single attention call (per-attention FP).
// NOT equivalent to DEQ-style per-block fixed-point (h_{t+1} = Block(h_t)).
// Per-block FP is a strictly larger experiment, deferred.
//
// RoPE applied externally once at iter=0; Q iterates POST-RoPE. The alternative
// (re-apply RoPE each iter, treating Q's position as drifting) is semantically
// odd but mathematically valid; this experiment does NOT test that interpretation.
//
// The fixed point of this map is the SELF-CONSISTENCY POINT
//   Q* = softmax(Q* K^T / sqrt(d)) V
// NOT the energy minimum of Ramsauer's E. See PREDICTIONS.md P-20260526-07
// and brain experiment E45 for the symbol-collision discussion.
//
// Algorithm — per (batch, head):
//   1. Load initial Q from buffer into registers (q0, q1, q2 per thread; 3*32=96=head_dim)
//   2. For iter in 0..max_iter:
//        a. Save (q0, q1, q2) into (q0_prev, q1_prev, q2_prev) per thread
//        b. Run online-softmax pass over all KV pages with current (q0, q1, q2)
//           — produces (o0, o1, o2) per thread, the post-attention output
//           — K, V cache untouched; this kernel never writes pages or length_info
//        c. Picard update: (q0, q1, q2) = (o0, o1, o2)
//        d. Last-iter only: compute per-thread max(|o - q_prev|), tree-reduce across
//           workgroup to get layer-head-level convergence delta for telemetry
//   3. Write final (q0, q1, q2) into output_buf
//   4. Thread 0 writes telemetry: { final_diff_inf, iter_count, max_score, min_score }
//
// Telemetry buffer layout: telemetry[(layer * 32 + head) * 4 + slot] f32, where
// the per-layer base offset is supplied by the host via podArgs.telem_offset_words
// (= layer * 32 * 4). Same pattern as attention_scores.wgsl's layer_offset_words.
// The 4-f32 record per (layer, head) is:
//   [0] = ||Q_t - Q_{t-1}||_inf at the final iter (workgroup max across head_dim)
//   [1] = iter_count (always max_iter in Phase 1; meaningful when early-exit added)
//   [2] = max raw softmax score across all KV slots at iter 0 (numerical-stability sanity)
//   [3] = min raw softmax score across all KV slots at iter 0
//
// Pages layout (same as attention.wgsl):
//   pages[page * 98304 + head * 1536 + slot * 96 + dim]
//   K at offset 0, V at offset 49152

enable f16;

@group(0) @binding(0) var<storage, read> Q : array<f16>;
@group(0) @binding(1) var<storage, read> page_table_indptr : array<i32>;
@group(0) @binding(2) var<storage, read> page_table_values : array<i32>;
@group(0) @binding(3) var<storage, read> pages : array<f16>;
@group(0) @binding(4) var<storage, read> length_info : array<i32>;
@group(0) @binding(5) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(6) var<storage, read_write> telemetry : array<f32>;

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
  max_iter: i32,           // Phase 1: 100. Phase 2: pin smaller after seeing convergence shape.
  telem_offset_words: i32  // = layer * 32 * 4. Host-supplied so the kernel stays layer-agnostic.
}
@group(0) @binding(7) var<uniform> podArgs : PODArgs;

var<workgroup> score_reduce : array<f32, 32>;
var<workgroup> initial_max_score : f32;
var<workgroup> initial_min_score : f32;

@compute @workgroup_size(32, 1, 1)
fn attention_fixedpoint(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let batch : i32 = i32(blockIdx.x);
  let head : i32 = i32(blockIdx.y);
  let tid : i32 = i32(threadIdx.x);

  if (batch >= podArgs.B) { return; }

  // Initial Q load (post-RoPE; RoPE was applied by a prior dispatch).
  // Each thread owns 3 elements of Q (32 threads × 3 = 96 = head_dim).
  var q0 : f32 = f32(Q[batch * 3072 + head * 96 + tid * 3]);
  var q1 : f32 = f32(Q[batch * 3072 + head * 96 + tid * 3 + 1]);
  var q2 : f32 = f32(Q[batch * 3072 + head * 96 + tid * 3 + 2]);

  // Previous-iter Q (for convergence delta at the final iter).
  var q0_prev : f32 = 0.0;
  var q1_prev : f32 = 0.0;
  var q2_prev : f32 = 0.0;

  // Final convergence delta (||Q_t - Q_{t-1}||_inf at the last iter), reduced
  // across the workgroup. Written to telemetry by thread 0 at end.
  var final_diff : f32 = 0.0;

  // Page range for this batch. Constant across iters — K, V are NOT iterated.
  let indptr_begin : i32 = page_table_indptr[batch + podArgs.page_indptr_elem_offset];
  let indptr_end : i32 = page_table_indptr[batch + podArgs.page_indptr_elem_offset + 1];
  let kv_len : i32 = length_info[batch + podArgs.length_info_elem_offset];

  // Initialize iter-0 entropy probe (set by tid 0 during iter 0)
  if (tid == 0) {
    initial_max_score = -50000.0;
    initial_min_score = 50000.0;
  }
  workgroupBarrier();

  // ────────────────────────────────────────────────────────────────────────
  // PICARD LOOP — outer iteration over the online-softmax operator
  // ────────────────────────────────────────────────────────────────────────
  for (var iter : i32 = 0; iter < podArgs.max_iter; iter = iter + 1) {

    // Save previous Q for convergence diff on this iter
    q0_prev = q0;
    q1_prev = q1;
    q2_prev = q2;

    // Reset online-softmax state for this iter
    var m : f32 = -50000.0;
    var d : f32 = 0.0;
    var o0 : f32 = 0.0;
    var o1 : f32 = 0.0;
    var o2 : f32 = 0.0;

    // ── Inner: one pass of online softmax over all KV positions ──
    // Identical structure to attention.wgsl — same kernel body, repeated.
    for (var page_idx : i32 = indptr_begin; page_idx < indptr_end; page_idx = page_idx + 1) {
      let page_no : i32 = page_table_values[page_idx + podArgs.page_values_elem_offset];
      let page_start : i32 = (page_idx - indptr_begin) * 16;
      let slots_in_page : i32 = min(16, kv_len - page_start);

      for (var slot : i32 = 0; slot < slots_in_page; slot = slot + 1) {
        let k_base : i32 = page_no * 98304 + head * 1536 + slot * 96 + podArgs.pages_elem_offset;

        // Partial dot product: each thread computes 3 multiplies
        let partial : f32 = q0 * f32(pages[k_base + tid * 3])
                          + q1 * f32(pages[k_base + tid * 3 + 1])
                          + q2 * f32(pages[k_base + tid * 3 + 2]);

        // Tree reduction across 32 threads to get full dot product
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

        let s : f32 = score_reduce[0] * podArgs.sm_scale;

        // Iter-0 numerical-stability sanity: track max/min raw softmax score
        // across all KV slots in this token. Telemetry only — does not affect math.
        if (iter == 0 && tid == 0) {
          initial_max_score = max(initial_max_score, s);
          initial_min_score = min(initial_min_score, s);
        }

        // Online softmax update
        let m_prev : f32 = m;
        m = max(m, s);
        let scale_prev : f32 = exp(m_prev - m);
        let scale_new : f32 = exp(s - m);

        d = d * scale_prev + scale_new;

        // Load V and accumulate
        let v_base : i32 = k_base + 49152;
        o0 = o0 * scale_prev + scale_new * f32(pages[v_base + tid * 3]);
        o1 = o1 * scale_prev + scale_new * f32(pages[v_base + tid * 3 + 1]);
        o2 = o2 * scale_prev + scale_new * f32(pages[v_base + tid * 3 + 2]);
      }
    }

    // Normalize this iter's output
    if (d > 0.0) {
      let inv_d : f32 = 1.0 / d;
      o0 = o0 * inv_d;
      o1 = o1 * inv_d;
      o2 = o2 * inv_d;
    }

    // Picard update: Q ← O
    q0 = o0;
    q1 = o1;
    q2 = o2;

    // Convergence delta — only meaningful on the final iter. Computed here
    // so the workgroup state is fresh; written to telemetry after the loop.
    if (iter == podArgs.max_iter - 1) {
      let local_diff : f32 = max(max(abs(q0 - q0_prev), abs(q1 - q1_prev)), abs(q2 - q2_prev));
      score_reduce[tid] = local_diff;
      workgroupBarrier();
      if (tid < 16) { score_reduce[tid] = max(score_reduce[tid], score_reduce[tid + 16]); }
      workgroupBarrier();
      if (tid < 8) { score_reduce[tid] = max(score_reduce[tid], score_reduce[tid + 8]); }
      workgroupBarrier();
      if (tid < 4) { score_reduce[tid] = max(score_reduce[tid], score_reduce[tid + 4]); }
      workgroupBarrier();
      if (tid < 2) { score_reduce[tid] = max(score_reduce[tid], score_reduce[tid + 2]); }
      workgroupBarrier();
      if (tid < 1) { score_reduce[tid] = max(score_reduce[tid], score_reduce[tid + 1]); }
      workgroupBarrier();
      final_diff = score_reduce[0];
    }
  }
  // ────────────────────────────────────────────────────────────────────────

  // Write final Q to output_buf — this is the per-head per-batch attention
  // output that the O-proj matmul will read. Layout matches attention.wgsl.
  output_buf[batch * 3072 + head * 96 + tid * 3]     = f16(q0);
  output_buf[batch * 3072 + head * 96 + tid * 3 + 1] = f16(q1);
  output_buf[batch * 3072 + head * 96 + tid * 3 + 2] = f16(q2);

  // Telemetry write — thread 0 only. 4 f32 slots per (layer, head). Layer
  // offset comes from the host (= layer * 32 * 4); the kernel itself stays
  // layer-agnostic. Batch index assumed 1 for decode (B=1).
  if (tid == 0) {
    let telem_base : i32 = podArgs.telem_offset_words + head * 4;
    telemetry[telem_base]     = final_diff;
    telemetry[telem_base + 1] = f32(podArgs.max_iter);  // iter_count (always max_iter in Phase 1)
    telemetry[telem_base + 2] = initial_max_score;
    telemetry[telem_base + 3] = initial_min_score;
  }
}
