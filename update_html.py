#!/usr/bin/env python3
"""
Script to update the HTML file with expanded CUDA Primer and Task 5 sections.
"""

import re

def escape_html(text):
    """Escape HTML special characters."""
    return (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;'))

def format_text(text, bold=False, code=False):
    """Format text with appropriate HTML spans."""
    if code:
        return f'<span class="c8 c3">{escape_html(text)}</span>'
    elif bold:
        return f'<span class="c11">{text}</span>'
    else:
        return f'<span class="c0">{text}</span>'

def create_paragraph(text, class_name="c10"):
    """Create a paragraph element."""
    return f'<p class="{class_name}"><span class="c0">{text}</span></p>'

def create_heading(text, level=2, id_attr=None):
    """Create a heading element."""
    if level == 2:
        class_name = "c7"
        span_class = "c2 c18 c11"
    elif level == 3:
        class_name = "c17"
        span_class = "c2 c13 c11"
    else:
        class_name = "c17"
        span_class = "c2 c13 c11"
    
    id_attr = f' id="{id_attr}"' if id_attr else ''
    return f'<h{level} class="{class_name}"{id_attr}><span class="{span_class}">{text}</span></h{level}>'

def create_list_item(text, indent=0):
    """Create a list item."""
    if indent == 0:
        return f'<li class="c5 li-bullet-0"><span class="c0">{text}</span></li>'
    else:
        return f'<li class="c10 c21 li-bullet-0"><span class="c0">{text}</span></li>'

def create_code_block(code, language="c"):
    """Create a code block (simplified - will use pre/code tags)."""
    escaped = escape_html(code)
    return f'<p class="c4"><span class="c1 c3">{escaped}</span></p>'

# Read the HTML file
with open('webpage/webpage.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Build the new CUDA Primer section
cuda_primer_html = '''<h2 class="c7" id="h.86rlcz2pvytg"><span class="c2 c18 c11">CUDA Primer (read first)</span></h2>
<h3 class="c17"><span class="c2 c13 c11">Core Concepts</span></h3>
<ul class="c15 lst-kix_n50pa8c09fq6-0 start">
<li class="c5 li-bullet-0"><span>A </span><span class="c11">kernel</span><span>&nbsp;is a C/C++ function annotated </span><span class="c8 c3">__global__</span><span>&nbsp;and launched with </span><span class="c8 c3">&lt;&lt;&lt;grid, block&gt;&gt;&gt;</span><span class="c0">.</span></li>
<li class="c5 li-bullet-0"><span>Each launch creates a 2&#8209;D/3&#8209;D </span><span class="c11">grid</span><span>&nbsp;of </span><span class="c11">blocks</span><span>; each block contains many </span><span class="c11">threads</span><span class="c0">. Every thread runs the same kernel on different data.</span></li>
</ul>
<p class="c10"><span class="c0"><span class="c11">Example:</span> </span><span class="c8 c3">myKernel&lt;&lt;&lt;dim3(2,2), dim3(16,16)&gt;&gt;&gt;(args)</span><span class="c0">&nbsp;creates a 2×2 grid of blocks, each with 16×16 threads (total: 4 blocks × 256 threads = 1024 threads).</span></p>
<ul class="c15 lst-kix_n50pa8c09fq6-0">
<li class="c5 li-bullet-0"><span>Built&#8209;in variables inside kernels: </span><span class="c8 c3">blockIdx.{x,y,z}</span><span>, </span><span class="c8 c3">threadIdx.{x,y,z}</span><span>, </span><span class="c8 c3">blockDim.{x,y,z}</span><span>, </span><span class="c8 c3">gridDim.{x,y,z}</span><span class="c0">.</span></li>
</ul>
<p class="c10"><span class="c11">2D Indexing Example:</span></p>
<p class="c4"><span class="c1 c3">int i = blockIdx.y * blockDim.y + threadIdx.y;  // row</span></p>
<p class="c4"><span class="c1 c3">int j = blockIdx.x * blockDim.x + threadIdx.x;  // column</span></p>
<p class="c10"><span class="c11">Grid size calculation:</span><span class="c0">&nbsp;When dimensions don&rsquo;t divide evenly, round up:</span></p>
<p class="c4"><span class="c1 c3">dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);</span></p>
<h3 class="c17"><span class="c2 c13 c11">Memory Hierarchy</span></h3>
<ul class="c15 lst-kix_n50pa8c09fq6-0">
<li class="c5 li-bullet-0"><span class="c11">Global</span><span class="c0">&nbsp;(device DRAM): large, high&#8209;latency (~400-800 cycles); visible to all threads. Allocated with </span><span class="c8 c3">cudaMalloc()</span><span class="c0">, accessed via pointers. Persists across kernel launches.</span></li>
<li class="c5 li-bullet-0"><span class="c11">Shared</span><span class="c0">&nbsp;(on&#8209;chip, per&#8209;block): small (~48KB per SM), low&#8209;latency (~20-30 cycles); visible to threads in the same block. Declared with </span><span class="c8 c3">__shared__</span><span class="c0">. Cleared between kernel launches.</span></li>
<li class="c5 li-bullet-0"><span class="c11">Registers</span><span class="c0">&nbsp;(per&#8209;thread): fastest (~1 cycle), limited (~255 per thread). Automatic for local variables. Private to each thread.</span></li>
</ul>
<p class="c10"><span class="c11">Memory transfer patterns:</span></p>
<ul class="c15 lst-kix_n50pa8c09fq6-0">
<li class="c5 li-bullet-0"><span>Host → Device: </span><span class="c8 c3">cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice)</span></li>
<li class="c5 li-bullet-0"><span>Device → Host: </span><span class="c8 c3">cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost)</span></li>
<li class="c5 li-bullet-0"><span>Device → Device: </span><span class="c8 c3">cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice)</span></li>
</ul>
<h3 class="c17"><span class="c2 c13 c11">Performance Concepts</span></h3>
<ul class="c15 lst-kix_n50pa8c09fq6-0">
<li class="c5 li-bullet-0"><span class="c11">Memory coalescing:</span><span class="c0">&nbsp;threads in a warp (32 threads) should access consecutive addresses to combine loads/stores into few transactions.</span></li>
</ul>
<ul class="c15 lst-kix_n50pa8c09fq6-1 start">
<li class="c10 c21 li-bullet-0"><span>✅ </span><span class="c11">Good:</span><span class="c0">&nbsp;</span><span class="c8 c3">thread 0 → addr[0], thread 1 → addr[1], ..., thread 31 → addr[31]</span><span class="c0">&nbsp;(coalesced into 1-2 transactions)</span></li>
<li class="c10 c21 li-bullet-0"><span>❌ </span><span class="c11">Bad:</span><span class="c0">&nbsp;</span><span class="c8 c3">thread 0 → addr[0], thread 1 → addr[100], thread 2 → addr[200]...</span><span class="c0">&nbsp;(32 separate transactions)</span></li>
</ul>
<ul class="c15 lst-kix_n50pa8c09fq6-0">
<li class="c5 li-bullet-0"><span class="c11">Warp:</span><span class="c0">&nbsp;A group of 32 threads that execute in lockstep (SIMT - Single Instruction, Multiple Threads). Threads in a warp share the same instruction stream. Warps are the fundamental unit of execution on the GPU.</span></li>
<li class="c5 li-bullet-0"><span class="c11">Synchronization:</span><span>&nbsp;</span><span class="c8 c3">__syncthreads()</span><span class="c0">&nbsp;is a barrier for all threads in a block (not across blocks). Use before reading shared memory written by other threads. </span><span class="c11">Important:</span><span class="c0">&nbsp;All threads in a block must reach the barrier, or you&rsquo;ll deadlock.</span></li>
<li class="c5 li-bullet-0"><span class="c11">Occupancy:</span><span class="c0">&nbsp;The ratio of active warps to maximum warps per SM. Higher occupancy doesn&rsquo;t always mean better performance (register pressure, shared memory limits).</span></li>
</ul>
<h3 class="c17"><span class="c2 c13 c11">Common Patterns &amp; Best Practices</span></h3>
<p class="c10"><span class="c0">You&rsquo;ll always (1) map data to threads, (2) ensure bounds checks, (3) choose grid/block sizes, and (4) verify results.</span></p>
<p class="c10"><span class="c11">Bounds checking pattern:</span></p>
<p class="c4"><span class="c1 c3">int idx = blockIdx.x * blockDim.x + threadIdx.x;</span></p>
<p class="c4"><span class="c1 c3">if (idx &gt;= N) return;  // Guard against out-of-bounds</span></p>
<p class="c10"><span class="c11">Block size guidelines:</span></p>
<ul class="c15 lst-kix_n50pa8c09fq6-0">
<li class="c5 li-bullet-0"><span>Use multiples of 32 (warp size): 32, 64, 128, 256, 512, 1024</span></li>
<li class="c5 li-bullet-0"><span>For 2D: Common sizes are 8×8, 16×16, 32×32</span></li>
<li class="c5 li-bullet-0"><span>Balance: Larger blocks = more shared memory per block, but fewer blocks = less parallelism</span></li>
</ul>
<h3 class="c17"><span class="c2 c13 c11">Common Pitfalls</span></h3>
<ul class="c15 lst-kix_n50pa8c09fq6-0">
<li class="c5 li-bullet-0"><span class="c11">Forgetting bounds checks:</span><span class="c0">&nbsp;Always check </span><span class="c8 c3">if (i &gt;= rows || j &gt;= cols) return;</span><span class="c0">&nbsp;when grid size rounds up.</span></li>
<li class="c5 li-bullet-0"><span class="c11">Wrong indexing order:</span><span class="c0">&nbsp;Use </span><span class="c8 c3">threadIdx.x</span><span class="c0">&nbsp;for columns (fastest dimension) to enable coalescing in row-major layouts.</span></li>
<li class="c5 li-bullet-0"><span class="c11">Missing synchronization:</span><span class="c0">&nbsp;If threads write then read shared memory, use </span><span class="c8 c3">__syncthreads()</span><span class="c0">&nbsp;between.</span></li>
<li class="c5 li-bullet-0"><span class="c11">Not checking errors:</span><span class="c0">&nbsp;Always call </span><span class="c8 c3">checkCuda(cudaGetLastError())</span><span class="c0">&nbsp;after kernel launches.</span></li>
<li class="c5 li-bullet-0"><span class="c11">Race conditions:</span><span class="c0">&nbsp;Multiple threads writing to the same global memory location without atomics.</span></li>
<li class="c5 li-bullet-0"><span class="c11">Register spilling:</span><span class="c0">&nbsp;Too many local variables can cause register spilling to local memory (slow).</span></li>
</ul>
<h3 class="c17"><span class="c2 c13 c11">Debugging Tips</span></h3>
<ul class="c15 lst-kix_n50pa8c09fq6-0">
<li class="c5 li-bullet-0"><span class="c11">Start small:</span><span class="c0">&nbsp;Test with 8×8 matrices before scaling up.</span></li>
<li class="c5 li-bullet-0"><span>Use </span><span class="c8 c3">printf</span><span>&nbsp;in kernels (sparingly) for small sizes: </span><span class="c8 c3">if (threadIdx.x == 0 &amp;&amp; blockIdx.x == 0) printf("value: %f\n", x);</span></li>
<li class="c5 li-bullet-0"><span class="c11">Run cuda-memcheck:</span><span>&nbsp;</span><span class="c8 c3">cuda-memcheck ./your_program</span><span class="c0">&nbsp;to catch memory errors, race conditions, and invalid accesses.</span></li>
<li class="c5 li-bullet-0"><span class="c11">Check occupancy:</span><span class="c0">&nbsp;Use </span><span class="c8 c3">nvidia-smi</span><span class="c0">&nbsp;to monitor GPU usage, or profiling tools if available.</span></li>
<li class="c5 li-bullet-0"><span class="c11">Synchronize before copying:</span><span>&nbsp;Use </span><span class="c8 c3">cudaDeviceSynchronize()</span><span class="c0">&nbsp;before copying results back to host.</span></li>
<li class="c5 li-bullet-0"><span class="c11">Verify with small inputs:</span><span class="c0">&nbsp;Always test correctness on small, known inputs first.</span></li>
</ul>
<h3 class="c17"><span class="c2 c13 c11">CUDA Documentation &amp; Finding Functions</span></h3>
<p class="c10"><span class="c11">You will need to look up CUDA functions in the official documentation.</span><span class="c0">&nbsp;For each function you use:</span></p>
<ul class="c15 lst-kix_n50pa8c09fq6-0">
<li class="c5 li-bullet-0"><span class="c11">If the function has its own page</span><span class="c0">&nbsp;(e.g., </span><span class="c8 c3">cudaMalloc</span><span>, </span><span class="c8 c3">cudaMemcpy</span><span class="c0">), link directly to that page.</span></li>
<li class="c5 li-bullet-0"><span class="c11">If the function is part of a larger API section</span><span class="c0">&nbsp;(e.g., vector types like </span><span class="c8 c3">float4</span><span class="c0">), link to the relevant section of the programming guide.</span></li>
</ul>
<p class="c10"><span class="c11">Key Documentation Resources:</span></p>
<ul class="c15 lst-kix_n50pa8c09fq6-0">
<li class="c5 li-bullet-0"><span class="c11">CUDA C++ Programming Guide</span><span>&nbsp;</span><a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/"><span class="c0">https://docs.nvidia.com/cuda/cuda-c-programming-guide/</span></a><span class="c0">&nbsp;- Complete reference for CUDA programming</span></li>
</ul>
<ul class="c15 lst-kix_n50pa8c09fq6-1 start">
<li class="c10 c21 li-bullet-0"><a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy"><span class="c0">Memory Hierarchy</span></a><span class="c0">&nbsp;- Detailed memory model</span></li>
<li class="c10 c21 li-bullet-0"><a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-variables"><span class="c0">Built-in Variables</span></a><span class="c0">&nbsp;- </span><span class="c8 c3">blockIdx</span><span>, </span><span class="c8 c3">threadIdx</span><span class="c0">, etc.</span></li>
<li class="c10 c21 li-bullet-0"><a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#vector-types"><span class="c0">Vector Types</span></a><span class="c0">&nbsp;- </span><span class="c8 c3">float4</span><span>, </span><span class="c8 c3">int4</span><span class="c0">, etc.</span></li>
<li class="c10 c21 li-bullet-0"><a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions"><span class="c0">Synchronization Functions</span></a><span class="c0">&nbsp;- </span><span class="c8 c3">__syncthreads()</span><span class="c0">, etc.</span></li>
</ul>
<ul class="c15 lst-kix_n50pa8c09fq6-0">
<li class="c5 li-bullet-0"><span class="c11">CUDA Runtime API</span><span>&nbsp;</span><a href="https://docs.nvidia.com/cuda/cuda-runtime-api/"><span class="c0">https://docs.nvidia.com/cuda/cuda-runtime-api/</span></a><span class="c0">&nbsp;- Host-side functions</span></li>
</ul>
<ul class="c15 lst-kix_n50pa8c09fq6-1 start">
<li class="c10 c21 li-bullet-0"><a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html"><span class="c0">Memory Management</span></a><span class="c0">&nbsp;- </span><span class="c8 c3">cudaMalloc</span><span>, </span><span class="c8 c3">cudaFree</span><span>, </span><span class="c8 c3">cudaMemcpy</span></li>
<li class="c10 c21 li-bullet-0"><a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html"><span class="c0">Device Management</span></a><span class="c0">&nbsp;- </span><span class="c8 c3">cudaDeviceSynchronize</span><span>, </span><span class="c8 c3">cudaGetLastError</span></li>
</ul>
<ul class="c15 lst-kix_n50pa8c09fq6-0">
<li class="c5 li-bullet-0"><a href="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/"><span class="c11">CUDA Best Practices Guide</span></a><span class="c0">&nbsp;- Performance optimization tips</span></li>
</ul>
<p class="c10"><span class="c11">Example:</span><span class="c0">&nbsp;If you use </span><span class="c8 c3">cudaMalloc</span><span>, link to: </span><a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356"><span class="c0">https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356</span></a></p>
<p class="c10"><span class="c0">If you use </span><span class="c8 c3">float4</span><span>, link to: </span><a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#vector-types"><span class="c0">https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#vector-types</span></a></p>
<h3 class="c17"><span class="c2 c13 c11">Quick Reference</span></h3>
<table style="border-collapse: collapse; width: 100%; margin: 12pt 0; border: 1px solid #000000;">
<thead>
<tr style="background-color: #f0f0f0;">
<th style="border: 1px solid #000000; padding: 8pt 12pt; text-align: left; font-weight: 700; font-family: Arial; font-size: 11pt;"><span class="c0 c11">Concept</span></th>
<th style="border: 1px solid #000000; padding: 8pt 12pt; text-align: left; font-weight: 700; font-family: Arial; font-size: 11pt;"><span class="c0 c11">Syntax</span></th>
<th style="border: 1px solid #000000; padding: 8pt 12pt; text-align: left; font-weight: 700; font-family: Arial; font-size: 11pt;"><span class="c0 c11">Notes</span></th>
</tr>
</thead>
<tbody>
<tr>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c0">Kernel launch</span></td>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c8 c3">kernel&lt;&lt;&lt;grid, block&gt;&gt;&gt;(args)</span></td>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c0">Grid/block are </span><span class="c8 c3">dim3</span></td>
</tr>
<tr>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c0">Global memory alloc</span></td>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c8 c3">cudaMalloc(&amp;ptr, size)</span></td>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c0">Returns device pointer</span></td>
</tr>
<tr>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c0">Copy to device</span></td>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c8 c3">cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice)</span></td>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c0"></span></td>
</tr>
<tr>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c0">Copy from device</span></td>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c8 c3">cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost)</span></td>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c0"></span></td>
</tr>
<tr>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c0">Shared memory</span></td>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c8 c3">__shared__ float sdata[256]</span></td>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c0">Per-block, ~48KB limit</span></td>
</tr>
<tr>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c0">Synchronization</span></td>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c8 c3">__syncthreads()</span></td>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c0">Block-level barrier</span></td>
</tr>
<tr>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c0">Error check</span></td>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c8 c3">checkCuda(cudaGetLastError())</span></td>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c0">After kernel launch</span></td>
</tr>
<tr>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c0">Device sync</span></td>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c8 c3">cudaDeviceSynchronize()</span></td>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c0">Wait for kernel completion</span></td>
</tr>
<tr>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c0">Vector load</span></td>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c8 c3">float4 vec = *reinterpret_cast&lt;float4*&gt;(&amp;ptr[i])</span></td>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c0">Aligned access</span></td>
</tr>
<tr>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c0">Vector store</span></td>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c8 c3">*reinterpret_cast&lt;float4*&gt;(&amp;ptr[i]) = vec</span></td>
<td style="border: 1px solid #000000; padding: 8pt 12pt; font-family: Arial; font-size: 11pt;"><span class="c0">Aligned access</span></td>
</tr>
</tbody>
</table>
<p class="c10 c6"><span class="c0"></span></p>'''

# Build the new Task 5 section
task5_html = '''<h2 class="c7" id="h.nluym1yc8qbd"><span class="c2 c18 c11">Task 5 &mdash; (Optional) Performance Engineering (15 EC)</span></h2>
<p class="c10"><span>Make it </span><span class="c11">faster</span><span class="c0">. Ideas:</span></p>
<ul class="c15 lst-kix_pm62bnpnrfb1-0 start">
<li class="c5 li-bullet-0"><span class="c11">Shared&#8209;memory tiling</span><span class="c0">&nbsp;(classic 16×16 or 32×32 tiles).</span></li>
<li class="c5 li-bullet-0"><span class="c11">Register tiling</span><span>: each thread computes a small </span><span class="c8 c3">r×c</span><span class="c0">&nbsp;tile.</span></li>
<li class="c5 li-bullet-0"><span class="c11">Loop unrolling &amp; </span><span class="c3 c11 c23">#pragma unroll</span><span>&nbsp;for </span><span class="c8 c3">k</span><span class="c0">.</span></li>
<li class="c5 li-bullet-0"><span class="c11">Occupancy tuning</span><span>: vary </span><span class="c8 c3">blockDim</span><span class="c0">&nbsp;to trade registers vs parallelism.</span></li>
<li class="c5 li-bullet-0"><span class="c11">Mixed precision</span><span>: keep FP32 accumulate but try </span><span class="c8 c3">__half</span><span class="c0">&nbsp;inputs (only if you also keep a FP32 correctness path for grading).</span></li>
<li class="c5 li-bullet-0"><span class="c11">Software prefetch</span><span>: read the next </span><span class="c8 c3">k</span><span class="c0">&nbsp;slice early.</span></li>
</ul>
<h3 class="c17"><span class="c2 c13 c11">Performance Targets &amp; Grading</span></h3>
<p class="c10"><span>We&rsquo;ll measure performance in </span><span class="c11">GFLOP/s</span><span class="c0">&nbsp;(billions of floating-point operations per second) on a standard GPU. The formula is: </span><span class="c8 c3">GFLOP/s = (2 × M × N × K) / (time_in_seconds × 1e9)</span><span class="c0">.</span></p>
<p class="c10"><span class="c11">Grading Rubric:</span></p>
<ul class="c15 lst-kix_pm62bnpnrfb1-0">
<li class="c5 li-bullet-0"><span class="c11">Full credit (15 pts):</span><span>&nbsp;≥ 2.0× speedup over Task 4 baseline</span></li>
</ul>
<ul class="c15 lst-kix_pm62bnpnrfb1-1 start">
<li class="c10 c21 li-bullet-0"><span>Example: If Task 4 runs at 500 GFLOP/s, you need ≥ 1000 GFLOP/s</span></li>
<li class="c10 c21 li-bullet-0"><span>Must maintain correctness (within 1e-4 tolerance)</span></li>
<li class="c10 c21 li-bullet-0"><span>Must include analysis in README-perf.md</span></li>
</ul>
<ul class="c15 lst-kix_pm62bnpnrfb1-0">
<li class="c5 li-bullet-0"><span class="c11">Partial credit (10 pts):</span><span>&nbsp;1.5× - 2.0× speedup over Task 4 baseline</span></li>
</ul>
<ul class="c15 lst-kix_pm62bnpnrfb1-1 start">
<li class="c10 c21 li-bullet-0"><span>Example: If Task 4 runs at 500 GFLOP/s, you need 750-1000 GFLOP/s</span></li>
<li class="c10 c21 li-bullet-0"><span>Must maintain correctness</span></li>
<li class="c10 c21 li-bullet-0"><span>Must include analysis in README-perf.md</span></li>
</ul>
<ul class="c15 lst-kix_pm62bnpnrfb1-0">
<li class="c5 li-bullet-0"><span class="c11">Minimal credit (5 pts):</span><span>&nbsp;1.2× - 1.5× speedup over Task 4 baseline</span></li>
</ul>
<ul class="c15 lst-kix_pm62bnpnrfb1-1 start">
<li class="c10 c21 li-bullet-0"><span>Example: If Task 4 runs at 500 GFLOP/s, you need 600-750 GFLOP/s</span></li>
<li class="c10 c21 li-bullet-0"><span>Must maintain correctness</span></li>
<li class="c10 c21 li-bullet-0"><span>Must include analysis in README-perf.md</span></li>
</ul>
<ul class="c15 lst-kix_pm62bnpnrfb1-0">
<li class="c5 li-bullet-0"><span class="c11">No credit (0 pts):</span><span>&nbsp;&lt; 1.2× speedup, or correctness failures, or missing README-perf.md</span></li>
</ul>
<p class="c10"><span class="c11">Note:</span><span class="c0">&nbsp;Baseline Task 4 performance will vary by GPU. We&rsquo;ll normalize based on the reference implementation&rsquo;s performance on the grading hardware. The speedup ratios above are relative to your Task 4 implementation.</span></p>
<p class="c10"><span>We&rsquo;ll publish a lightweight leaderboard (GFLOP/s). Please write a short </span><span class="c11">README-perf.md</span><span class="c0">&nbsp;describing:</span></p>
<ul class="c15 lst-kix_pm62bnpnrfb1-0">
<li class="c5 li-bullet-0"><span>What optimizations you tried</span></li>
<li class="c5 li-bullet-0"><span>Why each optimization helped (or didn&rsquo;t)</span></li>
<li class="c5 li-bullet-0"><span>Performance measurements (GFLOP/s) for Task 4 baseline and your optimized version</span></li>
<li class="c5 li-bullet-0"><span>Any insights about memory access patterns, occupancy, or other performance factors</span></li>
</ul>
<p class="c10 c6"><span class="c0"></span></p>'''

# Replace CUDA Primer section
cuda_primer_pattern = r'(<h2[^>]*id="h\.86rlcz2pvytg"[^>]*>.*?CUDA Primer.*?</h2>.*?)(<h2[^>]*id="h\.zgmt231tcy9k"[^>]*>.*?Task 1)'
match = re.search(cuda_primer_pattern, html, re.DOTALL | re.IGNORECASE)
if match:
    html = html[:match.start()] + cuda_primer_html + html[match.end(1):]
    print("✓ Replaced CUDA Primer section")
else:
    print("✗ Could not find CUDA Primer section to replace")

# Replace Task 5 section
task5_pattern = r'(<h2[^>]*id="h\.nluym1yc8qbd"[^>]*>.*?Task 5.*?</h2>.*?)(<h2[^>]*id="h\.[^"]*"[^>]*>.*?Command|</body>)'
match2 = re.search(task5_pattern, html, re.DOTALL | re.IGNORECASE)
if match2:
    html = html[:match2.start()] + task5_html + html[match2.end(1):]
    print("✓ Replaced Task 5 section")
else:
    print("✗ Could not find Task 5 section to replace")

# Write the updated HTML
with open('webpage/webpage.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("✓ HTML file updated successfully!")

