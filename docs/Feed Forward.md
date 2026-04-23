# Neural Network from Scratch — Feed Forward Implementation Notes

This document examines the design and implementation of a feed-forward neural network built from scratch in C and CUDA. It walks through the core structural choices shared across both serial and parallel execution paths. The document then weighs the tradeoffs of this approach, noting strengths like cache-friendly memory layout and a clean stateless interface alongside limitations such as the absence of retained intermediate activations (which precludes backpropagation) and the lack of batched inference support. Finally, it contrasts the serial host implementation with the parallel CUDA kernel, detailing how the device version maps output neurons to threads, uses shared memory tiling to reduce redundant global memory reads, and relies on synchronization barriers to maintain correctness across the retained sequential layer loop.

## Feed Forward Operation Approach

### Flat Array Layout

All weights and biases are stored in flat, contiguous arrays rather than as jagged 2D arrays or linked structures. Weight matrices are laid out row-major — the weights connecting layer `l-1` to layer `l` occupy a contiguous block of `layers[l] × layers[l-1]` doubles within the global `weights` array. A companion `layer_offsets` array (computed once by `compute_layer_offsets`) stores the starting index for each layer's weight block, making random access into any layer's weights O(1).

This layout was a deliberate choice to support both the host and device paths with the same memory representation: flat arrays transfer to device memory cleanly, avoid pointer indirection that CUDA kernels cannot follow, and are friendly to cache prefetchers on both CPU and GPU.

### Ping-Pong Buffers

Rather than allocating a separate output vector for each layer, the implementation uses two pre-allocated buffers (`buffer1` and `buffer2`) that are swapped after each layer. The convention is that `buffer1` holds the active input at the start of the call and, after all layers complete, also holds the final output. The swap is a pointer exchange and has zero memory cost.

Both buffers are sized to `max(num_features, max(layers))` — the largest vector that will ever pass through the network. This means a single allocation up front is sufficient for the full forward pass, regardless of network depth or layer widths.

### Activation Functions via Enum Dispatch

Activation functions are represented as an `ActivationFunc` enum and resolved through a switch statement in `activation_func_hst` / `activation_func_dvc`. Each layer carries its own activation specification in the `activation_fns` array, so mixed-activation networks are supported without any structural change.

### Bias Handling

Biases are stored in a single flat array ordered by layer, with the first `layers[0]` entries being the biases for the first layer, the next `layers[1]` for the second, and so on. The running `total_neurons` counter inside the forward pass loop serves as the offset into this array and is incremented after each layer.

## Advantages of This Structure

**Memory efficiency.** The ping-pong buffer strategy means memory usage scales with the widest layer, not the sum of all layer sizes. For deep networks this is a significant saving.

**Zero-copy data layout.** The flat weight and bias arrays can be passed directly to `cudaMemcpy` without any reformatting. There is no marshalling step between the host and device paths.

**Simple caller interface.** The caller is responsible for pre-computing `layer_offsets` and allocating the two buffers, but after that the forward pass is a single function call. The network's topology is fully described by the four arrays (`layers`, `layer_offsets`, `activation_fns`, weights/biases) and does not require a struct or object.

**Separation of structure from compute.** The forward pass functions are stateless and take everything they need as parameters. There is no global or hidden state, which makes the functions straightforward to test in isolation and easy to call repeatedly on different inputs without reinitializing any object.

## Disadvantages and Current Limitations

**No intermediate activations are retained.** Because buffers are swapped after every layer, the activation values from intermediate layers are overwritten. This makes backpropagation impossible with the current structure — a training pass would need to either cache each layer's output separately or recompute them during the backward pass.

**Manual memory management at the call site.** The caller is responsible for allocating and freeing buffers, computing layer offsets, and ensuring all array sizes are consistent. There is no bounds checking or size validation inside the forward pass functions. A mismatch between the `layer_offsets` array and the actual weight layout will silently produce wrong results.

**Single-sample forward pass only.** The current interface processes one feature vector at a time. Batched inference (processing a matrix of samples in parallel) would require a significant rework of both the function signatures and the GPU kernel's thread distribution strategy.

**`total_neurons` offset computation is coupled to the loop.** The bias indexing relies on `total_neurons` accumulating correctly across layers inside the loop body. This works correctly but means the offset logic is implicit rather than pre-computed (unlike weights, which have a dedicated `layer_offsets` array). A parallel `bias_offsets` array would make the structure more symmetric and self-documenting.

**Buffer ownership is ambiguous after the call.** Because the buffers are swapped an unknown number of times (once per layer), the result ends up in whichever buffer corresponds to the final swap state. The convention — result is always in `buffer1` — is documented in the header but relies on the caller knowing that the parity of the number of layers determines which physical buffer `buffer1` points to on return. A wrapper or a returned pointer would make this less error-prone.

## `feed_forward_hst` vs. `feed_forward_dvc`

Both functions accept the same logical parameters and implement the same mathematical operation. The differences are entirely about execution model and memory location.

### Execution Model

`feed_forward_hst` is a standard C function that runs serially on the CPU. It uses a three-level nested loop: outer over layers, middle over neurons in the current layer, inner over inputs to each neuron. There is no parallelism; each neuron's weighted sum is computed sequentially.

`feed_forward_dvc` is a CUDA `__global__` kernel. The outer loop over layers is retained (layers must still be processed in order due to data dependencies), but the inner two loops are replaced by parallel thread execution. Each thread in the x-dimension is responsible for computing the output of one neuron in the current layer (`idx = blockIdx.x * blockDim.x + threadIdx.x`).

### Shared Memory Tiling

The device version introduces a tiled loading strategy using `__shared__ double tile[TILE_SIZE]`. Before computing weighted sums, threads cooperatively load a tile of the current input buffer (`buffer1`) into shared memory. The y-dimension of the thread block (`threadIdx.y`) drives this load: each y-thread loads one element of the tile from global memory, while the x-threads (which do the arithmetic) wait at a `__syncthreads()` barrier. Once the tile is loaded, every x-thread reads from shared memory rather than global memory for that tile's portion of the dot product.

This amortizes global memory latency across the tile: without shared memory, each of the `curr_neurons_out` threads would independently fetch the same `TILE_SIZE` input values from global memory, causing redundant and expensive reads. With the tile, those values are fetched once per block and reused by all x-threads.

Note: the current implementation loads the tile using only `threadIdx.x == 0` threads in the x-dimension to avoid race conditions, meaning the full y-dimension width of threads participates in loading but only a single x-lane does the work. A comment in the source flags this as a known inefficiency ("x dimension threads are wasted"), and a more complete implementation would have all x-threads cooperate in loading.

### Thread and Block Dimensions

The host version requires no grid configuration. The device version requires the caller to launch with a 2D block (`dim3 threads_per_block(TILE_SIZE, TILE_SIZE)`) where:

- The **x-dimension** maps one thread to one output neuron.
- The **y-dimension** drives cooperative loading of input tiles into shared memory.

The number of blocks in x is the ceiling division of the largest layer size by `TILE_SIZE`, ensuring all neurons in the widest layer are covered. `TILE_SIZE` is defined as `32` in `constants.h`, matching the CUDA warp size to avoid partially-filled warps.

### Memory Residency

For `feed_forward_hst`, all arrays (`weights`, `biases`, `buffer1`, `buffer2`, etc.) live in host (CPU) memory. For `feed_forward_dvc`, they must all reside in device (GPU) memory before the kernel is launched. The companion function `nn_load_dvc` handles this: it allocates device memory and copies the network's structure, weights, biases, and buffers to the GPU. The input feature vector must then be copied to `buffer1` on the device (via `cudaMemcpy`) before each kernel launch, and the output must be copied back from `buffer1` on the device after the kernel returns.

### Layer Loop Synchronization

Both versions use the same ping-pong buffer swap after each layer. In the host version this is a trivial pointer swap on the stack. In the device version, the same local pointer swap happens inside the kernel — but because all threads see the same pointer values (passed as kernel parameters and modified locally within the kernel function), this works correctly as long as `__syncthreads()` barriers ensure all writes to `buffer2` are complete before the pointers are swapped and the next layer begins. The two `__syncthreads()` calls per tile iteration — one before computing (to ensure the tile is loaded) and one after (to ensure all threads have finished reading from shared memory before it is potentially overwritten) — collectively enforce this ordering.

### Summary Table

| Aspect | `feed_forward_hst` | `feed_forward_dvc` |
|---|---|---|
| Language | C (`neural_net.c`) | CUDA C++ (`neural_net.cu`) |
| Execution | Serial, single-threaded | Parallel, one thread per output neuron |
| Memory | Host (CPU) | Device (GPU) |
| Layer loop | Retained; serial | Retained; serial (data dependency) |
| Neuron loop | Serial inner loop | Parallelized via thread x-index |
| Input loading | Direct array access | Tiled via `__shared__` memory |
| Synchronization | None needed | `__syncthreads()` per tile |
| Setup cost | None | `nn_load_dvc()` + per-sample `cudaMemcpy` |
| Tile awareness | None | `TILE_SIZE` from `constants.h` |
