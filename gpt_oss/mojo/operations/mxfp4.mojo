import compiler
from math import ceildiv
from runtime.asyncrt import DeviceContextPtr
from gpu import block_idx, thread_idx
import gpu.primitives.warp as warp
from gpu.memory import AddressSpace
from gpu.sync import barrier
from memory import stack_allocation
from tensor import InputTensor, OutputTensor, ManagedTensorSlice, foreach
from utils.index import IndexList
from bit import log2_floor
from math import exp2


########################################################
# Shared helpers
########################################################

@always_inline
fn fp4_values[
    simd_width: Int
](
    idx: SIMD[DType.uint8, simd_width]
) -> SIMD[DType.float32, simd_width]:
    var lut = SIMD[DType.float32, 16](
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    )
    return lut._dynamic_shuffle[simd_width](idx)


########################################################
# Unpack kernel (MXFP4 -> BF16)
########################################################

@compiler.register("mxfp4_unpack")
struct MxFp4Unpack:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        outp: OutputTensor[dtype=DType.bfloat16, rank=2],
        blocks: InputTensor[dtype=DType.uint8, rank=2],
        scales: InputTensor[dtype=DType.int32, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "gpu":
            _mxfp4_unpack_gpu(outp, blocks, scales, ctx)
        elif target == "cpu":
            @parameter
            @always_inline
            fn unpack[
                simd_width: Int
            ](idx: IndexList[outp.rank]) -> SIMD[DType.bfloat16, simd_width]:
                var row = idx[0]
                var col = idx[1]
                var block_idx = col >> 1
                var byte = blocks.load[1](IndexList[2](row, block_idx)).cast[DType.uint8]()
                var lo = byte & 0x0F
                var hi = byte >> 4
                var nibble = lo
                if (col & 1) == 1:
                    nibble = hi
                var value = fp4_values[1](SIMD[DType.uint8, 1](nibble))
                var scale = scales.load[1](IndexList[1](row)).cast[DType.float32]()
                var scaled = value * exp2(SIMD[DType.float32, 1](scale))
                return scaled.cast[DType.bfloat16]()

            foreach[unpack, target=target, simd_width=1](outp, ctx)
        else:
            raise Error("No known target:", target)


fn _mxfp4_unpack_gpu(
    outp: ManagedTensorSlice[mut=True, dtype=DType.bfloat16, rank=2],
    blocks: ManagedTensorSlice[dtype=DType.uint8, rank=2],
    scales: ManagedTensorSlice[dtype=DType.int32, rank=1],
    ctx: DeviceContextPtr,
) raises:
    comptime THREADS_X = 128
    comptime BYTES_PER_TILE = THREADS_X * 2
    var gpu_ctx = ctx.get_device_context()
    var rows = outp.dim_size(0)
    var bytes_per_row = blocks.dim_size(1)
    if rows == 0 or bytes_per_row == 0:
        return
    var tiles_per_row = ceildiv(bytes_per_row, BYTES_PER_TILE)
    var total_tiles = rows * tiles_per_row

    @parameter
    fn kernel(rows: Int, bytes_per_row: Int, tiles_per_row: Int):
        var tile_idx = Int(block_idx.x)
        if tile_idx < rows * tiles_per_row:
            var row = tile_idx // tiles_per_row
            var tile = tile_idx - row * tiles_per_row
            var scale = scales.load[1](IndexList[1](row)).cast[DType.float32]()
            var scale_vec = SIMD[DType.float32, 2](scale)
            var byte0 = tile * BYTES_PER_TILE + Int(thread_idx.x)
            if byte0 < bytes_per_row:
                var packed0 = blocks.load[1](IndexList[2](row, byte0)).cast[DType.uint8]()
                var lo0 = packed0 & 0x0F
                var hi0 = packed0 >> 4
                var nibble0 = SIMD[DType.uint8, 2](lo0, hi0)
                var value0 = fp4_values[2](nibble0)
                var scaled0 = value0 * exp2(scale_vec)
                var out0 = scaled0.cast[DType.bfloat16]()
                var col0 = byte0 * 2
                outp.store[2](IndexList[2](row, col0), out0)
            var byte1 = byte0 + THREADS_X
            if byte1 < bytes_per_row:
                var packed1 = blocks.load[1](IndexList[2](row, byte1)).cast[DType.uint8]()
                var lo1 = packed1 & 0x0F
                var hi1 = packed1 >> 4
                var nibble1 = SIMD[DType.uint8, 2](lo1, hi1)
                var value1 = fp4_values[2](nibble1)
                var scaled1 = value1 * exp2(scale_vec)
                var out1 = scaled1.cast[DType.bfloat16]()
                var col1 = byte1 * 2
                outp.store[2](IndexList[2](row, col1), out1)

    var grid_x = total_tiles
    gpu_ctx.enqueue_function_experimental[kernel](
        rows,
        bytes_per_row,
        tiles_per_row,
        grid_dim=(grid_x, 1, 1),
        block_dim=(THREADS_X, 1, 1),
    )


########################################################
# Fused GEMM kernel (FP32 accumulate)
########################################################

@compiler.register("mxfp4_gemm")
struct MxFp4Gemm:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        outp: OutputTensor[dtype=DType.bfloat16, rank=3],
        x: InputTensor[dtype=DType.bfloat16, rank=3],
        blocks: InputTensor[dtype=DType.uint8, rank=4],
        scales: InputTensor[dtype=DType.int32, rank=3],
        expert_idx: InputTensor[dtype=DType.int32, rank=2],
        bias: InputTensor[dtype=DType.bfloat16, rank=2],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "gpu":
            _mxfp4_gemm_gpu(outp, x, blocks, scales, expert_idx, bias, ctx)
        else:
            raise Error("No known target:", target)


fn _mxfp4_gemm_gpu(
    outp: ManagedTensorSlice[mut=True, dtype=DType.bfloat16, rank=3],
    x: ManagedTensorSlice[dtype=DType.bfloat16, rank=3],
    blocks: ManagedTensorSlice[dtype=DType.uint8, rank=4],
    scales: ManagedTensorSlice[dtype=DType.int32, rank=3],
    expert_idx: ManagedTensorSlice[dtype=DType.int32, rank=2],
    bias: ManagedTensorSlice[dtype=DType.bfloat16, rank=2],
    ctx: DeviceContextPtr,
) raises:
    comptime THREADS_X = 128
    var gpu_ctx = ctx.get_device_context()
    var batch = x.dim_size(0)
    var experts_per_token = x.dim_size(1)
    var in_dim = x.dim_size(2)
    var rows = blocks.dim_size(1)
    var block_groups = blocks.dim_size(2)
    var block_bytes = blocks.dim_size(3)
    if batch == 0 or experts_per_token == 0 or rows == 0:
        return
    if in_dim != block_groups * block_bytes * 2:
        raise Error("Input dim mismatch:", in_dim)
    var total = batch * experts_per_token * rows

    @parameter
    fn kernel(
        batch: Int,
        experts_per_token: Int,
        rows: Int,
        block_groups: Int,
        block_bytes: Int,
        in_dim: Int,
        total: Int,
    ):
        var idx = Int(block_idx.x) * THREADS_X + Int(thread_idx.x)
        if idx < total:
            var row = idx % rows
            var tmp = idx // rows
            var expert_slot = tmp % experts_per_token
            var batch_idx = tmp // experts_per_token
            var expert = Int(
                expert_idx
                .load[1](IndexList[2](batch_idx, expert_slot))
                .cast[DType.int32]()
            )
            var acc: Float32 = 0.0
            for g in range(block_groups):
                var scale = scales.load[1](IndexList[3](expert, row, Int(g))).cast[DType.float32]()
                var scale_f = exp2(scale)
                var base = (g * block_bytes) * 2
                for b in range(block_bytes):
                    var packed = blocks.load[1](IndexList[4](expert, row, Int(g), Int(b))).cast[DType.uint8]()
                    var lo = packed & 0x0F
                    var hi = packed >> 4
                    var v0 = fp4_values[1](SIMD[DType.uint8, 1](lo))[0]
                    var v1 = fp4_values[1](SIMD[DType.uint8, 1](hi))[0]
                    var col = base + b * 2
                    var x0 = x.load[1](IndexList[3](batch_idx, expert_slot, col)).cast[DType.float32]()
                    var x1 = x.load[1](IndexList[3](batch_idx, expert_slot, col + 1)).cast[DType.float32]()
                    acc += v0 * scale_f * x0 + v1 * scale_f * x1
            var bias_val = bias.load[1](IndexList[2](expert, row)).cast[DType.float32]()
            var out_val = (acc + bias_val).cast[DType.bfloat16]()
            outp.store[1](IndexList[3](batch_idx, expert_slot, row), SIMD[DType.bfloat16, 1](out_val))

    var grid_x = ceildiv(total, THREADS_X)
    gpu_ctx.enqueue_function_experimental[kernel](
        batch,
        experts_per_token,
        rows,
        block_groups,
        block_bytes,
        in_dim,
        total,
        grid_dim=(grid_x, 1, 1),
        block_dim=(THREADS_X, 1, 1),
    )


########################################################
# Fused GEMM kernel (BF16 accumulate)
########################################################

@compiler.register("mxfp4_gemm_bf16acc")
struct MxFp4GemmBf16Acc:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        outp: OutputTensor[dtype=DType.bfloat16, rank=3],
        x: InputTensor[dtype=DType.bfloat16, rank=3],
        blocks: InputTensor[dtype=DType.uint8, rank=4],
        scales: InputTensor[dtype=DType.int32, rank=3],
        expert_idx: InputTensor[dtype=DType.int32, rank=2],
        bias: InputTensor[dtype=DType.bfloat16, rank=2],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "gpu":
            _mxfp4_gemm_bf16acc_gpu(outp, x, blocks, scales, expert_idx, bias, ctx)
        else:
            raise Error("No known target:", target)


fn _mxfp4_gemm_bf16acc_gpu(
    outp: ManagedTensorSlice[mut=True, dtype=DType.bfloat16, rank=3],
    x: ManagedTensorSlice[dtype=DType.bfloat16, rank=3],
    blocks: ManagedTensorSlice[dtype=DType.uint8, rank=4],
    scales: ManagedTensorSlice[dtype=DType.int32, rank=3],
    expert_idx: ManagedTensorSlice[dtype=DType.int32, rank=2],
    bias: ManagedTensorSlice[dtype=DType.bfloat16, rank=2],
    ctx: DeviceContextPtr,
) raises:
    comptime THREADS_X = 128
    var gpu_ctx = ctx.get_device_context()
    var batch = x.dim_size(0)
    var experts_per_token = x.dim_size(1)
    var in_dim = x.dim_size(2)
    var rows = blocks.dim_size(1)
    var block_groups = blocks.dim_size(2)
    var block_bytes = blocks.dim_size(3)
    if batch == 0 or experts_per_token == 0 or rows == 0:
        return
    if in_dim != block_groups * block_bytes * 2:
        raise Error("Input dim mismatch:", in_dim)
    var total = batch * experts_per_token * rows

    @parameter
    fn kernel(
        batch: Int,
        experts_per_token: Int,
        rows: Int,
        block_groups: Int,
        block_bytes: Int,
        in_dim: Int,
        total: Int,
    ):
        var idx = Int(block_idx.x) * THREADS_X + Int(thread_idx.x)
        if idx < total:
            var row = idx % rows
            var tmp = idx // rows
            var expert_slot = tmp % experts_per_token
            var batch_idx = tmp // experts_per_token
            var expert = Int(
                expert_idx
                .load[1](IndexList[2](batch_idx, expert_slot))
                .cast[DType.int32]()
            )
            var acc = SIMD[DType.bfloat16, 1](0.0)
            for g in range(block_groups):
                var scale = scales.load[1](IndexList[3](expert, row, Int(g))).cast[DType.float32]()
                var scale_f = exp2(scale)
                var base = (g * block_bytes) * 2
                for b in range(block_bytes):
                    var packed = blocks.load[1](IndexList[4](expert, row, Int(g), Int(b))).cast[DType.uint8]()
                    var lo = packed & 0x0F
                    var hi = packed >> 4
                    var v0 = fp4_values[1](SIMD[DType.uint8, 1](lo))[0]
                    var v1 = fp4_values[1](SIMD[DType.uint8, 1](hi))[0]
                    var col = base + b * 2
                    var x0 = x.load[1](IndexList[3](batch_idx, expert_slot, col)).cast[DType.float32]()
                    var x1 = x.load[1](IndexList[3](batch_idx, expert_slot, col + 1)).cast[DType.float32]()
                    var term = v0 * scale_f * x0 + v1 * scale_f * x1
                    acc = acc + SIMD[DType.bfloat16, 1](term)
            var bias_val = bias.load[1](IndexList[2](expert, row))
            var out_val = acc + bias_val
            outp.store[1](IndexList[3](batch_idx, expert_slot, row), out_val)

    var grid_x = ceildiv(total, THREADS_X)
    gpu_ctx.enqueue_function_experimental[kernel](
        batch,
        experts_per_token,
        rows,
        block_groups,
        block_bytes,
        in_dim,
        total,
        grid_dim=(grid_x, 1, 1),
        block_dim=(THREADS_X, 1, 1),
    )


########################################################
# Fused GEMM kernel (tiled, FP32 accumulate)
########################################################

@compiler.register("mxfp4_gemm_tiled")
struct MxFp4GemmTiled:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        outp: OutputTensor[dtype=DType.bfloat16, rank=3],
        x: InputTensor[dtype=DType.bfloat16, rank=3],
        blocks: InputTensor[dtype=DType.uint8, rank=4],
        scales: InputTensor[dtype=DType.int32, rank=3],
        expert_idx: InputTensor[dtype=DType.int32, rank=2],
        bias: InputTensor[dtype=DType.bfloat16, rank=2],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "gpu":
            _mxfp4_gemm_tiled_gpu(outp, x, blocks, scales, expert_idx, bias, ctx)
        else:
            raise Error("No known target:", target)


fn _mxfp4_gemm_tiled_gpu(
    outp: ManagedTensorSlice[mut=True, dtype=DType.bfloat16, rank=3],
    x: ManagedTensorSlice[dtype=DType.bfloat16, rank=3],
    blocks: ManagedTensorSlice[dtype=DType.uint8, rank=4],
    scales: ManagedTensorSlice[dtype=DType.int32, rank=3],
    expert_idx: ManagedTensorSlice[dtype=DType.int32, rank=2],
    bias: ManagedTensorSlice[dtype=DType.bfloat16, rank=2],
    ctx: DeviceContextPtr,
) raises:
    comptime TILE_ROWS = 128
    comptime BLOCK_BYTES = 16
    comptime VALUES_PER_BLOCK = BLOCK_BYTES * 2
    comptime TILE_GROUPS = 4
    comptime TILE_VALUES = TILE_GROUPS * VALUES_PER_BLOCK
    var gpu_ctx = ctx.get_device_context()
    var batch = x.dim_size(0)
    var experts_per_token = x.dim_size(1)
    var in_dim = x.dim_size(2)
    var rows = blocks.dim_size(1)
    var block_groups = blocks.dim_size(2)
    var block_bytes = blocks.dim_size(3)
    if batch == 0 or experts_per_token == 0 or rows == 0:
        return
    if block_bytes != BLOCK_BYTES:
        raise Error("Unsupported block_bytes:", block_bytes)
    if in_dim != block_groups * block_bytes * 2:
        raise Error("Input dim mismatch:", in_dim)
    var row_tiles = ceildiv(rows, TILE_ROWS)
    var tiles_per_batch = experts_per_token * row_tiles
    var total_tiles = batch * tiles_per_batch
    var k_tiles = ceildiv(block_groups, TILE_GROUPS)

    @parameter
    fn kernel(
        batch: Int,
        experts_per_token: Int,
        rows: Int,
        block_groups: Int,
        in_dim: Int,
        row_tiles: Int,
        tiles_per_batch: Int,
        total_tiles: Int,
        k_tiles: Int,
    ):
        var tile_idx = Int(block_idx.x)
        if tile_idx < total_tiles:
            var batch_idx = tile_idx // tiles_per_batch
            var rem = tile_idx - batch_idx * tiles_per_batch
            var expert_slot = rem // row_tiles
            var row_tile = rem - expert_slot * row_tiles
            var row = row_tile * TILE_ROWS + Int(thread_idx.x)
            var expert = Int(
                expert_idx
                .load[1](IndexList[2](batch_idx, expert_slot))
                .cast[DType.int32]()
            )
            var acc: Float32 = 0.0
            var x_shared = stack_allocation[
                TILE_VALUES,
                DType.float32,
                address_space = AddressSpace.SHARED,
            ]()

            for k_tile in range(k_tiles):
                var g0 = k_tile * TILE_GROUPS
                var t = Int(thread_idx.x)
                if t < TILE_VALUES:
                    var g_local = t // VALUES_PER_BLOCK
                    var offset = t - g_local * VALUES_PER_BLOCK
                    var b = offset // 2
                    var pair = offset - b * 2
                    var g = g0 + g_local
                    var col = g * VALUES_PER_BLOCK + b * 2 + pair
                    if g < block_groups and col < in_dim:
                        x_shared[t] = x.load[1](IndexList[3](batch_idx, expert_slot, col)).cast[DType.float32]()
                    else:
                        x_shared[t] = 0.0
                barrier()

                if row < rows:
                    for g_local in range(TILE_GROUPS):
                        var g = g0 + g_local
                        if g < block_groups:
                            var scale = scales.load[1](IndexList[3](expert, row, Int(g))).cast[DType.float32]()
                            var scale_f = exp2(scale)
                            var base = g_local * VALUES_PER_BLOCK
                            for b in range(BLOCK_BYTES):
                                var packed = blocks.load[1](IndexList[4](expert, row, Int(g), Int(b))).cast[DType.uint8]()
                                var lo = packed & 0x0F
                                var hi = packed >> 4
                                var v0 = fp4_values[1](SIMD[DType.uint8, 1](lo))[0]
                                var v1 = fp4_values[1](SIMD[DType.uint8, 1](hi))[0]
                                var x0 = x_shared[base + b * 2]
                                var x1 = x_shared[base + b * 2 + 1]
                                acc += v0 * scale_f * x0 + v1 * scale_f * x1
                barrier()

            if row < rows:
                var bias_val = bias.load[1](IndexList[2](expert, row)).cast[DType.float32]()
                var out_val = (acc + bias_val).cast[DType.bfloat16]()
                outp.store[1](IndexList[3](batch_idx, expert_slot, row), SIMD[DType.bfloat16, 1](out_val))

    var grid_x = total_tiles
    gpu_ctx.enqueue_function_experimental[kernel](
        batch,
        experts_per_token,
        rows,
        block_groups,
        in_dim,
        row_tiles,
        tiles_per_batch,
        total_tiles,
        k_tiles,
        grid_dim=(grid_x, 1, 1),
        block_dim=(TILE_ROWS, 1, 1),
    )


########################################################
# Fused GEMM kernel (tiled, BF16 accumulate)
########################################################

@compiler.register("mxfp4_gemm_tiled_bf16acc")
struct MxFp4GemmTiledBf16Acc:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        outp: OutputTensor[dtype=DType.bfloat16, rank=3],
        x: InputTensor[dtype=DType.bfloat16, rank=3],
        blocks: InputTensor[dtype=DType.uint8, rank=4],
        scales: InputTensor[dtype=DType.int32, rank=3],
        expert_idx: InputTensor[dtype=DType.int32, rank=2],
        bias: InputTensor[dtype=DType.bfloat16, rank=2],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "gpu":
            _mxfp4_gemm_tiled_bf16acc_gpu(outp, x, blocks, scales, expert_idx, bias, ctx)
        else:
            raise Error("No known target:", target)


fn _mxfp4_gemm_tiled_bf16acc_gpu(
    outp: ManagedTensorSlice[mut=True, dtype=DType.bfloat16, rank=3],
    x: ManagedTensorSlice[dtype=DType.bfloat16, rank=3],
    blocks: ManagedTensorSlice[dtype=DType.uint8, rank=4],
    scales: ManagedTensorSlice[dtype=DType.int32, rank=3],
    expert_idx: ManagedTensorSlice[dtype=DType.int32, rank=2],
    bias: ManagedTensorSlice[dtype=DType.bfloat16, rank=2],
    ctx: DeviceContextPtr,
) raises:
    comptime TILE_ROWS = 128
    comptime BLOCK_BYTES = 16
    comptime VALUES_PER_BLOCK = BLOCK_BYTES * 2
    comptime TILE_GROUPS = 4
    comptime TILE_VALUES = TILE_GROUPS * VALUES_PER_BLOCK
    var gpu_ctx = ctx.get_device_context()
    var batch = x.dim_size(0)
    var experts_per_token = x.dim_size(1)
    var in_dim = x.dim_size(2)
    var rows = blocks.dim_size(1)
    var block_groups = blocks.dim_size(2)
    var block_bytes = blocks.dim_size(3)
    if batch == 0 or experts_per_token == 0 or rows == 0:
        return
    if block_bytes != BLOCK_BYTES:
        raise Error("Unsupported block_bytes:", block_bytes)
    if in_dim != block_groups * block_bytes * 2:
        raise Error("Input dim mismatch:", in_dim)
    var row_tiles = ceildiv(rows, TILE_ROWS)
    var tiles_per_batch = experts_per_token * row_tiles
    var total_tiles = batch * tiles_per_batch
    var k_tiles = ceildiv(block_groups, TILE_GROUPS)

    @parameter
    fn kernel(
        batch: Int,
        experts_per_token: Int,
        rows: Int,
        block_groups: Int,
        in_dim: Int,
        row_tiles: Int,
        tiles_per_batch: Int,
        total_tiles: Int,
        k_tiles: Int,
    ):
        var tile_idx = Int(block_idx.x)
        if tile_idx < total_tiles:
            var batch_idx = tile_idx // tiles_per_batch
            var rem = tile_idx - batch_idx * tiles_per_batch
            var expert_slot = rem // row_tiles
            var row_tile = rem - expert_slot * row_tiles
            var row = row_tile * TILE_ROWS + Int(thread_idx.x)
            var expert = Int(
                expert_idx
                .load[1](IndexList[2](batch_idx, expert_slot))
                .cast[DType.int32]()
            )
            var acc = SIMD[DType.bfloat16, 1](0.0)
            var x_shared = stack_allocation[
                TILE_VALUES,
                DType.float32,
                address_space = AddressSpace.SHARED,
            ]()

            for k_tile in range(k_tiles):
                var g0 = k_tile * TILE_GROUPS
                var t = Int(thread_idx.x)
                if t < TILE_VALUES:
                    var g_local = t // VALUES_PER_BLOCK
                    var offset = t - g_local * VALUES_PER_BLOCK
                    var b = offset // 2
                    var pair = offset - b * 2
                    var g = g0 + g_local
                    var col = g * VALUES_PER_BLOCK + b * 2 + pair
                    if g < block_groups and col < in_dim:
                        x_shared[t] = x.load[1](IndexList[3](batch_idx, expert_slot, col)).cast[DType.float32]()
                    else:
                        x_shared[t] = 0.0
                barrier()

                if row < rows:
                    for g_local in range(TILE_GROUPS):
                        var g = g0 + g_local
                        if g < block_groups:
                            var scale = scales.load[1](IndexList[3](expert, row, Int(g))).cast[DType.float32]()
                            var scale_f = exp2(scale)
                            var base = g_local * VALUES_PER_BLOCK
                            for b in range(BLOCK_BYTES):
                                var packed = blocks.load[1](IndexList[4](expert, row, Int(g), Int(b))).cast[DType.uint8]()
                                var lo = packed & 0x0F
                                var hi = packed >> 4
                                var v0 = fp4_values[1](SIMD[DType.uint8, 1](lo))[0]
                                var v1 = fp4_values[1](SIMD[DType.uint8, 1](hi))[0]
                                var x0 = x_shared[base + b * 2]
                                var x1 = x_shared[base + b * 2 + 1]
                                var term = v0 * scale_f * x0 + v1 * scale_f * x1
                                acc = acc + SIMD[DType.bfloat16, 1](term)
                barrier()

            if row < rows:
                var bias_val = bias.load[1](IndexList[2](expert, row))
                var out_val = acc + bias_val
                outp.store[1](IndexList[3](batch_idx, expert_slot, row), out_val)

    var grid_x = total_tiles
    gpu_ctx.enqueue_function_experimental[kernel](
        batch,
        experts_per_token,
        rows,
        block_groups,
        in_dim,
        row_tiles,
        tiles_per_batch,
        total_tiles,
        k_tiles,
        grid_dim=(grid_x, 1, 1),
        block_dim=(TILE_ROWS, 1, 1),
    )


########################################################
# Fused GEMM kernel (warp-tiled, FP32 accumulate)
########################################################

@compiler.register("mxfp4_gemm_warp")
struct MxFp4GemmWarp:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        outp: OutputTensor[dtype=DType.bfloat16, rank=3],
        x: InputTensor[dtype=DType.bfloat16, rank=3],
        blocks: InputTensor[dtype=DType.uint8, rank=4],
        scales: InputTensor[dtype=DType.int32, rank=3],
        expert_idx: InputTensor[dtype=DType.int32, rank=2],
        bias: InputTensor[dtype=DType.bfloat16, rank=2],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "gpu":
            _mxfp4_gemm_warp_gpu(outp, x, blocks, scales, expert_idx, bias, ctx)
        else:
            raise Error("No known target:", target)


fn _mxfp4_gemm_warp_gpu(
    outp: ManagedTensorSlice[mut=True, dtype=DType.bfloat16, rank=3],
    x: ManagedTensorSlice[dtype=DType.bfloat16, rank=3],
    blocks: ManagedTensorSlice[dtype=DType.uint8, rank=4],
    scales: ManagedTensorSlice[dtype=DType.int32, rank=3],
    expert_idx: ManagedTensorSlice[dtype=DType.int32, rank=2],
    bias: ManagedTensorSlice[dtype=DType.bfloat16, rank=2],
    ctx: DeviceContextPtr,
) raises:
    comptime WARP = 32
    comptime WARPS_PER_BLOCK = 4
    comptime THREADS = WARP * WARPS_PER_BLOCK
    comptime BLOCK_BYTES = 16
    comptime VALUES_PER_GROUP = BLOCK_BYTES * 2
    comptime TILE_GROUPS = 8
    comptime TILE_VALUES = TILE_GROUPS * VALUES_PER_GROUP
    var gpu_ctx = ctx.get_device_context()
    var batch = x.dim_size(0)
    var experts_per_token = x.dim_size(1)
    var in_dim = x.dim_size(2)
    var rows = blocks.dim_size(1)
    var block_groups = blocks.dim_size(2)
    var block_bytes = blocks.dim_size(3)
    if batch == 0 or experts_per_token == 0 or rows == 0:
        return
    if block_bytes != BLOCK_BYTES:
        raise Error("Unsupported block_bytes:", block_bytes)
    if in_dim != block_groups * block_bytes * 2:
        raise Error("Input dim mismatch:", in_dim)
    var row_tiles = ceildiv(rows, WARPS_PER_BLOCK)
    var tiles_per_batch = experts_per_token * row_tiles
    var total_tiles = batch * tiles_per_batch
    var k_tiles = ceildiv(block_groups, TILE_GROUPS)

    @parameter
    fn kernel(
        batch: Int,
        experts_per_token: Int,
        rows: Int,
        block_groups: Int,
        in_dim: Int,
        row_tiles: Int,
        tiles_per_batch: Int,
        total_tiles: Int,
        k_tiles: Int,
    ):
        var tile_idx = Int(block_idx.x)
        var lane = Int(thread_idx.x)
        var warp_id = Int(thread_idx.y)
        if tile_idx < total_tiles:
            var batch_idx = tile_idx // tiles_per_batch
            var rem = tile_idx - batch_idx * tiles_per_batch
            var expert_slot = rem // row_tiles
            var row_tile = rem - expert_slot * row_tiles
            var row = row_tile * WARPS_PER_BLOCK + warp_id
            var expert = 0
            if row < rows:
                expert = Int(
                    expert_idx
                    .load[1](IndexList[2](batch_idx, expert_slot))
                    .cast[DType.int32]()
                )
            var acc: Float32 = 0.0
            var x_shared = stack_allocation[
                TILE_VALUES * 2,
                DType.float32,
                address_space = AddressSpace.SHARED,
            ]()

            for k_tile in range(k_tiles):
                var buf = k_tile & 1
                var base_shared = buf * TILE_VALUES
                var t = warp_id * WARP + lane
                @parameter
                for load_base in range(0, TILE_VALUES, THREADS):
                    var t_idx = load_base + t
                    if t_idx < TILE_VALUES:
                        var g_local = t_idx // VALUES_PER_GROUP
                        var value_idx = t_idx - g_local * VALUES_PER_GROUP
                        var g = k_tile * TILE_GROUPS + g_local
                        var col = g * VALUES_PER_GROUP + value_idx
                        if g < block_groups and col < in_dim:
                            x_shared[base_shared + t_idx] = x.load[1](IndexList[3](batch_idx, expert_slot, col)).cast[DType.float32]()
                        else:
                            x_shared[base_shared + t_idx] = 0.0
                barrier()

                if row < rows:
                    var value_idx = lane
                    var byte = value_idx // 2
                    var pair = value_idx - byte * 2
                    for g_local in range(TILE_GROUPS):
                        var g = k_tile * TILE_GROUPS + g_local
                        if g < block_groups:
                            var scale = scales.load[1](IndexList[3](expert, row, Int(g))).cast[DType.float32]()
                            var scale_f = exp2(scale)
                            var packed = blocks.load[1](IndexList[4](expert, row, Int(g), Int(byte))).cast[DType.uint8]()
                            var nibble = packed & 0x0F
                            if pair == 1:
                                nibble = packed >> 4
                            var v = fp4_values[1](SIMD[DType.uint8, 1](nibble))[0]
                            var xval = x_shared[base_shared + g_local * VALUES_PER_GROUP + value_idx]
                            acc += v * scale_f * xval
                barrier()

            var sum = SIMD[DType.float32, 1](acc)
            comptime limit = log2_floor(WARP)
            @parameter
            for mask in reversed(range(limit)):
                sum += warp.shuffle_down(sum, 1 << mask)
            if row < rows and lane == 0:
                var bias_val = bias.load[1](IndexList[2](expert, row)).cast[DType.float32]()
                var out_val = (sum + SIMD[DType.float32, 1](bias_val)).cast[DType.bfloat16]()
                outp.store[1](IndexList[3](batch_idx, expert_slot, row), SIMD[DType.bfloat16, 1](out_val))

    var grid_x = total_tiles
    gpu_ctx.enqueue_function_experimental[kernel](
        batch,
        experts_per_token,
        rows,
        block_groups,
        in_dim,
        row_tiles,
        tiles_per_batch,
        total_tiles,
        k_tiles,
        grid_dim=(grid_x, 1, 1),
        block_dim=(WARP, WARPS_PER_BLOCK, 1),
    )


########################################################
# Fused GEMM kernel (warp-tiled, BF16 accumulate)
########################################################

@compiler.register("mxfp4_gemm_warp_bf16acc")
struct MxFp4GemmWarpBf16Acc:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        outp: OutputTensor[dtype=DType.bfloat16, rank=3],
        x: InputTensor[dtype=DType.bfloat16, rank=3],
        blocks: InputTensor[dtype=DType.uint8, rank=4],
        scales: InputTensor[dtype=DType.int32, rank=3],
        expert_idx: InputTensor[dtype=DType.int32, rank=2],
        bias: InputTensor[dtype=DType.bfloat16, rank=2],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "gpu":
            _mxfp4_gemm_warp_bf16acc_gpu(outp, x, blocks, scales, expert_idx, bias, ctx)
        else:
            raise Error("No known target:", target)


fn _mxfp4_gemm_warp_bf16acc_gpu(
    outp: ManagedTensorSlice[mut=True, dtype=DType.bfloat16, rank=3],
    x: ManagedTensorSlice[dtype=DType.bfloat16, rank=3],
    blocks: ManagedTensorSlice[dtype=DType.uint8, rank=4],
    scales: ManagedTensorSlice[dtype=DType.int32, rank=3],
    expert_idx: ManagedTensorSlice[dtype=DType.int32, rank=2],
    bias: ManagedTensorSlice[dtype=DType.bfloat16, rank=2],
    ctx: DeviceContextPtr,
) raises:
    comptime WARP = 32
    comptime WARPS_PER_BLOCK = 4
    comptime THREADS = WARP * WARPS_PER_BLOCK
    comptime BLOCK_BYTES = 16
    comptime VALUES_PER_GROUP = BLOCK_BYTES * 2
    comptime TILE_GROUPS = 8
    comptime TILE_VALUES = TILE_GROUPS * VALUES_PER_GROUP
    var gpu_ctx = ctx.get_device_context()
    var batch = x.dim_size(0)
    var experts_per_token = x.dim_size(1)
    var in_dim = x.dim_size(2)
    var rows = blocks.dim_size(1)
    var block_groups = blocks.dim_size(2)
    var block_bytes = blocks.dim_size(3)
    if batch == 0 or experts_per_token == 0 or rows == 0:
        return
    if block_bytes != BLOCK_BYTES:
        raise Error("Unsupported block_bytes:", block_bytes)
    if in_dim != block_groups * block_bytes * 2:
        raise Error("Input dim mismatch:", in_dim)
    var row_tiles = ceildiv(rows, WARPS_PER_BLOCK)
    var tiles_per_batch = experts_per_token * row_tiles
    var total_tiles = batch * tiles_per_batch
    var k_tiles = ceildiv(block_groups, TILE_GROUPS)

    @parameter
    fn kernel(
        batch: Int,
        experts_per_token: Int,
        rows: Int,
        block_groups: Int,
        in_dim: Int,
        row_tiles: Int,
        tiles_per_batch: Int,
        total_tiles: Int,
        k_tiles: Int,
    ):
        var tile_idx = Int(block_idx.x)
        var lane = Int(thread_idx.x)
        var warp_id = Int(thread_idx.y)
        if tile_idx < total_tiles:
            var batch_idx = tile_idx // tiles_per_batch
            var rem = tile_idx - batch_idx * tiles_per_batch
            var expert_slot = rem // row_tiles
            var row_tile = rem - expert_slot * row_tiles
            var row = row_tile * WARPS_PER_BLOCK + warp_id
            var expert = 0
            if row < rows:
                expert = Int(
                    expert_idx
                    .load[1](IndexList[2](batch_idx, expert_slot))
                    .cast[DType.int32]()
                )
            var acc = SIMD[DType.bfloat16, 1](0.0)
            var x_shared = stack_allocation[
                TILE_VALUES * 2,
                DType.float32,
                address_space = AddressSpace.SHARED,
            ]()

            for k_tile in range(k_tiles):
                var buf = k_tile & 1
                var base_shared = buf * TILE_VALUES
                var t = warp_id * WARP + lane
                @parameter
                for load_base in range(0, TILE_VALUES, THREADS):
                    var t_idx = load_base + t
                    if t_idx < TILE_VALUES:
                        var g_local = t_idx // VALUES_PER_GROUP
                        var value_idx = t_idx - g_local * VALUES_PER_GROUP
                        var g = k_tile * TILE_GROUPS + g_local
                        var col = g * VALUES_PER_GROUP + value_idx
                        if g < block_groups and col < in_dim:
                            x_shared[base_shared + t_idx] = x.load[1](IndexList[3](batch_idx, expert_slot, col)).cast[DType.float32]()
                        else:
                            x_shared[base_shared + t_idx] = 0.0
                barrier()

                if row < rows:
                    var value_idx = lane
                    var byte = value_idx // 2
                    var pair = value_idx - byte * 2
                    for g_local in range(TILE_GROUPS):
                        var g = k_tile * TILE_GROUPS + g_local
                        if g < block_groups:
                            var scale = scales.load[1](IndexList[3](expert, row, Int(g))).cast[DType.float32]()
                            var scale_f = exp2(scale)
                            var packed = blocks.load[1](IndexList[4](expert, row, Int(g), Int(byte))).cast[DType.uint8]()
                            var nibble = packed & 0x0F
                            if pair == 1:
                                nibble = packed >> 4
                            var v = fp4_values[1](SIMD[DType.uint8, 1](nibble))[0]
                            var xval = x_shared[base_shared + g_local * VALUES_PER_GROUP + value_idx]
                            var term = v * scale_f * xval
                            acc = acc + SIMD[DType.bfloat16, 1](term)
                barrier()

            var sum = acc
            comptime limit = log2_floor(WARP)
            @parameter
            for mask in reversed(range(limit)):
                sum = sum + warp.shuffle_down(sum, 1 << mask)
            if row < rows and lane == 0:
                var bias_val = bias.load[1](IndexList[2](expert, row))
                var out_val = sum + SIMD[DType.bfloat16, 1](bias_val)
                outp.store[1](IndexList[3](batch_idx, expert_slot, row), out_val)

    var grid_x = total_tiles
    gpu_ctx.enqueue_function_experimental[kernel](
        batch,
        experts_per_token,
        rows,
        block_groups,
        in_dim,
        row_tiles,
        tiles_per_batch,
        total_tiles,
        k_tiles,
        grid_dim=(grid_x, 1, 1),
        block_dim=(WARP, WARPS_PER_BLOCK, 1),
    )


########################################################
# Legacy Unpack (foreach reference)
########################################################

# import compiler
# from runtime.asyncrt import DeviceContextPtr
# from tensor import InputTensor, OutputTensor, foreach
# from utils.index import IndexList
# from math import exp2


# @always_inline
# fn fp4_values[
#     simd_width: Int
# ](
#     idx: SIMD[DType.uint8, simd_width]
# ) -> SIMD[DType.float32, simd_width]:
#     var lut = SIMD[DType.float32, 16](
#         0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
#         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
#     )
#     return lut._dynamic_shuffle[simd_width](idx)


# @compiler.register("mxfp4_unpack")
# struct MxFp4Unpack:
#     @staticmethod
#     fn execute[
#         target: StaticString,
#     ](
#         outp: OutputTensor[dtype=DType.bfloat16, rank=2],
#         blocks: InputTensor[dtype=DType.uint8, rank=2],
#         scales: InputTensor[dtype=DType.int32, rank=1],
#         ctx: DeviceContextPtr,
#     ) raises:
#         @parameter
#         @always_inline
#         fn unpack[
#             simd_width: Int
#         ](idx: IndexList[outp.rank]) -> SIMD[DType.bfloat16, simd_width]:
#             var row = idx[0]
#             var col = idx[1]
#             @parameter
#             if simd_width == 1:
#                 var block_idx = col >> 1
#                 var byte = blocks.load[1](IndexList[2](row, block_idx)).cast[DType.uint8]()
#                 var lo = byte & 0x0F
#                 var hi = byte >> 4
#                 var nibble = lo
#                 if (col & 1) == 1:
#                     nibble = hi
#                 var value = fp4_values[1](SIMD[DType.uint8, 1](nibble))
#                 var scale = scales.load[1](IndexList[1](row)).cast[DType.float32]()
#                 var scaled = value * exp2(SIMD[DType.float32, 1](scale))
#                 return scaled.cast[DType.bfloat16]()
#             @parameter
#             if simd_width == 2:
#                 var block_idx = col >> 1
#                 var byte = blocks.load[1](IndexList[2](row, block_idx)).cast[DType.uint8]()
#                 var lo = byte & 0x0F
#                 var hi = byte >> 4
#                 var nibble = SIMD[DType.uint8, 2](lo, hi)
#                 var value = fp4_values[2](nibble)
#                 var scale = scales.load[1](IndexList[1](row)).cast[DType.float32]()
#                 var scale_vec = SIMD[DType.float32, 2](scale)
#                 var scaled = value * exp2(scale_vec)
#                 var out_val = scaled.cast[DType.bfloat16]()
#                 return rebind[SIMD[DType.bfloat16, simd_width]](out_val)
#             return SIMD[DType.bfloat16, simd_width]()

#         foreach[unpack, target=target, simd_width=2](outp, ctx)
