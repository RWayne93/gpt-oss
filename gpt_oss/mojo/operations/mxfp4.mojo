import compiler
from math import ceildiv
from runtime.asyncrt import DeviceContextPtr
from gpu import block_idx, thread_idx
from tensor import InputTensor, OutputTensor, ManagedTensorSlice, foreach
from utils.index import IndexList
from math import exp2


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
# Initial MxFp4Unpack implementation
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