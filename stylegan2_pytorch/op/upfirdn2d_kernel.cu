#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <ATen/cuda/CUDAApplyUtils.cuh>

/**
 * Using macros (((a) + (b) - 1) / (b)) or (((a) - 1) / (b) + 1) are wrong for
 *non-positive numbers Note that these macros are also not identical in
 *behaviour, the former is "less wrong"
 **/
static __host__ __device__ __forceinline__ int ceil_div(int a, int b) {
	int c = a / b;

	if (c * b < a) {
		c++;
	}

	return c;
}

/**
 * Options for UpFirDn kernel launch
 **/
struct UpFirDn2DKernelParams {
	// Up / down sampling factors
	int up_x;
	int up_y;
	int down_x;
	int down_y;

	// Paddings
	int pad_x0;
	int pad_x1;
	int pad_y0;
	int pad_y1;

	// Input Dimensions
	int n;
	int in_h;
	int in_w;

	// Kernel Dimensions
	int kernel_h;
	int kernel_w;

	// Output Dimensions
	int out_h;
	int out_w;

	// Per-thread loop count along dimension n
	int loop_n;
};

/**
 * Generic implementation of UpFirDn2d (without any caching via shared memory)
 **/
template <typename scalar_t>
__global__ void upfirdn2d_kernel_generic(scalar_t* out, const scalar_t* input,
										 const scalar_t*			 kernel,
										 const UpFirDn2DKernelParams p) {
	// Output pixel(s) (base) coordinates
	const int out_x		 = blockIdx.x * blockDim.x + threadIdx.x;
	const int out_y		 = blockIdx.y * blockDim.y + threadIdx.y;
	const int out_n_base = blockIdx.z * p.loop_n;

	if (out_x >= p.out_w || out_y >= p.out_h || out_n_base >= p.n) {
		return;
	}

	// Calculate middle layer (after upsampling) coordinates
	const int mid_x = out_x * p.down_x - p.pad_x0;
	const int mid_y = out_y * p.down_y - p.pad_y0;

	const int in_x = min(max(ceil_div(mid_x, p.up_x), 0), p.in_w);
	const int w =
		min(max(ceil_div(mid_x + p.kernel_w, p.up_x), 0), p.in_w) - in_x;
	const int kernel_x = p.kernel_w - 1 + mid_x - in_x * p.up_x;

	const int in_y = min(max(ceil_div(mid_y, p.up_y), 0), p.in_h);
	const int h =
		min(max(ceil_div(mid_y + p.kernel_h, p.up_y), 0), p.in_h) - in_y;
	const int kernel_y = p.kernel_h - 1 + mid_y - in_y * p.up_y;

	// Loop over DIM N
	for (int loop_n = 0, out_n = out_n_base; loop_n < p.loop_n && out_n < p.n;
		 loop_n++, out_n++) {
		// Pointer to start of input and kernel
		const scalar_t* x_p = &input[(out_n * p.in_h + in_y) * p.in_w + in_x];
		const scalar_t* k_p = &kernel[kernel_y * p.kernel_w + kernel_x];

		// Pointer step sizes in DIM x
		const int x_px = 1;
		const int k_px = -p.up_x;

		// Pointer step sizes to move from (end_x, y) to (start_x, y+1)
		const int x_py = p.in_w - w * x_px;
		const int k_py = -p.up_y * p.kernel_w - w * k_px;

		scalar_t v = 0.0;

		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				// Accumulate sum-product
				v += static_cast<scalar_t>(*x_p) * static_cast<scalar_t>(*k_p);
				// Move pointer in x-direction
				x_p += x_px;
				k_p += k_px;
			}

			x_p += x_py;
			k_p += k_py;
		}

		// Store output pixel
		out[(out_n * p.out_h + out_y) * p.out_w + out_x] = v;
	}
}

/**
 * Templated implementation of UpFirDn2d with:
 * 1. shared memory caching for improved read-access
 * 2. loop unrolling due to known kernel size
 *
 * Note that benefits of shared memory can only be realized fully
 * if each thread handles more than one pixel
 *
 * Each block is reponsible for a tile (e.g. N x 1 x 32 x 32) in
 * the output image. Threads inside the block split up the work
 * by looping over loop_n and the tile dimensions.
 **/
template <typename scalar_t, int up_x, int up_y, int down_x, int down_y,
		  int kernel_h, int kernel_w, int tile_out_h, int tile_out_w>
__global__ void upfirdn2d_kernel(scalar_t* out, const scalar_t* input,
								 const scalar_t*			 kernel,
								 const UpFirDn2DKernelParams p) {
	// Size of input tile to obtain desired size of output tile (reverse of the
	// previous step in calculating overall output size)
	constexpr int tile_in_h =
		((tile_out_h - 1) * down_y + kernel_h - 1) / up_y + 1;
	constexpr int tile_in_w =
		((tile_out_w - 1) * down_x + kernel_w - 1) / up_x + 1;

	// Shared (transposed) kernel and shared x
	__shared__ volatile float sk[kernel_h][kernel_w];
	__shared__ volatile float sx[tile_in_h][tile_in_w];

	// Block Information (no checks on block_out_x < p.out_w cuz it will be
	// checked on host)
	const int tile_out_x	  = blockIdx.x * tile_out_w;
	const int tile_out_y	  = blockIdx.y * tile_out_h;
	const int tile_out_n_base = blockIdx.z * p.loop_n;

	if (tile_out_x >= p.out_w | tile_out_y >= p.out_h |
		tile_out_n_base >= p.n) {
		return;
	}

	// Load shared (transposed) kernel
	for (int tap_idx = threadIdx.x; tap_idx < kernel_h * kernel_w;
		 tap_idx += blockDim.x) {
		int ky = tap_idx / kernel_w;
		int kx = tap_idx - ky * kernel_w;

		sk[ky][kx] =
			kernel[(p.kernel_h - 1 - ky) * p.kernel_w + (p.kernel_w - 1 - kx)];
	}

	// Loop over channels
	for (int loop_n = 0, tile_out_n = tile_out_n_base;
		 loop_n < p.loop_n & tile_out_n < p.n; loop_n++, tile_out_n++) {
		// Starting coordinates of block's output tile
		int tile_mid_x = tile_out_x * down_x - p.pad_x0;
		int tile_mid_y = tile_out_y * down_y - p.pad_y0;
		int tile_in_x  = ceil_div(tile_mid_x, up_x);
		int tile_in_y  = ceil_div(tile_mid_y, up_y);

		__syncthreads();

		// Load shared input
		for (int in_idx = threadIdx.x; in_idx < tile_in_h * tile_in_w;
			 in_idx += blockDim.x) {
			// Calculate relative coordinates in input
			int rel_in_y = in_idx / tile_in_w;
			int rel_in_x = in_idx - rel_in_y * tile_in_w;
			int in_x	 = rel_in_x + tile_in_x;
			int in_y	 = rel_in_y + tile_in_y;

			scalar_t v = 0.0;

			if (in_x >= 0 & in_y >= 0 & in_x < p.in_w & in_y < p.in_h) {
				v = input[(tile_out_n * p.in_h + in_y) * p.in_w + in_x];
			}

			// Imperative to initialize all tensor elements to 0 if not
			// covered by input
			sx[rel_in_y][rel_in_x] = v;
		}

		__syncthreads();

		// Accumulate output
		for (int out_idx = threadIdx.x; out_idx < tile_out_h * tile_out_w;
			 out_idx += blockDim.x) {
			// Calculate relative coordinates in output
			int rel_out_y = out_idx / tile_out_w;
			int rel_out_x = out_idx - rel_out_y * tile_out_w;
			int out_x	  = rel_out_x + tile_out_x;
			int out_y	  = rel_out_y + tile_out_y;

			// Calculate cooresponding coordinates in input
			int mid_x	 = tile_mid_x + rel_out_x * down_x;
			int mid_y	 = tile_mid_y + rel_out_y * down_y;
			int in_x	 = ceil_div(mid_x, up_x);
			int in_y	 = ceil_div(mid_y, up_y);
			int rel_in_x = in_x - tile_in_x;
			int rel_in_y = in_y - tile_in_y;
			int kernel_x = in_x * up_x - mid_x;
			int kernel_y = in_y * up_y - mid_y;

			scalar_t v = 0.0;

#pragma unroll
			for (int y = 0; y < kernel_h / up_y; y++)
#pragma unroll
				for (int x = 0; x < kernel_w / up_x; x++)
					v += sx[rel_in_y + y][rel_in_x + x] *
						 sk[kernel_y + y * up_y][kernel_x + x * up_x];

			if (out_x < p.out_w & out_y < p.out_h) {
				out[(tile_out_n * p.out_h + out_y) * p.out_w + out_x] = v;
			}
		}
	}
}

torch::Tensor upfirdn2d_op(const torch::Tensor& input,
						   const torch::Tensor& kernel, int up_x, int up_y,
						   int down_x, int down_y, int pad_x0, int pad_x1,
						   int pad_y0, int pad_y1) {
	/**
     * Code in original source, but already removed in NVIDIA's newest
     *implementation, should be useless int curDevice = -1;
     * cudaGetDevice(&curDevice);
     **/

	// Get CUDA stream
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();

	auto x = input.contiguous();
	auto k = kernel.contiguous();

	// Get parameters
	UpFirDn2DKernelParams p;
	p.n		   = x.size(0);
	p.in_h	   = x.size(2);
	p.in_w	   = x.size(3);
	p.kernel_h = k.size(0);
	p.kernel_w = k.size(1);
	p.up_x	   = up_x;
	p.up_y	   = up_y;
	p.down_x   = down_x;
	p.down_y   = down_y;
	p.pad_x0   = pad_x0;
	p.pad_x1   = pad_x1;
	p.pad_y0   = pad_y0;
	p.pad_y1   = pad_y1;

	// out_dim = ceil((in_dim * upsample + paddings - (kernel_dim - 1)) /
	// downsample)
	p.out_h = ceil_div(p.in_h * p.up_y + p.pad_y0 + p.pad_y1 - (p.kernel_h - 1),
					   p.down_y);
	p.out_w = ceil_div(p.in_w * p.up_x + p.pad_x0 + p.pad_x1 - (p.kernel_w - 1),
					   p.down_x);

	// Prepare output tensor
	auto out = at::empty({p.n, 1, p.out_h, p.out_w}, x.options());

	// Select kernel to use
	int mode = -1;

	if (p.up_x == 1 && p.up_y == 1 && p.down_x == 1 && p.down_y == 1 &&
		p.kernel_h == 4 && p.kernel_w == 4) {
		mode = 1;
	} else if (p.up_x == 2 && p.up_y == 2 && p.down_x == 1 &&
			   p.down_y == 1 && p.kernel_h == 4 && p.kernel_w == 4) {
		mode = 2;
	} else if (p.up_x == 1 && p.up_y == 1 && p.down_x == 2 &&
			   p.down_y == 2 && p.kernel_h == 4 && p.kernel_w == 4) {
		mode = 3;
	}

	dim3 block_size;
	dim3 grid_size;

	// Calculate block and grid sizes
	if (mode > 0) {
		p.loop_n   = max(ceil_div(p.n, 16384), 1);
		block_size = dim3(256, 1, 1);
		grid_size  = dim3(ceil_div(p.out_w, 32), ceil_div(p.out_h, 32),
						  ceil_div(p.n, p.loop_n));
	} else {
		p.loop_n   = max(ceil_div(p.n, 16384), 8);
		block_size = dim3(32, 32, 1);
		grid_size =
			dim3(ceil_div(p.out_w, block_size.x),
				 ceil_div(p.out_h, block_size.y), ceil_div(p.n, p.loop_n));
	}

	// Dispatch kernel
	AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "upfirdn2d_cuda", [&] {
		switch (mode) {
		case 1:
			upfirdn2d_kernel<scalar_t, 1, 1, 1, 1, 4, 4, 32, 32>
				<<<grid_size, block_size, 0, stream>>>(
					out.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
					k.data_ptr<scalar_t>(), p);
			break;

		case 2:
			upfirdn2d_kernel<scalar_t, 2, 2, 1, 1, 4, 4, 32, 32>
				<<<grid_size, block_size, 0, stream>>>(
					out.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
					k.data_ptr<scalar_t>(), p);
			break;

		case 3:
			upfirdn2d_kernel<scalar_t, 1, 1, 2, 2, 4, 4, 32, 32>
				<<<grid_size, block_size, 0, stream>>>(
					out.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
					k.data_ptr<scalar_t>(), p);
			break;

		default:
			upfirdn2d_kernel_generic<scalar_t>
				<<<grid_size, block_size, 0, stream>>>(
					out.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
					k.data_ptr<scalar_t>(), p);
			break;
		}
	});

	return out;
}