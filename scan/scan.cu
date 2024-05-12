#include <stdio.h>

__global__ void global_scan(float * d_out, float * d_in) 
{
	int idx = threadIdx.x;
	float out = 0.00f;
	d_out[idx] = d_in[idx];

	__syncthreads();

	for(int ind = 1; ind < sizeof(d_in); ind *= 2)
	{
		if(idx - ind >= 0)
		{
			out = d_out[idx] + d_out[idx-ind];
		}

		__syncthreads();

		if (idx - ind >= 0)
		{
			d_out[idx] = out;
			out = 0.00f;
		}

	}
	
}


int main(int argc, char** argy)
{
	const int ARRAY_SIZE = 8;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	float h_in[ARRAY_SIZE];
	for(int i=0; i<ARRAY_SIZE; i++)
	{
		h_in[i] = float(i);
	}

	float h_out[ARRAY_SIZE];

	float * d_in;
	float * d_out;

	cudaMalloc((void **) &d_in, ARRAY_BYTES);
	cudaMalloc((void **) &d_out, ARRAY_BYTES);

	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	global_scan<<<1, ARRAY_SIZE>>>(d_out, d_in);

	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	for(int i=0; i<ARRAY_SIZE; i++)
	{
		printf("%f", h_out[i]);
	}

	cudaFree(d_in);
	cudaFree(d_out);
	return 0;
}
