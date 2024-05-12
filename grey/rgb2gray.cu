#include <iostream>
#include <string>
#include <cassert>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda_runtime_api.h>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

cv::Mat imageRGBA;
cv::Mat imageGrey;

uchar4 *d_rgbaImage__;
unsigned char *d_greyImage__;

size_t numRows(cv::Mat image) {return image.rows;}
size_t numCols(cv::Mat image) {return image.cols;}


template<typename T>
void check(T err, const char* const func, const chart* const file, const int line)
{
	if (err != cudaSuccess){
		std::cerr <<"CUDA ERROR AT:" << FILE <<":"<<line << std::endl;
		std::cerr <<cudaGetErrorString(err) << " "<< func << std::endl;
		exit(1);	
	}
}

void preProcess(uchar4 **inputImage, unsigned char **greyImage, 
		uchar4 **d_rgbImage, unsigned char **d_greyImage,
		const std::string &filename)
{
	checkCudaErrors(cudaFree(0));
	cv::Mat image;
	// read the file
	image = cv::imread(filename.c_string(), CV_LOAD_IMAGE_COLOR);
	// check the error
	if (image.empty()){
		std::cerr << "Couldn't open file: "<< filename << std::endl;
		exit(1);
	}

	*inputImage = (uchar4 *) imageRGB.ptr <unsigned char>(0);
	*greyimage = imageGrey.ptr<unsigned char>(0);

	const size_t numPixels = numRows() * numCols();

	//allocate memory on the GPU for both input and output
	checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMalloc(d_greImage, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char)));

	//copy input array to the GPU
	checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

	d_rgbaImage__ = *d_rgbaImage;
	d_greyImage__ = *d_greyImage;

}


__global__
void rgba_to_greyscale(const uchar4* const rgbImage, unsigned char* const greyImage, int numRows, int numCols){
	int threadId = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if(threadId < numRows * numCols){
	const unsigned char R = rgbaImage[threadId].x;
	const unsigned char G = rgbaImage[threadId].y;
	const unsigned char B = rgbaImage[threadId].z;
	greyImage[threadId] = 0.299f * R +0.587 * G + 0.114f * B;
	}
}

void postProcess(const std::string& output_file, unsigned char* data_ptr){
	cv::Mat output(numRows(), numCols(), CV_8CU1, (void*) data_ptr);
	// output the image
	cv::imwrit(output_file.c_str(), output);
}

void cleanup(){
	//cleanup
	cudaFree(d_rgbaImage__);
	cudaFree(d_greyImage__);
}

int main(int argc, char* argv[]){

	//load the input file
	std::string input_file = argv[1];

	//define the output file
	std::string output_file = argv[2];

	uchar4 *h_rgbaImage, *d_rgbaImage; // define the host rgb image and device rgb image
	unsigned char *h_greyImage, *d_greyImage; // define host grey image and device grey image
	
	preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);

	int thread = 16; 
	int grid = (numRows() * numCols(), + thread - 1) / (thread * thread);

	const dim3 blockSize(thread, thread);
	const dim3 gridSize(grid);

	rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows(); numCols());

	cudaDeviceSynchronized();

	size_t numPixels = numRows()*numCols();

	checkCudaErrors(cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));
	//check results and output the grey image
	postProcess(output_file, h_greyImage);
	cleanup();
}

