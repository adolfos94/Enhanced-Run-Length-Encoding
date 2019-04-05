#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
//
//  main.cu
//  To compile the program:
//    nvcc main.cu -std=c++11 --expt-extended-lambda
//  Elegant Pairing Function
//
//  Created by Adolfo Solís on 4/4/19.
//  Copyright © 2019 Adolfo Solís. All rights reserved.
//

#include <iostream>
#include <math.h>
#include <omp.h>
#include <vector>
#define SIZE 10000
using namespace std;

struct RLE {
  int x;
  int y;
};

struct TIMER {
  float CPU;
  float GPU;
};

int elegantPair(int x, int y) {
  x = x >= 0 ? x * 2 : (x * -2) - 1;
  y = y >= 0 ? y * 2 : (y * -2) - 1;

  return (x >= y) ? (pow(x, 2) + x + y) : (pow(y, 2) + x);
}

vector<int> elegantUnpair(int z) {
  vector<int> tuple;

  int sqrtz = floor(sqrt(z));
  int sqz = sqrtz * sqrtz;

  if ((z - sqz) >= sqrtz) {
    tuple.push_back(sqrtz);
    tuple.push_back(z - sqz - sqrtz);
  } else {
    tuple.push_back(z - sqz);
    tuple.push_back(sqrtz);
  }

  tuple[0] = fmod(tuple[0], 2) == 0 ? tuple[0] / 2 : (tuple[0] + 1) / -2;
  tuple[1] = fmod(tuple[1], 2) == 0 ? tuple[1] / 2 : (tuple[1] + 1) / -2;

  return tuple;
}
// GPU Functions
thrust::device_vector<int> gpuEncoding(thrust::device_vector<RLE> rle) {
  thrust::device_vector<int> arrayCompressed(rle.size());

  // GPU - Elegant Pair Function
  auto gpuElegantPair = [=] __device__(RLE array) {
    int x = array.x;
    int y = array.y;
    x = x >= 0 ? x * 2 : (x * -2) - 1;
    y = y >= 0 ? y * 2 : (y * -2) - 1;

    return (x >= y) ? ((x * x) + x + y) : ((y * y) + x);
  };

  thrust::transform(rle.begin(), rle.end(), arrayCompressed.begin(),
                    gpuElegantPair);

  return arrayCompressed;
}

thrust::device_vector<RLE>
gpuDecoding(thrust::device_vector<int> arrayCompressed) {
  thrust::device_vector<RLE> rle(arrayCompressed.size());

  // GPU - Elegant Unpair Function
  auto gpuElegantUnpair = [=] __device__(int z) {
    RLE tuple;

    int sqrtz = floor(sqrt(z));
    int sqz = sqrtz * sqrtz;

    if ((z - sqz) >= sqrtz) {
      tuple.x = sqrtz;
      tuple.y = z - sqz - sqrtz;
    } else {
      tuple.x = z - sqz;
      tuple.y = sqrtz;
    }

    tuple.x = tuple.x % 2 == 0 ? tuple.x / 2 : (tuple.x + 1) / -2;
    tuple.y = tuple.y % 2 == 0 ? tuple.y / 2 : (tuple.y + 1) / -2;

    return tuple;
  };

  thrust::transform(arrayCompressed.begin(), arrayCompressed.end(), rle.begin(),
                    gpuElegantUnpair);

  return rle;
}

// CPU Functions
vector<int> cpuEncode(int *rle_1, int *rle_2, int size) {
  vector<int> arrayCompressed;
  for (int index = 0; index < size; index++) {
    arrayCompressed.push_back(elegantPair(rle_1[index], rle_2[index]));
  }
  return arrayCompressed;
}

vector<vector<int>> cpuDecode(vector<int> arrayCompressed) {
  vector<vector<int>> rle;
  for (int index = 0; index < arrayCompressed.size(); index++) {
    vector<int> tuple = elegantUnpair(arrayCompressed[index]);
    rle.push_back(tuple);
  }
  return rle;
}

float differentElements(vector<vector<int>> CPU, thrust::host_vector<RLE> GPU) {
  int diff = 0;
  for (int i = 0; i < CPU.size(); ++i) {
    if (CPU[i][0] != GPU[i].x || CPU[i][1] != GPU[i].y)
      ++diff;
  }
  return diff * 100 / CPU.size();
}

TIMER differenceExecTime(float cpu, float gpu) {
  TIMER result;
  float mayor = cpu > gpu ? cpu : gpu;
  float menor = cpu < gpu ? cpu : gpu;

  result.CPU = 100 - (menor * 100 / mayor);
  result.GPU = 100 - (menor * 100 / mayor);

  return result;
}

void whoWins(TIMER times) {
  if (times.CPU > times.GPU)
    cout << "CPU WINS! : " << times.CPU << "%%" << endl;
  else
    cout << "GPU WINS! : " << times.GPU << "%%" << endl;
}

int main(int argc, const char *argv[]) {
  cudaSetDevice(1);
  TIMER timer;
  srand((int)time(NULL));

  // Initialize the cuda timers
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;

  int *rle_1;
  int *rle_2;

  rle_1 = (int *)malloc(sizeof(int) * SIZE);
  rle_2 = (int *)malloc(sizeof(int) * SIZE);

  // Generate CPU array of size 'SIZE'
  thrust::host_vector<RLE> rle(SIZE);

  // Initialize Vectors CPU
  for (int i = 0; i < SIZE; i++) {
    rle[i].x = rle_1[i] = rand() % 100;
    rle[i].y = rle_2[i] = rand() % 100;
  }

  // Copy CPU vectors to GPU
  thrust::device_vector<RLE> d_rle = rle;

  // Compress on GPU
  cout << "Compressing GPU.." << endl;
  cudaEventRecord(start);
  thrust::device_vector<int> arrayCompressedDevice = gpuEncoding(d_rle);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  float timer_compress_gpu = milliseconds;
  cout << "GPU time compress: " << milliseconds << endl;
  // Copy GPU vectors to CPU
  thrust::host_vector<int> arrayCompressedHost = arrayCompressedDevice;

  // for (int i = 0; i < arrayCompressedHost.size(); i++) {
  //   cout << arrayCompressedHost[i] << endl;
  // }

  // // Decompress on GPU
  cout << "Decompressing GPU.." << endl;
  cudaEventRecord(start);
  thrust::device_vector<RLE> res_rle_gpu = gpuDecoding(arrayCompressedDevice);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  float timer_decompress_gpu = milliseconds;
  cout << "GPU time decompress: " << milliseconds << endl;

  // Copy GPU vectors to CPU
  thrust::host_vector<RLE> arrayDecompressedHost = res_rle_gpu;

  // for (int i = 0; i < arrayDecompressedHost.size(); i++) {
  //   cout << arrayDecompressedHost[i].x << ", " << arrayDecompressedHost[i].y
  //        << endl;
  // }

  // Compress on CPU
  cout << "Compressing CPU.." << endl;
  cudaEventRecord(start);
  vector<int> arrayCompressed = cpuEncode(rle_1, rle_2, SIZE);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  float timer_compress_cpu = milliseconds;
  cout << "CPU time compress: " << milliseconds << endl;

  // for (int index = 0; index < arrayCompressed.size(); index++) {
  //   cout << arrayCompressed[index] << endl;
  // }

  cout << "Decompressing CPU.." << endl;
  cudaEventRecord(start);
  vector<vector<int>> res_rle = cpuDecode(arrayCompressed);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  float timer_decompress_cpu = milliseconds;
  cout << "CPU time decompress: " << milliseconds << endl;

  // for (int index = 0; index < res_rle.size(); index++) {
  //   cout << res_rle[index][0] << ", " << res_rle[index][1] << endl;
  // }

  cout << "Percentage of different elements: "
       << differentElements(res_rle, arrayDecompressedHost) << endl;

  cout << "Compression CPU vs GPU..." << endl;
  timer = differenceExecTime(timer_compress_cpu, timer_compress_gpu);
  whoWins(timer);

  cout << "Decompression CPU vs GPU..." << endl;
  timer = differenceExecTime(timer_decompress_cpu, timer_decompress_gpu);
  whoWins(timer);

  free(rle_1);
  free(rle_2);

  return 0;
}
