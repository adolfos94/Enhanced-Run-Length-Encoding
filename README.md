# Enhanced Run Length Encoding

A parallel data compression algorithm for store or transmission purposes based on run length encoding and elegant pairing function. To achieve higher compression ratios, the proposed method encodes the run length encoding matrix through a pairing function. Because a pairing function is a unique and a bijective function, it is possible to recover the data without losing information. This implementation is really fast using vectors with high sizes. 

- An Enhanced Run Length Encoding for Image
Compression based on Discrete Wavelet Transform: 
- RLE: https://en.wikipedia.org/wiki/Run-length_encoding
- Elegant Pairing Function: http://szudzik.com/ElegantPairing.pdf

# CUDA 

This implementation uses the template library of CUDA and lambdas of C++. Also, we implemented a struct for RLE.

```cpp
    thrusth::

    #include <thrust/device_vector.h>
    #include <thrust/host_vector.h>
    #include <thrust/transform.h>
```

```cpp
    struct RLE {
        int x; // Value
        int y; // Number of repetitions
    };
```


## Example of use

1. Compile
```bash
    nvcc main.cu -std=c++11 --expt-extended-lambda
```
2. Generate CPU array of size 'SIZE'
```cpp
    #define SIZE 10000

    thrust::host_vector<RLE> rle(SIZE);
```
3. Initialize host_vector. **Import your RLE vectors**. For example purposes the rle vector is filled with rand values.
```cpp
  for (int i = 0; i < SIZE; i++) {
    rle[i].x = rand() % 100;
    rle[i].y = rand() % 100;
  }
```

4. Define a device_vector containing the run length encoder
```cpp
    thrust::device_vector<RLE> d_rle = rle;
```
5. Compress on GPU
```cpp
    thrust::device_vector<int> arrayCompressedDevice =  gpuEncoding(d_rle);
```


6. Copy the GPU to CPU for store or transmission purposes.
```cpp
    thrust::host_vector<int> arrayCompressedHost = arrayCompressedDevice;
```

7. Decompression on GPU

```cpp
    thrust::device_vector<RLE> res_rle_gpu = gpuDecoding(arrayCompressedDevice);
```

8. Copy the GPU vector to CPU. Since, this algorithm is a lossless compression algorithm, the decoded vector must be similar to the original vector.

```cpp
    thrust::host_vector<RLE> arrayDecompressedHost = res_rle_gpu;
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Authors and References

- Article "An Enhanced Run Length Encoding for Image
Compression based on Discrete Wavelet Transform" 
    *

- Proposed Enhanced Run Length Encoding:
    * [Adolfo Solis-Rosas](adolfo2794@gmail.com)
- Elegant Pairing Function
    * [Matthew Szudzik](http://szudzik.com/ElegantPairing.pdf)