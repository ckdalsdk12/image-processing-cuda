#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <sys/time.h>

// 두 시각 사이의 차를 구하기 위한 함수
void getGapTime(struct timeval* start_time, struct timeval* end_time, struct timeval* gap_time)
{
    gap_time->tv_sec = end_time->tv_sec - start_time->tv_sec;
    gap_time->tv_usec = end_time->tv_usec - start_time->tv_usec;
    if(gap_time->tv_usec < 0){
        gap_time->tv_usec = gap_time->tv_usec + 1000000;
        gap_time->tv_sec -= 1;
    }
}

// 시간 보정을 위한 함수
float timevalToFloat(struct timeval* time){
    double val;
    val = time->tv_sec;
    val += (time->tv_usec * 0.000001);
    return val;
}

// Jpg 파일을 그레이 스케일하는 커널 함수
__global__ void JpgToGray(unsigned char *img, int width, int height)
{
    unsigned char* pixelOffset;
    unsigned bytePerPixel = 3;

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // (i, j)번째 픽셀에 접근. 해당 픽셀은 R, G, B로 구성.
    if (tid < width * height)
    {
        pixelOffset = img + tid * bytePerPixel;
        unsigned char r = pixelOffset[0];
        unsigned char g = pixelOffset[1]; 
        unsigned char b = pixelOffset[2];
        unsigned char gray = r * 0.2126 + g * 0.7152 + b * 0.0722; // 그레이 스케일을 위한 공식 적용
        pixelOffset[0] = gray;
        pixelOffset[1] = gray;
        pixelOffset[2] = gray;
    }
}

int main(int argc, char *argv[])
{
    // 메모리 복사 및 GPU 연산 시작과 끝의 시간 저장을 위한 구조체 선언
    struct timeval htod_start, htod_end;
    struct timeval gpu_start, gpu_end;
    struct timeval dtoh_start, dtoh_end;

    // stb 라이브러리를 사용해 Jpg를 배열로 로드. Jpg 파일 이름은 argv[1]으로 받음.
    int width, height, channels;
    unsigned bytePerPixel = 3;
    unsigned char *img = stbi_load(argv[1], &width, &height, &channels, 0);
    unsigned char *d_img;

    int size = width*height*bytePerPixel*sizeof(char);
    int totalThread = width*height;
    int blockCount, threadCount;
    threadCount = 1024;
    blockCount = totalThread / threadCount;

    // 디바이스에 배열을 위한 메모리 할당
    cudaMalloc((void **)&d_img, size);

    // 호스트에서 디바이스로 배열 복사
    gettimeofday(&htod_start, NULL);
    cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);
    gettimeofday(&htod_end, NULL);

    // 그레이스케일 함수 호출
    gettimeofday(&gpu_start, NULL);
    JpgToGray<<<blockCount+1, threadCount>>>(d_img, width, height);
    // JpgToGray<<<65535,512>>(argv[1], argv[2]);
    cudaDeviceSynchronize();
    gettimeofday(&gpu_end, NULL);

    // 디바이스에서 호스트로 배열 복사
    gettimeofday(&dtoh_start, NULL);
    cudaMemcpy(img, d_img, size, cudaMemcpyDeviceToHost);
    gettimeofday(&dtoh_end, NULL);

    // stb 라이브러리를 사용해 배열을 Jpg로 저장. Jpg 파일 이름은 argv[2]으로 받음.
    stbi_write_jpg(argv[2], width, height, channels, img, 95);

    // 두 시각 사이의 차이 계산 및 출력
    struct timeval htod_gap, gpu_gap, dtoh_gap;
    getGapTime(&htod_start, &htod_end, &htod_gap);
    getGapTime(&gpu_start, &gpu_end, &gpu_gap);
    getGapTime(&dtoh_start, &dtoh_end, &dtoh_gap);
    
    float f_htod_gap = timevalToFloat(&htod_gap);
    float f_gpu_gap = timevalToFloat(&gpu_gap);
    float f_dtoh_gap = timevalToFloat(&dtoh_gap);
    float total_gap = f_htod_gap + f_gpu_gap + f_dtoh_gap;

    printf("GPU 버전 연산 시간 : %.6f + %.6f + %.6f = %.6f\n", f_htod_gap, f_gpu_gap, f_dtoh_gap, total_gap);

    stbi_image_free(img);
    cudaFree(d_img);

    return 0;
}