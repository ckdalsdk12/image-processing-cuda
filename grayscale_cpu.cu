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

// Jpg 파일을 그레이 스케일하는 함수
void JpgToGray(unsigned char *img, int width, int height)
{
    unsigned bytePerPixel = 3;
    unsigned char* pixelOffset;

    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            // (i, j)번째 픽셀에 접근. 해당 픽셀은 R, G, B로 구성.
            pixelOffset = img + (i + width * j) * bytePerPixel;
            unsigned char r = pixelOffset[0];
            unsigned char g = pixelOffset[1]; 
            unsigned char b = pixelOffset[2];
            unsigned char gray = r * 0.2126 + g * 0.7152 + b * 0.0722; // 그레이 스케일을 위한 공식 적용
            pixelOffset[0] = gray;
            pixelOffset[1] = gray;
            pixelOffset[2] = gray;
        }
    }
}

int main(int argc, char *argv[])
{
    // CPU 연산 시작과 끝의 시간 저장을 위한 구조체 선언
    struct timeval cpu_start, cpu_end;

    // stb 라이브러리를 사용해 Jpg를 배열로 로드. Jpg 파일 이름은 argv[1]으로 받음.
    int width, height, channels;
    unsigned char *img = stbi_load(argv[1], &width, &height, &channels, 0);

    // 그레이스케일 함수 호출
    gettimeofday(&cpu_start, NULL);
    JpgToGray(img, width, height);
    // JpgToGray(argv[1], argv[2]);
    gettimeofday(&cpu_end, NULL);

    // stb 라이브러리를 사용해 배열을 Jpg로 저장. Jpg 파일 이름은 argv[2]으로 받음.
    stbi_write_jpg(argv[2], width, height, channels, img, 95);

    // 두 시각 사이의 차이 계산 및 출력
    struct timeval cpu_gap;
    getGapTime(&cpu_start, &cpu_end, &cpu_gap);
    float f_cpu_gap = timevalToFloat(&cpu_gap);

    printf("CPU 버전 연산 시간 : %.6f\n", f_cpu_gap);
    
    stbi_image_free(img);

    return 0;
}