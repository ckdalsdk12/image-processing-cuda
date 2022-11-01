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

// Jpg 파일을 좌우 플립 하는 함수
void LRFlip(unsigned char *img, int width, int height)
{
    unsigned bytePerPixel = 3;
    unsigned char* pixelOffset;
    unsigned char* toOffset;
    unsigned char temp[3];

    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width/2; i++)
        {
            // (i, j)번째 픽셀에 접근. 해당 픽셀은 R, G, B로 구성.
            pixelOffset = img + (i + width * j) * bytePerPixel;
            toOffset = img + ((j * width) + (width - (i+1))) * bytePerPixel;
            for (int index = 0; index < 3; index++)
            {
                temp[index] = toOffset[index];
                toOffset[index] = pixelOffset[index];
                pixelOffset[index] = temp[index];
            }
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

    // LRFlip 함수 호출
    gettimeofday(&cpu_start, NULL);
    LRFlip(img, width, height);
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