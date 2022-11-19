# image-processing-cuda
CUDA C (GPU) / C (CPU)를 사용한 JPG 이미지 처리 프로그램

이미지를 좌우 반전 시키거나 그레이스케일 변환할 수 있습니다.

## 프로그램 설명
CUDA C / C와 stb 라이브러리(<https://github.com/nothings/stb>)를 사용하여  
이미지를 불러와 좌우 반전 또는 그레이스케일 변환하고 JPG로 저장하는 프로그램입니다.

본 프로그램은 리눅스 환경에서만 구동 가능합니다.

사용할 이미지 처리와 GPU / CPU 사용 여부에 따라 총 4가지 버전이 있습니다.

* flip_gpu.cu : 좌우 반전 - GPU 버전
* flip_cpu.cu : 좌우 반전 - CPU 버전
* grayscale_gpu.cu : 그레이 스케일 변환 - GPU 버전
* grayscale_cpu.cu : 그레이 스케일 변환 - CPU 버전

## 사용 방법
### 1. CUDA Toolkit 설치
[CUDA Toolkit 다운로드 페이지](https://developer.nvidia.com/cuda-downloads)에서 환경에 맞게 CUDA Toolkit을 다운로드하고 지시에 따라 설치합니다.

### 2. 소스 코드 다운로드 및 컴파일
저장소에서 소스 코드를 다운로드하고 해당 디렉토리에서 터미널을 연 뒤 nvcc 컴파일러로 컴파일합니다.  
```nvcc -o 프로그램명 소스코드명```명령으로 컴파일합니다.  
* 예시 : ```nvcc -o flip flip_gpu.cu```

### 3. 프로그램 실행
입력 이미지를 같은 디렉토리에 위치시키고 터미널에서 프로그램을 실행합니다.  
```./프로그램명 입력.jpg 출력.jpg```명령으로 실행합니다.  
* 예시 : ```./flip input.jpg output.jpg```

프로그램을 실행하면 터미널에 연산에 걸린 시간을 출력해 줍니다.

GPU 버전은 ```GPU 버전 연산 시간 : A + B + C = X```의 형태로 연산에 걸린 시간을 표기해 주고  
CPU 버전은 ```CPU 버전 연산 시간 : X```의 형태로 연산에 걸린 시간을 표기해 줍니다.

만약 연산 시간 출력이 필요하지 않은 경우  
```printf("GPU 버전 연산 시간 : %.6f + %.6f + %.6f = %.6f\n", f_htod_gap, f_gpu_gap, f_dtoh_gap, total_gap);``` 또는  
```printf("CPU 버전 연산 시간 : %.6f\n", f_cpu_gap);```  
코드를 주석 처리해 주시기 바랍니다.

### 4. 출력 이미지 확인
디렉터리에 저장된 이미지가 올바른지 확인합니다.
