
# CUDA<a name ="TOP"></a> 

nvcc --version <- CUDA compiler version check  
nvidia-smi <- GPU 사용량 


1. [extention](#extension)

+ \_\_global\_\_
  * 디바이스에서 실행  
  * 커널함수에 지정
  * \_\_global\_\_ functuon<<<number of block, thread per block >>>(arg)
  * 리턴은 void
  * 재귀 불가
  * static 변수 포함 불가
  * 가변형 인수 붉ㅏ
  * \_\_host\_\_ 공용불가
  * 공유메모리 이용
+ \_\_device\_\_
  * 다바이스에서 실행







