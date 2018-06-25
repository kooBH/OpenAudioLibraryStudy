
# CUDA<a name ="TOP"></a> 
1. [extention](#extention)
2. [function](#function)

nvcc --version <- CUDA compiler version check  
nvidia-smi <- GPU 사용량 


## 1. [extention](#TOP)<a name = "extention"></a>

### Function
+ 리턴은 void 

+ \_\_global\_\_
  * device(GPU)에서 실행  
  * host(CPU)에서 호출
  * \_\_global\_\_ functuon<<<number of block, thread per block >>>(args)    
  * 재귀 불가능
+ \_\_device\_\_
  * device에서 실행
  * device에서 호출
+ \_\_host\_\_
  * 기본값 : host 실행, host 호출 
  
### Variable

## 2. [function](#TOP)<a name="function"></a>

### Memory

cudaMalloc((void**)&대상포인터, 할당범위 )

```C++
int* device_pointer;
int array_size = 10;
cudaMalloc( (void**)&device_pointer,sizeof(int)*array_size);
```

cudaMemcpy(대상, 원본 , 크기 , 종류 )

종류 :   
+ cudaMemcpyHostToHost 	   : Host -> Host  
+ cudaMemcpyHostToDevice   :	Host -> Device  
+ cudaMemcpyDeviceToHost   :	Device -> Host  
+ cudaMemcpyDeviceToDevice :	Device -> Device   

```C++
int host_array[10] = {1,2,3,4,5,6,7,8,9,10};
int host_empty_array[10]={0,};

//host_array의 내용을 device_pointer로 
cudaMemcpy(device_pointer,host_array, sizeof(int) * array_size, cudaMemcpyHostToDevice);

//device_pointer의 내용을 host_empty_array로 
cudaMemcpy(host_empty_array,device_pointer, sizeof(int) * array_size, cudaMemcpyDeviceToHost);

// --> host_empty_array : {1,2,3,4,5,6,7,8,9,10}

```

cudaFree(대상포인터)

```C++
cudaFree(device_pointer)
```






