# cublasStatus_t

Value | Meaning
--- | ---
CUBLAS_STATUS_SUCCESS| 성공
CUBLAS_STATUS_NOT_INITIALIZED |초기화 되지 않음, cublasCreate()를 먼저 해줘야한다
CUBLAS_STATUS_ALLOC_FAILED | 할당 실패, cudaMalloc()이 제대로 되지 않았다. 메모리 해제 요망
CUBLAS_STATUS_INVALID_VALUE |함수에 유효한 인자가 전달되지 않았다. 인자의 타입을 확인 요망
CUBLAS_STATUS_ARCH_MISMATCH | 현재 장치에선 지원해지 않는 기능사용, 보통 double precision에서 발생
CUBLAS_STATUS_MAPPING_ERROR |GPU메모리 접근실패. texture 메모리 해제 요망
CUBLAS_STATUS_EXECUTION_FAILED |커널 함수 호출 실패. 드라이버 버전이나 라이브러리 확인 요망
CUBLAS_STATUS_INTERNAL_ERROR | 내부 cublas 실패. 드라이버 버전이나 하드웨어 또는 할당해제된 변수에 접근하지는 확인 바람
CUBLAS_STATUS_NOT_SUPPORTED |지원하지 않음
CUBLAS_STATUS_LICENSE_ERROR |The functionnality requested requires some license and an error was detected when trying to check the current licensing. This error can happen if the license is not present or is expired or if the environment variable NVIDIA_LICENSE_FILE is not set properly. 

# cudaError_t

Value | Meaing
---|---
cudaErrorSyncDepthExceeded = 68 |    This error indicates that a call to cudaDeviceSynchronize made from the device runtime failed because the call was made at grid depth greater than than either the default (2 levels of grids) or user specified device limit cudaLimitDevRuntimeSyncDepth. To be able to synchronize on launched grids at a greater depth successfully, the maximum nested depth at which cudaDeviceSynchronize will be called must be specified with the cudaLimitDevRuntimeSyncDepth limit to the cudaDeviceSetLimit api before the host-side launch of a kernel using the device runtime. Keep in mind that additional levels of sync depth require the runtime to reserve large amounts of device memory that cannot be used for user allocations. 
cudaErrorLaunchPendingCountExceeded = 69  |      This error indicates that a device runtime grid launch failed because the launch would exceed the limit cudaLimitDevRuntimePendingLaunchCount. For this launch to proceed successfully, cudaDeviceSetLimit must be called to set the cudaLimitDevRuntimePendingLaunchCount to be higher than the upper bound of outstanding launches that can be issued to the device runtime. Keep in mind that raising the limit of pending device runtime launches will require the runtime to reserve device memory that cannot be used for user allocations. 
cudaErrorNotPermitted = 70  |      This error indicates the attempted operation is not permitted. 
cudaErrorNotSupported = 71 |      This error indicates the attempted operation is not supported on the current system or device. 
cudaErrorHardwareStackError = 72     |        Device encountered an error in the call stack during kernel execution, possibly due to stack corruption or exceeding the stack size limit. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched. 
cudaErrorIllegalInstruction = 73    |        The device encountered an illegal instruction during kernel execution This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched. 
cudaErrorMisalignedAddress = 74    |        The device encountered a load or store instruction on a memory address which is not aligned. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched. 
cudaErrorInvalidAddressSpace = 75    |        While executing a kernel, the device encountered an instruction which can only operate on memory locations in certain address spaces (global, shared, or local), but was supplied a memory address not belonging to an allowed address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched. 
cudaErrorInvalidPc = 76    |        The device encountered an invalid program counter. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched. 
cudaErrorIllegalAddress = 77    |        The device encountered a load or store instruction on an invalid memory address. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched. 
cudaErrorInvalidPtx = 78    |        A PTX compilation failed. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device. 
cudaErrorInvalidGraphicsContext = 79     |        This indicates an error with the OpenGL or DirectX context. 
cudaErrorNvlinkUncorrectable = 80     |        This indicates that an uncorrectable NVLink error was detected during the execution. 
cudaErrorJitCompilerNotFound = 81    |        This indicates that the PTX JIT compiler library was not found. The JIT Compiler library is used for PTX compilation. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device. 
cudaErrorCooperativeLaunchTooLarge = 82    |        This error indicates that the number of blocks launched per grid for a kernel that was launched via either cudaLaunchCooperativeKernel or cudaLaunchCooperativeKernelMultiDevice exceeds the maximum number of blocks as allowed by cudaOccupancyMaxActiveBlocksPerMultiprocessor or cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors as specified by the device attribute cudaDevAttrMultiProcessorCount. 
cudaErrorStartupFailure = 0x7f    |        This indicates an internal startup failure in the CUDA runtime. 
cudaErrorApiFailureBase = 10000    |        Deprecated    This error return is deprecated as of CUDA 4.1.    Any unhandled CUDA driver error is added to this value and returned via the runtime. Production releases of CUDA should not return such errors.
