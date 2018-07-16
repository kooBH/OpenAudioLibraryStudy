# 자주 안써서 계속 까먹는 코드<a name = "TOP"></a>
+ [time.h](#time)
  * [날짜](#date)
  * [스탑워치](#stopwatch)
+ [하위폴더에서 파일찾아서 경로 받기](#tree)

## [time.h](#TOP)<a name ="time"></a>

### [날짜](#TOP)<a name = "date">
	
```C++
#include <stdio.h>
#include <time.h>

int main ()
{
  time_t rawtime;
  struct tm * timeinfo;

  time ( &rawtime );
  timeinfo = localtime ( &rawtime );
  printf ( "Current local time and date: %s", asctime (timeinfo) );

  return 0;
}

```

Member	|Type	|Meaning|	Range
---|---|---|---
tm_sec	|int	|seconds after the minute	|0-61(C++11 이상은 60)
tm_min	|int	|minutes after the hour	|0-59
tm_hour	|int	|hours since midnight	|0-23
tm_mday|	int	|day of the month|	1-31
tm_mon	|int	|months since January|	0-11
tm_year	|int	|years since 1900	
tm_wday	|int|	days since Sunday	|0-6
tm_yday	|int|	days since January 1|	0-365
tm_isdst	|int	|Daylight Saving Time flag	


### [millisecond stopwatch](#TOP)<a name ="stopwatch"></a>



일반적인 clock() 를 사용한 스탑워치는 cpu clock 을 기준으로 측정하기 때문에  
openMP 같은 다중 쓰레딩을 사용했을 때, 시간측정이 잘 되지 않는다 

<details><summary>stopwatch for linux</summary>
	
```C++
stopwatch(0);
   //작업 
stopwatch(1);
```

```C++
void stopwatch(int flag)
{
	enum clock_unit{nano = 0, micro , milli, sec} unit;
	
	const long long NANOS = 1000000000LL;
	static struct timespec startTS,endTS;
	static long long diff = 0;

	/*
		여기서 단위 조정
		nano, micro, milli, sec
	*/
	unit = micro;

	//start
	if(flag == 0)
	{
		diff = 0;
		if(-1 == clock_gettime(CLOCK_MONOTONIC,&startTS))
			printf("Failed to call clock_gettime\n");
	}
	//end
	else if(flag == 1)
	{		
		if(-1 == clock_gettime(CLOCK_MONOTONIC,&endTS))
			printf("Failed to call clock_gettime\n");
		diff = NANOS * (endTS.tv_sec - startTS.tv_sec) + (endTS.tv_nsec - startTS.tv_nsec);

		switch(unit)		
		{
			case nano :
				printf("elapsed time : % lld nano sec\n",diff);
			break;
			case micro :
				printf("elapsed time : % lld micro sec\n",diff/1000);
			break;
			case sec :
				printf("elapsed time : % lld sec\n",diff/1000000000);
			break;
			default :
				printf("elapsed time : % lld milli sec\n",diff/100000);
			break;	

		}
	}
	else
	{
		printf("wrong flag | 0 : start, 1 : end\n");
	}

}

 
```
</details>

윈도우의 경우 마이크로 초 까지는 되나, 나노초 까지는 지원하지 않는 거 같다.(장확하게 나노초를 해주는 방법을 아직 못 찾음)
[참고 링크](https://stackoverflow.com/questions/5404277/porting-clock-gettime-to-windows)


<details><summary>stopwatch_for_windows</summary>

```C
#include <time.h>
#include <windows.h>

LARGE_INTEGER getFILETIMEoffset()
{
SYSTEMTIME s;
FILETIME f;
LARGE_INTEGER t;
s.wYear = 1970;
s.wMonth = 1;
s.wDay = 1;
s.wHour = 0;
s.wMinute = 0;
s.wSecond = 0;
s.wMilliseconds = 0;
SystemTimeToFileTime(&s, &f);
t.QuadPart = f.dwHighDateTime;
t.QuadPart <<= 32;
t.QuadPart |= f.dwLowDateTime;
return (t);
}

int
clock_gettime( struct timeval *tv)
{
LARGE_INTEGER           t;
FILETIME    		        f;
double                  microseconds;
static LARGE_INTEGER    offset;
static double           frequencyToMicroseconds;
static int              initialized = 0;
static BOOL							usePerformanceCounter = 0;
if (!initialized)
{
	LARGE_INTEGER performanceFrequency;
	initialized = 1;
	usePerformanceCounter = QueryPerformanceFrequency(&performanceFrequency);
	if (usePerformanceCounter)
 	{
		QueryPerformanceCounter(&offset);
		frequencyToMicroseconds = (double)performanceFrequency.QuadPart / 1000000.;
	}
 	else
 	{
		offset = getFILETIMEoffset();
		frequencyToMicroseconds = 10.;
	}
}
if (usePerformanceCounter)
	QueryPerformanceCounter(&t);
else
{
	GetSystemTimeAsFileTime(&f);
	t.QuadPart = f.dwHighDateTime;
	t.QuadPart <<= 32;
	t.QuadPart |= f.dwLowDateTime;
}
t.QuadPart -= offset.QuadPart;
microseconds = (double)t.QuadPart / frequencyToMicroseconds;
t.QuadPart = microseconds;
tv->tv_sec = t.QuadPart / 1000000;
tv->tv_usec = t.QuadPart % 1000000;
return (0);
}

void stopwatch(int flag)
{
	static struct timeval startTV, endTV;
	static long long diff;
	const long long MICRO = 1000000LL;

	if(flag == 0)
		clock_gettime(&startTV);
	else
	{
		clock_gettime(&endTV);
		diff = MICRO *(endTV.tv_sec - startTV.tv_sec) + (endTV.tv_usec - startTV.tv_usec);
		printf("elapsed time : %lld micro seconds\n",diff);
	}

}

```

</details>


## [하위폴더에서 파일찾아서 경로 받기](#TOP)<a name ="tree"></a>

```bash
tree - ifpugDs | grep <filename>
```
