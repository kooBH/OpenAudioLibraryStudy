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

**주의** Visual Studio 에서는 안됨 CLOCK_MONOTONIC 이 없다.

일반적인 clock() 를 사용한 스탑워치는 cpu clock 을 기준으로 헤아리기 때문에  
openMP 같은 다중 쓰레딩을 사용했을 때, 시간측정이 잘 되지 않는다 

윈도우의 경우에는 [참고](https://stackoverflow.com/questions/5404277/porting-clock-gettime-to-windows)

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

## [하위폴더에서 파일찾아서 경로 받기](#TOP)<a name ="tree"></a>

```bash
tree - ifpugDs | grep <filename>
```
