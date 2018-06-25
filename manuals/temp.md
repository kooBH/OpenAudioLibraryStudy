# 자주 안써서 계속 까먹는 코드
+ time.h





## time.h

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
