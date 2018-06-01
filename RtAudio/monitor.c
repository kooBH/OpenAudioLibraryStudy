#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#define STRLEN 200

//run record, reutnrn pid
int run_record(char*,int,int,int,double,char *);
//return wavs floder's recent modified minutes
int check_folder(char *);
//return current time's minutes
int current_minutes();
//kill record process by given pid
void kill_record(int);

int main(int argc, char* argv[])
{
	int pid=0;
	int cur_t=0,dir_t=0,gap =0;
	int i;
	

	if(argc<7)
	{
		printf("Not enough arguments, at least 5 (path , device, rate,channels,time, wd)\n");
		return -1;
	}
	for(i=0;i<argc;i++)
		printf("arg[%d] : %s \n",i,argv[i]);  

	pid = run_record(argv[1],atoi(argv[2]),atoi(argv[3]),atoi(argv[4]),atof(argv[5]),argv[6]);
	if(pid == 0)
	{
		printf("failed to run 'record'\n");
		return -1;
	}
	else
		printf("pid of record = %d\n",pid);

	sleep (120);
	/*monitoring
	Check Modified time of raw folder every 2 minutes
	If there is no modofication over 4 minutes
	kill 'record' process and run new one
	*/	
	while(1)
	{
		dir_t = check_folder(argv[1]);
		cur_t = current_minutes();
		gap = dir_t > cur_t ? 60 + cur_t - dir_t: cur_t - dir_t;
		printf("gap : %d | dir_t : %d | cur_t : %d\n",gap,dir_t,cur_t);

		if(gap > 3)
		{
			printf("Someone is not working\n");
			kill_record(pid);		
			pid = run_record(argv[1],atoi(argv[2]),atoi(argv[3]),atoi(argv[4]),atof(argv[5]),argv[6]);
		}
		sleep(120);	
	}	

	printf("Monitor exit\n ");
	return 0;

}

int run_record(char * path, int device_num,int rate,int channels,double time,char * wd)
{
	FILE *fp;
    char cmd[STRLEN];
	char tmp[STRLEN];
	char run[STRLEN];
	int pid;
	int found_flag = 0;
	//printf("stat : %d\n",system("gnome-terminal -x 'bash' -c  './p'  "));
 	
	int record_pid=0;
    char dot = '"';
	
	sprintf(run,"gnome-terminal -x 'bash'  -c '%s/record %c%s%c %d %d %d %f ;bash'",wd,dot,path,dot,device_num,rate,channels,time);		
	printf("run record : %s\n",run);
	//system("gnome-terminal -x 'bash'  -c './record'");
	system(run);
	fp = popen("ps -e", "r");
	/*
	 * PID		TTY		TIME		CMD
	 * 3775		pts/2	00:00:00	ps
	 * 3695     pts/2   00:00:00    record
	 * ...
	 * need to get PID of CMD 'record'
	 */	
	//First line
	if(fp == NULL)
	{
		printf("failed to popen(\"ps -e\")\n");
		return -1;
	}

	fscanf(fp,"%s%s%s%s",tmp,tmp,tmp,tmp);
	while( (fscanf(fp,"%d%s%s%[^\n]s",&pid,tmp,tmp,cmd) != EOF) && !found_flag )
	{
	//	printf("%s : %d\n",cmd,pid);
		
	//We read including spaces, cmd contains ' ' in front of it
		if(!strcmp(cmd," record"))
		{
			printf("PID of process 'record' is %d\n",pid);
			record_pid = pid;
			found_flag =1;
		}
	}
	if(!found_flag)
	{
		printf("Can't find 'record'\n");
		//scanf("%s");
	}

	//close cmd ps -e
	pclose(fp);

	return record_pid;	
}

int check_folder(char * path)
{
	FILE * fp;
	char tmp[STRLEN];
	char time[STRLEN];
	char folder[STRLEN];
	char cmd[STRLEN];
	int tmp_i;	
	int cur_t=0;
	
	//found_flag == 1
	
	sprintf(cmd,"ls -l '%s'",path);

	fp = popen(cmd, "r");
//	printf("cmd : %s\n",cmd);
	/*
	 total 180
	 -rw-rw-r-- 1 ffe ffe  12419  5월 24 11:18 CMakeCache.txt
	 drwxrwxr-x 7 ffe ffe   4096  5월 25 10:37 CMakeFiles
	 -rw-rw-r-- 1 ffe ffe   1380  5월 24 16:10 cmake_install.cmake
	 -rwxrwxr-x 1 ffe ffe 115656  5월 24 16:10 librtaudio.so
	 -rw-rw-r-- 1 ffe ffe   6941  5월 25 10:37 Makefile
	 -rwxrwxr-x 1 ffe ffe   8992  5월 25 10:31 monitor
	 drwxrwxr-x 2 ffe ffe   4096  5월 25 10:40 raws
	 -rwxrwxr-x 1 ffe ffe  15040  5월 25 10:37 record
	 * */
	//total xxx
	if(fp == NULL)
	{
		printf("failed to popen(\"ls -l\")\n");	
	}
	
	fscanf(fp,"%s%s",tmp,tmp);
	while(fscanf(fp,"%s%s%s%s%s%s%s%s%[^\n]s",tmp,tmp,tmp,tmp,tmp,tmp,tmp,time,folder) !=EOF)
	{
		if(!strcmp(" wavs",folder))
		{
		//	printf("%s : %s\n",folder,time);	
			break;
		}	
	}
	sscanf(time,"%d:%d",&tmp_i,&cur_t);
	//clsoe cmd ls -la		
	pclose(fp);

	return cur_t;

}

int current_minutes()
{
	time_t raw_time;
	struct tm * info_time;
	time(&raw_time);
	info_time = localtime(&raw_time);

	return info_time->tm_min;
}

void kill_record(int pid)
{
	char cmd[STRLEN];
	sprintf(cmd,"kill -9 %d",pid);
	system(cmd);
}
