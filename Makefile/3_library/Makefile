#hello.o는 라이브러리로 사용
CC=gcc
OBJS = main
#라이브러리로 만들 파일
#.c 와 .h 둘 다 사용할 것이기에 my_lib을 매크로로 해서
# $(LIBS).c  $(LIBS).h 로 사용
LIBS = my_lib

# <math.h> 를 사용하기 위한 옵션
FLAG = -lm

TARGET=hello

#라이브러리 사용 방식을 받을 매크로
#make 시 
#make LIB_OPTION=<옵션> 으로 해야한다
LIB_OPTION=


# 빌드는는
# make static : 정적
# make shared : 동적
# 으로 먼저 라이브러리를 생성하고 해야한다

default:
#옵션으로 SHARED 를 받았을 때
ifeq ($(LIB_OPTION), SHARED)
	@echo "SHARED"	
	$(CC) -c $(OBJS).c
	$(CC) -o $(TARGET) $(OBJS).o -L. -l$(LIBS)
	@echo "You need to export PATH to library"
	@echo "ex) export /PATH/TO/LIBRARY.so"
	@echo "or send libeary to /usr/lib"
else
#옵션으로 STATIC 을 받았을 때
ifeq ($(LIB_OPTION), STATIC)
	@echo "STATIC"	
	$(CC) -c $(OBJS).c
	$(CC) -o $(TARGET) $(OBJS).o -L. -l$(LIBS)  
#옵션이 없거나 잘못 되었을 때
else
	@echo "USAGE"
	@echo "make LIB_OPTION=<OPTION>"
	@echo "<OPTION> : SHARED | STATIC"
endif
endif

#정적 라이브러리 생성
static : $(LIBS).c $(LIBS).h
	$(CC) -c $(LIBS).c
	ar cr lib$(LIBS).a $(LIBS).o

#동적 라이브러리 생성
shared : $(LIBS).c $(LIBS).h
	$(CC) -c -fPIC $(LIBS).c
	$(CC) -shared -fPIC -o lib$(LIBS).so $(LIBS).o 
	

clean : 
	rm *.o
   
