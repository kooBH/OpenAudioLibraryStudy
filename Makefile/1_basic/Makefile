#기본 타겟(가장 위에 있기 때문에) hello :  조건은 hello 와 main이 충족되어야한다
#조건이 맞다면 gcc -o <실행파일> <목적파일1> <목적파일2> 을 한다
exec : hello main
	gcc -o hello main.o  hello.o 

#main.o 를 만든다. main.c 가 있어야한다
main : main.c
	gcc -c main.c

#hello.o 를 만든다. hello.h 와 hello.c가 있어야한다
hello : hello.h hello.c
	gcc -c hello.c

#clean 타겟, make clean시 호출된다. 사용된 목적파일을 지운다
clean :
	rm *.o
