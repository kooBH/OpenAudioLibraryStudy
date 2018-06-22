#wildcard 는 위치에서 조건에 맞는 모든 파일을 받는다
SRC = $(wildcard */*.c)
#이 매크로는 SRC에 현재 위치에서 하위 디렉토리의 모든 .c파일을 받는다(경로포함)
OBJ = $(SRC:.c=.o)
INC = -Ihead

#.PHONY 타겟은 실제론 수행하지 않는다
# 만일 all 이나 clean라는 파일이 생긴다면 해당 이름의 타겟이 수행되지 않음에도 수행되었다고 판단할 수 있기 때문에, 명령이란 것을 명시한다
.PHONY: all clean

all:$(OBJ) main.o 
#	@echo "$(SRC)"
#	@echo "$(OBJ)"
#	$(notdir ) 은 이름에서 주소 부분을 제거한다
#	$(notdir d1/c1.c) -> c1.c
#	반대로 dir은 이름에서 주소부분만 가져온다
#   $(dir d1/c1.c) -> d1/
	gcc -o hello $(notdir $^)
	rm *.o

$(OBJ):
#	@echo "target :  $@ "
#	$(대상:A=B) 는 대상에 있는 문자열 A를 B로 바꾼다
#	$(c1.o:.o=.c) -> c1.c
	gcc -c $(INC) $(@:.o=.c)

main.o: main.c
	gcc -c $(INC) main.c

clean : 
	rm *.o


