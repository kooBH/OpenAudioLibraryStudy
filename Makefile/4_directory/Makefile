#미리 지정된 매크로 'CC' : .c 파일의 컴파일러 
CC=gcc


SRC =hello

#헤더 폴더를 받는 매크로
# -I<경로>  를 하면 해당 경로에서 헤더를 찾는다
DIR=-Iinclude

hello : $(SRC).o main.o
	$(CC) -o $@ $^	

#src/ 폴더에있는 파일들을 컴파일한다
$(SRC).o :  
	$(CC)  $(DIR) -c src/$(SRC).c

clean : 
	rm *.o
   
