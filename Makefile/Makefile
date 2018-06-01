#define customized suffixes
.SUFFIXES : .c .o

#Macro for objects
OBJ = main.o hello.o 

#output file
TARGET = hello

#make method
CC = gcc

#flag for g++ #use CFLAGS for gcc -I: including dictionary
CCFLAGS = -c -Isrc/ -Iinclude

all :
	@echo "all"
	$(TARGET) : $(OBJ)
		$(CC) -o $(TARGET) $(OBJ)

#remove used object files
clean :
	@echo "clean"
	@rm -rf $(OBJ) $(TARGET) core
