
CC = gcc
CCFLAGS =-fopenmp 
COMFLAGS=-fopenmp
TARGET=

$(TARGET) : $(TARGET).o
	$(CC) -o $@ $^ $(CCFLAGS)

$(TARGET).o : $(TARGET).c
	$(CC) -c $^ $(COMFLAGS)
