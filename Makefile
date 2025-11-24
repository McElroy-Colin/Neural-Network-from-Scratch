CC = gcc

CFLAGS = -lm -Iinclude
RLFLAGS = $(CFLAGS) -O3 -march=native -Wno-sign-compare 
DBGFLAGS = $(CFLAGS) -g -O0 -Wall -Wextra -Wpedantic 

SRCS = main.c $(wildcard src/*.c) $(wildcard utils/*.c)

RL_OBJS = $(SRCS:.c=.o)
DBG_OBJS = $(SRCS:.c=.dbg.o)

RL_TARGET = main.exe
DBG_TARGET = main_dbg.exe

# Default target
all: main main_dbg

# Link object files to create executable
main: $(RL_OBJS)
	$(CC) $(RL_OBJS) $(CFLAGS) -o $(RL_TARGET)
	rm -f $(RL_OBJS)

main_dbg: $(DBG_OBJS)
	$(CC) $(DBG_OBJS) $(DBGFLAGS) -o $(DBG_TARGET)
	rm -f $(DBG_OBJS)

%.dbg.o: %.c
	$(CC) $(DBGFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(RLFLAGS) -c $< -o $@

# Clean generated files
clean:
	rm -f $(RL_OBJS) $(DBG_OBJS) $(RL_TARGET) $(DBG_TARGET)

.PHONY: all clean
