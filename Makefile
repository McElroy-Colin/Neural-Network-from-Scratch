CC = gcc

CFLAGS = -lm -Iinclude
RLFLAGS = $(CFLAGS) -O3 -march=native -Wno-sign-compare
DBGFLAGS = $(CFLAGS) -g -O0 -Wall -Wextra -Wpedantic

SRCS = main.c $(wildcard src/*.c) $(wildcard utils/*.c)

RL_OBJS = $(SRCS:.c=.o)
DBG_OBJS = $(SRCS:.c=.dbg.o)

RL_OBJS_FSLASH := $(foreach obj,$(subst /,\,$(RL_OBJS)),$(obj))
DBG_OBJS_FSLASH := $(foreach obj,$(subst /,\,$(DBG_OBJS)),$(obj))

# Default target
all: main main_dbg

# Link object files to create executable
main: $(RL_OBJS)
	$(CC) $(CFLAGS) -o main $(RL_OBJS)
	cmd /C del /Q $(RL_OBJS_FSLASH)

main_dbg: $(DBG_OBJS)
	$(CC) $(DBGFLAGS) -o main_dbg $(DBG_OBJS)
	cmd /C del /Q $(DBG_OBJS_FSLASH)

%.dbg.o: %.c
	$(CC) $(DBGFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(RLFLAGS) -c $< -o $@

# Clean generated files
clean:
	del /Q main.exe main_dbg.exe $(RL_OBJS_FSLASH) $(DBG_OBJS_FSLASH)

.PHONY: all clean
