# Compiler
CC = gcc

# Compiler flags
CFLAGS = -lm -Iinclude
RLFLAGS = $(CFLAGS) -O3 -march=native -Wno-sign-compare
DBGFLAGS = $(CFLAGS) -g -O0 -Wall -Wextra -Wpedantic

SRCS = main.c $(wildcard utils/*.c)
OBJS = $(SRCS:.c=.o)
OBJS_FSLASH := $(foreach obj,$(subst /,\,$(OBJS)),$(obj))

# Default target
all: main main_dbg

# Link object files to create executable
main: $(OBJS)
	$(CC) $(CFLAGS) -o main $(OBJS)
	cmd /C del /Q $(OBJS_FSLASH)

main_dbg: $(OBJS)
	$(CC) $(DBGFLAGS) -o main_dbg $(OBJS)
	cmd /C del /Q $(OBJS_FSLASH)

# Compile .c files to .o
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean generated files
clean:
	del /Q main.exe main_dbg.exe $(OBJS_FSLASH)

.PHONY: all clean
