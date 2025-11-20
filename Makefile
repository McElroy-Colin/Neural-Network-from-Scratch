# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -Wextra -g -O0 -Iinclude

SRCS = main.c $(wildcard utils/*.c)
OBJS = $(SRCS:.c=.o)
OBJS_FSLASH := $(foreach obj,$(subst /,\,$(OBJS)),$(obj))

# Default target
all: main

# Link object files to create executable
main: $(OBJS)
	$(CC) $(CFLAGS) -o main $(OBJS)
	cmd /C del /Q $(OBJS_FSLASH)

# Compile .c files to .o
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean generated files
clean:
	del /Q main.exe $(OBJS_FSLASH)

.PHONY: all clean
