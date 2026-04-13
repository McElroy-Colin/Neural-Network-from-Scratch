CC = gcc
NVCC = nvcc

CFLAGS = -lm -I include -Wno-sign-compare -O3 -march=native
NVCC_FLAGS = -I include -rdc=true

DBGFLAGS = $(CFLAGS) -g -O0 -Wall -Wextra -Wpedantic

# Default to cuda, override with: make BUILD=serial
BUILD ?= cuda

MAIN = ff_test

ifeq ($(BUILD), cuda)
    CU_SRCS = $(MAIN).cu $(wildcard src/*.cu) $(wildcard utils/*.cu)
    C_SRCS = $(wildcard utils/*.c)
    TARGET = goat_program.exe
else
    C_SRCS = $(wildcard src/*.c) $(wildcard utils/*.c)
    CU_SRCS =
    TARGET = goat_program_serial.exe
endif

C_OBJS = $(C_SRCS:.c=_c.o)
CU_OBJS = $(CU_SRCS:.cu=_cu.o)

all: main

ifeq ($(BUILD), cuda)
main: $(CU_OBJS) $(C_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $(TARGET)
	rm -f $(C_OBJS) $(CU_OBJS)
else
main: $(C_OBJS)
	$(CC) $(CFLAGS) $^ -o $(TARGET)
	rm -f $(C_OBJS)
endif

%_c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%_cu.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(C_OBJS) $(CU_OBJS) $(TARGET)

.PHONY: all clean main