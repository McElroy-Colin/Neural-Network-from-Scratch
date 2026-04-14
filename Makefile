CC = gcc
NVCC = nvcc

CFLAGS = -I include -Wno-sign-compare
LDFLAGS = -lm
NVCC_FLAGS = -I include -rdc=true

C_RLFLAGS = -O3 -march=native
C_DBGFLAGS = $(CFLAGS) -g -O0 -Wall -Wextra -Wpedantic
CU_DBGFLAGS = -g -G -O0

#Default to release, override with: make BUILD=debug
BUILD ?= release

CU_SRCS = $(wildcard src/*.cu) $(wildcard utils/*.cu)
C_SRCS = $(wildcard src/*.c) $(wildcard utils/*.c)

TARGET = build/main

C_OBJS = $(C_SRCS:.c=_c.o)
CU_OBJS = $(CU_SRCS:.cu=_cu.o)

ifeq ($(BUILD), debug)
all: debug_hst debug_dvc
	rm -f $(C_OBJS) $(CU_OBJS) main_c.o main_cu.o
else
all: main_hst main_dvc
	rm -f $(C_OBJS) $(CU_OBJS) main_c.o main_cu.o
endif

main_dvc: main_cu.o $(CU_OBJS) $(C_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $(TARGET)_dvc.exe

main_hst: main_c.o $(C_OBJS)
	$(CC) $(CFLAGS) $(C_RLFLAGS) $^ -o $(TARGET)_hst.exe $(LDFLAGS)

debug_hst: main_c.o $(C_OBJS)
	$(CC) $(CFLAGS) $(C_DBGFLAGS) $^ -o $(TARGET)_hst_dbg.exe $(LDFLAGS)

debug_dvc: main_cu.o $(CU_OBJS) $(C_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(CU_DBGFLAGS) $^ -o $(TARGET)_dvc_dbg.exe

ifeq ($(BUILD), debug)
%_c.o: %.c
	$(CC) $(CFLAGS) $(C_DBGFLAGS) -c $< -o $@

%_cu.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(CU_DBGFLAGS) -c $< -o $@
else
%_c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%_cu.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
endif

clean:
	rm -f $(C_OBJS) $(CU_OBJS) main_c.o main_cu.o

.PHONY: all clean main