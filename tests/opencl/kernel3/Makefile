ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := kernel3

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.cc

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

common.h: $(SRC_DIR)/common.h
	cp $< $@

KERNEL_SRCS := kernel.cl common.h

OPTS ?=

include ../common.mk
