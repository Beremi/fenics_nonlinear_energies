# Makefile for custom_coloring.so
#
# Usage:
#   make              — build with -O3 -march=native
#   make clean        — remove .so
#   make rebuild      — clean + build
#
# The resulting .so is committed to git so most users do not need to rebuild.
# Rebuild only if you modify custom_coloring.c or change target architecture.

CC      = gcc
CFLAGS  = -O3 -march=native -shared -fPIC
SRC     = graph_coloring/custom_coloring.c
TARGET  = graph_coloring/custom_coloring.so

.PHONY: all clean rebuild

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

rebuild: clean all
