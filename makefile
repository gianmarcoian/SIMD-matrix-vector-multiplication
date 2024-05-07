CC = gcc
CFLAGS = -Wall -I.
LDFLAGS = -lm -pthread  # Add -pthread for POSIX Threads

TARGETS = opt0 opt1 opt2 opt3

all: $(TARGETS)

opt0: optimized.c microtime.o
	$(CC) -o $@ $^ $(CFLAGS) -O0 $(LDFLAGS)

opt1: optimized.c microtime.o
	$(CC) -o $@ $^ $(CFLAGS) -O1 $(LDFLAGS)

opt2: optimized.c microtime.o
	$(CC) -o $@ $^ $(CFLAGS) -O2 $(LDFLAGS)

opt3: optimized.c microtime.o
	$(CC) -o $@ $^ $(CFLAGS) -O3 $(LDFLAGS)

microtime.o: microtime.c microtime.h
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o *~ core $(TARGETS)
