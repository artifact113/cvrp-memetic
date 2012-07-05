
GCC_DEBUG=-ggdb

make: main.o instance.o utilities.o timer.o ls.o
	gcc $(GCC_DEBUG) -O main.o timer.o instance.o utilities.o ls.o -o cvrp-test -lm -ansi -Wall

main.o: instance.o utilities.o main.h main.c
	gcc $(GCC_DEBUG) -O -c main.c -o main.o -ansi -Wall

timer.o: timer.h timer.c
	gcc $(GCC_DEBUG) -O -c timer.c -o timer.o -ansi -Wall

instance.o: instance.h instance.c
	gcc $(GCC_DEBUG) -O -c instance.c -o instance.o -ansi -Wall

utilities.o: utilities.h utilities.c
	gcc $(GCC_DEBUG) -O -c utilities.c -o utilities.o -ansi -Wall

ls.o: ls.h ls.c
	gcc $(GCC_DEBUG) -O -c ls.c -o ls.o -ansi -Wall

clean:
	rm -f *.o *~ cvrp-test
