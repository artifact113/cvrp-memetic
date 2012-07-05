#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "instance.h"
#include "utilities.h"
#include "ls.h"

typedef struct 
{
    int city0;
    int city1;
}city_pair;

typedef struct
{
    int *arr;
    int *limits;
    int nTours;
}cvrp_sol;

int mcap, mst, dt, xdepot, ydepot, dBest, *caps;

extern double   *xc, *yc;
extern int   round_distance (int i, int j);

/*
 * Evaluates a population a gets the best individual
 *
 * Input: pop -> Population to be evaluated
 *        spop -> Size of the population
 *        dist -> Distance matrix
 * Output: Pointer to the best element of the population
 * Side effects: None
 */
cvrp_sol *evaluate_pop(cvrp_sol **pop, int spop, int **dist);

/*
 * Returns de total distance for making a tour
 *
 * Input: arr -> Array containing the solution
 *        limits -> Tour limits
 *        nTours -> Number of tours
 *        dist -> Distance matrix
 * Output: Total distance to be traveled in a solution
 * Side effect: None
 */
int evaluate_sol(int *arr, int *limits, int nTours, int **dist);

/*
 * Performs an optimization over all the elements of the given population
 *
 * Input: pop -> Population to be optimzed
 *        spop -> Size of the population
 *        dist -> Distance matrix
 * Output: none
 * Side effect: A local search has been applied to every member of the population
 */
void optimize(cvrp_sol **pop, int spop, int **dist);

/*
 * Given an array of cities and a distance matrix, this function delimits a valid
 * tour.
 *
 * Input: arr -> Array with the cities
 *        limits -> Array where the limits should be saved
 *        dist -> Distance matrix
 *        ncities -> Number of cities in arr
 * Output: Number of tours generated
 * Side effect: limits has the limits of each tour in the array
 */
int delimit(int *arr, int *limits, int **dist, int ncities);

/*
 * Function that creates an initial population of n individuals.
 * This procedure generates a population with 2 elements, but allocates enough
 * space for the n individuals that will be added as the algorithm runs.
 *
 * Input: n -> number of individuals to create
 *        dist -> Distance matrix
 *        nc -> Number of cities
 * Output: Array containing the indiciduals of an initial population
 * Side effects: None
 */
cvrp_sol **generate_initial_pop(int n, int **dist, int nc);

/*
 * Function that evaluates wether a tour is better than another
 *
 * Input: arr0 -> Cities that describe solution 0
 *        limits0 -> Limits of solution 0
 *        nTours0 -> Number of tours represented on solution 0
 *        arr1 -> Cities that describe solution 1
 *        limits1 -> Limits of solution 1
 *        nTours1 -> Number of tours represented on solution 1
 *        dist -> Distance matrix
 * Output: 1 if arr0 is a better solution than arr1, 0 otherwise
 * Side effects: none
 */
int compareTours(int *arr0, int *limits0, int nTours0, int *arr1, int *limits1, int nTours1, int **dist);

/*
 * Function that prints the solution specified by the parameters
 *
 * Input: arr -> Array containing the cities
 *        limits -> Limits of the tours in the preceding array
 *        nTours -> Ammount of tours contained in the solution
 *        dist -> Distance matrix
 * Output: none
 * Side effects: none
 */
void printSol(int *arr, int *limits, int nTours, int **dist);

/*
 * Function that reads an instance of a crvp problem and sets up the corresponding
 * variables
 *
 * Input: string -> filename which contains the graph
 * Output: none
 * Side effects: ncitites, capacity, maxtime, droptime, xdepot, ydepot, xc, yc, zc are set
 */
void read_cvrp(char *filename);

/*
 * Function that finds an initial solution to the CVRP using the heuristic
 * proposed by Clark and Wright, the Savings algorithm
 * 
 * Input: arr -> Array where the initial solution should be set
 *        limits -> Array where the values of the tour limits should be set
 *        dist -> Matrix containing the distances to reach each city from
 *                   from each other
 *        ncities -> Number of cities being considered
 * Output: Integer contining the number of tours generated
 * Side effects: tours contains all the tours that conform an initial solution
 *               to the CVRP and limits contains the limits of each tour in the
 *               preceding array
 */
int initSol(int *arr, int *limits, int **dist, int ncities);

/*
 * Function that sorts a group of pairs of cities in descending order using 
 * mergeSort
 *
 * Input: arr -> Array to be sorted
 *        s -> Savings matrix
 *        start -> Position from which to start sorting
 *        end -> Position at which the function should end the sort 
 * Output: none
 * Side effects: arr is sorted
 */
void sortPairs(city_pair **arr, int **s, int start, int end);

/*Function that solves the instance of TSP contained in the array 'arr'
 * between 'lower' and 'upper' using the function tsp.
 *
 * Input: arr -> Array containing the instances of TSP
 *        lower -> Position to start taking cities from arr
 *        upper -> Last position from which to take cities from arr
 *        distMatGen -> Distance matrix
 * Output: None
 * Side effect: arr has, between lower and upper, a solution to a local TSP
 */
void boxTSP(int *arr, int lower, int upper, int **distMatGen);

/*
 * Function that solves an instance of a tsp using ILS as heuristic
 * Input: best -> Pointer to the array where the best solution should be saved
 * Output: None
 * Side effects: The best solution found by the heuristic is saved on 'best'.
 */
void tsp(int *best);

/*
 * Function that generates a random neighbour of t using a random position swap
 * Input: t0 -> Pointer to the tour that wants to be modified
          t1 -> Pointer to an array with enough memory allocated to store a neighbour
 * Output: None
 * Side effects: t1 contains a neighbour of t0
 */
void random_neighbour(int *t);

/*
 * Function that calculates the probability used to select a neighbour on SHC
 * Input: vc -> Current tour
 *        vn -> Neighbour
 *        T -> Temperature
 * Output: Double containing the probability with which the neighbour should be
 *         chosen
 */
double calculate_prob(int *vc, int *vn, double T);

/*
 * Function that compares two tours to check if they are the same
 * Input: t0 -> Tour to be compared
 *        t1 -> Tour to be compared
 * Output: 1 if the tours are the same, 0 otherwise
 */
int is_same(int *t0, int *t1);

/*
 * Function that copies a tour into another variable
 * Input: t0 -> Tour to be copied
 *        t1 -> Pointer to an array with enough memory allocated to store t0
 * Output: none
 * Side effects: t1 has an exact copy of t0
 */
void copy_tour(int *t0, int *t1);
