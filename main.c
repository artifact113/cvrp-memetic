#include "main.h"

int main(int argc, char **argv)
{
    int i, j, its = 40, itBest = 0, popSize = 10, ncitiesGen;
    cvrp_sol *best;
    cvrp_sol **pop, **offs;
    time_t tBest, start, end;
    int **distMatGen;
    char fileName[100], *tmp, fnd[100];
    FILE *outFile;
    
    read_cvrp(argv[1]);
    ncitiesGen = ncities;

    distMatGen = compute_distances();

    pop = generate_initial_pop(popSize, distMatGen, ncitiesGen);
    
    /*Print initial population*/
    /*for(i = 0; i < popSize; ++i){
        printf("\nSolution %d: \n", i);
        if(pop[i] == NULL){
            printf("NULL\n");
        }else{
            printSol(pop[i]->arr, pop[i]->limits, pop[i]->nTours, distMatGen);
        }
    }*/
    
    optimize(pop, popSize,distMatGen);
    
    /*Print initial population optimized*/
    /*for(i = 0; i < popSize; ++i){
        printf("\nSolution %d: \n", i);
        if(pop[i] == NULL){
            printf("NULL\n");
        }else{
            printSol(pop[i]->arr, pop[i]->limits, pop[i]->nTours, distMatGen);
        }
    }*/
    
    best = evaluate_pop(pop, popSize, distMatGen);
    /*printSol(best->arr, best->limits, best->nTours, distMatGen);
    printf("Total service time: %d\n", evaluate_sol(best->arr, best->limits, best->nTours, distMatGen));*/
    

    /*Start iterating*/
    for(i = 0; i < its; ++ i){
        /*
         * The selection for crossing is using all the individuals of the population
         * and pair them according to their possition in the array
         */
        offs = cross_pop(pop, popSize, distMatGen);
        optimize(offs, popSize, distMatGen);
        
        /*Mutation algorithm over 'offs' here*/
        
        optimize(offs, popSize, distMatGen);
        
        /*Selection algorithm between 'pop' and 'offs' here*/
        
        best = evaluate_pop(pop, popSize, distMatGen);
    }
        
    
    for(i = 0; i < popSize; ++ i)
        free(pop[i]);    
    free(pop);
    printf("Program has ended successfully\n");
    return 0;
}

void read_cvrp(char *filename)
{

    FILE *cvrp_file = fopen(filename, "r");
    if ( cvrp_file == NULL ) {
        fprintf(stderr,"No instance file specified, abort\n");
        exit(1);
    }
    printf("\nreading cvrp-file %s ... \n\n", filename);

    /* number of customers, vehicle capacity, maximum route time, drop time */
    fscanf(cvrp_file, "%d %d %d %d", &ncities, &mcap, &mst, &dt);
    /* printf("ncities: %d, mcap: %d, mst: %d, dt: %d\n", ncities, mcap, mst, dt); */
    ncities ++;

    if( (xc = malloc(sizeof(double) * ncities)) == NULL )
	exit(EXIT_FAILURE);
    if( (yc = malloc(sizeof(double) * ncities)) == NULL )
	exit(EXIT_FAILURE);

    /* depot x-coordinate, depot y-coordinate */
    fscanf(cvrp_file, "%lf %lf", &xc[0], &yc[0]);

    if( (caps = malloc(sizeof(int) * ncities)) == NULL )
	exit(EXIT_FAILURE);
   
    caps = malloc(sizeof(int) * (ncities));
    /* for each customer in turn: x-coordinate, y-coordinate, quantity */
    int i;
    for (i = 1 ; i < ncities ; i++ ) {
        fscanf(cvrp_file,"%lf %lf %d", &xc[i], &yc[i], &caps[i] );
    }

    distance = round_distance;
    fclose(cvrp_file);
}

void boxTSP(int *arr, int lower, int upper, int **distMatGen){
    int *best;
    int i, j;
    
    /* esto no incluye el depósito */
    ncities = upper - lower + 1;
    
    /* hay que agregar espacio para la ciudad final que es igual a la inicial */
    best = malloc(sizeof(int) * (ncities + 1));

    /* Code taken from compute_distances (With light modifications) */
    if((distMat = malloc(sizeof( int) * ncities * ncities +
                sizeof( int *) * ncities )) == NULL){
        printf("Out of memory, exit.");
        exit(1);
    }
    
    for ( i = 0 ; i < ncities ; i++ ) {
        distMat[i] = (int *)(distMat + ncities) + i*ncities;
        /* la ciudad cero es el depósito. todas las demás están desplazadas por él */
        int ik = 0 == i ? 0 : i + lower - 1;
        for ( j = 0  ; j < ncities ; j++ ) {
            /* la ciudad cero es el depósito. todas las demás están desplazadas por él */
            int jk = 0 == j ? 0 : j + lower - 1;
            distMat[i][j] = distMatGen[arr[ik]][arr[jk]];
        }
    }
    /*End of code taken from compute_distances*/
    
    tsp(best);
    
    int dp;
    for(i = 0; i < ncities; ++i){
        if (best[i] == 0) dp = i;
        best[i] = 0 == best[i] ? 0 : arr[best[i] + lower - 1];
    }

    for(i = 1; i < ncities; ++i) {
        arr[i + lower - 1] = best[((i + dp) % ncities)];
    }

    free(distMat);
    free(best);
    
    return;
}

void tsp(int *best){
    int *vc, *vn, *tmp;
    int t, max = 5000, i;
    double T = 1500000, prob, ran;
    clock_t ticks;

    nn_ls = MIN (ncities - 1, 40);
    nnMat = compute_NNLists();

    
    vc = generate_random_vector();
    vc[ncities] = vc[0];
    tmp = malloc(sizeof(int) * (ncities + 1));
    vn = malloc(sizeof(int) * (ncities + 1));

    for(i = 0; i <= ncities; ++i){
        best[i] = vc[i];
    }

    for(t = 0; t < max; ++t){
        copy_tour(vc, tmp);
        
        dlb = calloc(ncities, sizeof(int));
        three_opt_first(vc);
        
        copy_tour(vc, vn);
        if(is_same(tmp, vc)){
            random_neighbour(vn);
        }else{
            copy_tour(tmp, vc);
        }

        prob = calculate_prob(vc, vn, T);
        /*printf("%f\n", prob);*/
        ticks = clock();
        seed = (int) ticks;
        ran = ran01(&seed);
        /*printf("%f ? %f\n",ran, prob);*/
        if(ran < prob){
            for(i = 0; i <= ncities; ++ i){
                vc[i] = vn[i];
            }
        }
        
        /*printf("%d, %d\n", compute_length(vc), compute_length(best));*/
        if(compute_length(vc) < compute_length(best)){
            for(i = 0; i <= ncities; ++i)
                best[i] = vc[i];
        }
        
        free(dlb);
    }

    free(tmp);
    free(vn);
    free(vc);
    
    return;
}

void random_neighbour(int *t){
    int r0, r1;
    clock_t ticks;
    
    ticks = clock();
    
    /*Se le suma 1 porque, si se llega con 0 ticks, se quiere que la semilla no
      sea 0, ya que esto no generará ningún número
      */
    seed = (int)ticks + 1;
    
    r0 = random_number(&seed) % ncities;
    r1 = random_number(&seed) % ncities;
    int tmp = t[r0];
    t[r0] = t[r1];
    t[r1] = tmp;
    if (0 == r0) {
        t[ncities] = t[r0];
    } else if (0 == r1) {
        t[ncities] = t[r1];
    }
    
    return;
}

double calculate_prob(int *vc, int *vn, double T){
    return 1 / (1 + exp((compute_length(vc) - compute_length(vn)) / T));
}

int is_same(int *t0, int *t1){
    int i;
    for(i = 0; i <= ncities; ++ i)
        if(t0[i] != t1[i])
            return 0;
    return 1;    
}

void copy_tour(int *t0, int *t1){
    int i;
    for(i = 0; i <= ncities; ++i)
        t1[i] = t0[i];
}

int initSol(int *arr, int *limits, int **dist, int ncities){
    int i, j, k, numPairs = ((ncities - 1) * ((ncities - 1) - 1) / 2), ammTour = 0, start;
    int foundMatch = 0;
    int isNotAvailable[ncities], isInTour[ncities];
    int tourTime[numPairs], ammCitiesTour[numPairs], tourCap[numPairs], *at;
    int **savings;
    int *tours[numPairs];
    city_pair **pairs = malloc(numPairs * sizeof(city_pair *));
    city_pair *pair;
   
    memset(tourTime, 0, numPairs * sizeof(int));
    memset(ammCitiesTour, 0, numPairs * sizeof(int));
    memset(tourCap, 0, numPairs * sizeof(int));
    memset(isNotAvailable, 0, ncities * sizeof(int));
    memset(isInTour, 0, ncities * sizeof(int));
    
    /*Memory allocation for tours*/
    for(i = 0; i < numPairs; ++ i)
        tours[i] = (int *)malloc(ncities * sizeof(int));
    /*Memory allocation for savings*/
    if((savings = malloc(sizeof( int) * (ncities + 1) * (ncities + 1) +
                sizeof( int *) * (ncities + 1))) == NULL){
        printf("Out of memory, exit.");
        exit(1);
    }
    
    for ( i = 1; i < ncities ; ++ i) {
        savings[i] = (int *)(savings + ncities) + i*ncities;
        for ( j = i  ; j < ncities ; ++ j) {
            savings[i][j] = (2 * dist[0][i] + 2 * dist[0][j]) - (dist[0][i] + dist[i][j] + dist[j][0]);
        }
    }
    
    k = 0;
    for(i = 1; i < ncities; ++ i){
        for(j = i + 1; j < ncities; ++ j){
            if((pair = (city_pair *)malloc(sizeof(city_pair))) == NULL){
                printf("Out of memory, exit.");
                exit(1);
            }
            pair->city0 = i;
            pair->city1 = j;
            pairs[k] = pair;
            ++ k;
        }
    }
    
    sortPairs(pairs, savings, 0, numPairs);
    /*
     * Parallel version of the Clark and Wright algorithm
     */
    for(i = 0; i < numPairs; ++i){
        /*printf("Probando el par: %d - %d\n", pairs[i]->city0, pairs[i]->city1);*/
        if(isNotAvailable[pairs[i]->city0] || isNotAvailable[pairs[i]->city1])
            continue;
        
        foundMatch = 0;
        for(j = 0; j < ammTour; ++ j){
            at = tours[j];
            /*
             * If the pair represents the corners of the tour, do not proceed
             */
            if((pairs[i]->city0 == at[0] && pairs[i]->city1 == at[ammCitiesTour[j] - 1])
                ||
                (pairs[i]->city1 == at[0] && pairs[i]->city0 == at[ammCitiesTour[j] - 1])){
                foundMatch = 1;
                break;
            }
            
            /*
             * Either city is already part of a tour, but cannot be joined with the current tour
             */ 
            if((isInTour[pairs[i]->city0] && (pairs[i]->city0 != at[0]) && (pairs[i]->city0 != at[ammCitiesTour[j] - 1]))
                || 
                (isInTour[pairs[i]->city1] && (pairs[i]->city1 != at[0]) && (pairs[i]->city1 != at[ammCitiesTour[j] - 1])))
                continue;
            
            if(pairs[i]->city0 == at[0]){
                if(((tourTime[j] - dist[0][at[0]]) + 
                    (dist[0][pairs[i]->city1] + dist[pairs[i]->city1][at[0]] + dt) < mst)
                    &&
                    (tourCap[j] + caps[pairs[i]->city1] < mcap)){
                        for(k = ammCitiesTour[j]; k >= 0; -- k)
                            at[k + 1] = at[k];
                        at[0] = pairs[i]->city1;
                        isNotAvailable[at[1]] = 1;
                        tourTime[j] = (tourTime[j] - dist[0][at[1]])
                                    + (dist[0][at[0]] + dist[at[0]][at[1]]
                                       + dt);
                        tourCap[j] = tourCap[j] + caps[pairs[i]->city1];
                        ++ ammCitiesTour[j];
                        foundMatch = 1;
                        break;
                }
            }else if(pairs[i]->city0 == at[ammCitiesTour[j] - 1]){
                if(((tourTime[j] - dist[0][at[ammCitiesTour[j] - 1]]) + 
                    (dist[0][pairs[i]->city1] + dist[pairs[i]->city1][at[ammCitiesTour[j] - 1]] 
                     + dt) < mst)
                    &&
                    (tourCap[j] + caps[pairs[i]->city1] < mcap)){
                        at[ammCitiesTour[j]] = pairs[i]->city1;
                        isNotAvailable[at[ammCitiesTour[j] - 1]] = 1;
                        tourTime[j] = (tourTime[j] - dist[0][at[ammCitiesTour[j] - 1]])
                                    + (dist[0][pairs[i]->city1] + dist[pairs[i]->city1][at[ammCitiesTour[j] - 1]]
                                       + dt);
                        tourCap[j] = tourCap[j] + caps[pairs[i]->city1];
                        ++ ammCitiesTour[j];
                        foundMatch = 1;
                        break;
                }
            }else if(pairs[i]->city1 == at[0]){
                if(((tourTime[j] - dist[0][at[0]]) + 
                    (dist[0][pairs[i]->city0] + dist[pairs[i]->city0][at[0]] + dt) < mst)
                    &&
                    (tourCap[j] + caps[pairs[i]->city0] < mcap)){
                        for(k = ammCitiesTour[j]; k >= 0; -- k)
                            at[k + 1] = at[k];
                        at[0] = pairs[i]->city0;
                        isNotAvailable[at[1]] = 1;
                        tourTime[j] = (tourTime[j] - dist[0][at[1]])
                                    + (dist[0][at[0]] + dist[at[0]][at[1]]
                                       + dt);
                        tourCap[j] = tourCap[j] + caps[pairs[i]->city0];
                        ++ ammCitiesTour[j];
                        foundMatch = 1;
                        break;
                }            
            }else if(pairs[i]->city1 == at[ammCitiesTour[j] - 1]){
                if(((tourTime[j] - dist[0][at[ammCitiesTour[j] - 1]]) + 
                    (dist[0][pairs[i]->city0] + dist[pairs[i]->city0][at[ammCitiesTour[j] - 1]] 
                     + dt) < mst)
                    &&
                    (tourCap[j] + caps[pairs[i]->city0] < mcap)){
                        at[ammCitiesTour[j]] = pairs[i]->city0;
                        isNotAvailable[at[ammCitiesTour[j] - 1]] = 1;
                        tourTime[j] = (tourTime[j] - dist[0][at[ammCitiesTour[j] - 1]])
                                    + (dist[0][pairs[i]->city0] + dist[pairs[i]->city0][at[ammCitiesTour[j] - 1]]
                                       + dt);
                        tourCap[j] = tourCap[j] + caps[pairs[i]->city0];
                        ++ ammCitiesTour[j];
                        foundMatch = 1;
                        break;
                }            
            }
        }
        
        if(!foundMatch && !isInTour[pairs[i]->city0] && !isInTour[pairs[i]->city1]){
            if((dist[0][pairs[i]->city0] + dist[pairs[i]->city0][pairs[i]->city1] + dist[0][pairs[i]->city1]
                + 2 * dt < mst)
                &&
                caps[pairs[i]->city0] + caps[pairs[i]->city1] < mcap){
                tours[ammTour][0] = pairs[i]->city0;
                tours[ammTour][1] = pairs[i]->city1;
                tourTime[ammTour] = dist[0][pairs[i]->city0] + dist[pairs[i]->city0][pairs[i]->city1] + dist[0][pairs[i]->city1]
                                    + 2 * dt;
                tourCap[ammTour] = caps[pairs[i]->city0] + caps[pairs[i]->city1];
                ammCitiesTour[ammTour] = 2;
                ++ ammTour;
                foundMatch = 1;
            }
        }
        
        /*
         * Both cities are necessarily now part of a tour if they were matched
         */
        if(foundMatch){
            isInTour[pairs[i]->city0] = 1;
            isInTour[pairs[i]->city1] = 1;
        }
       /* 
        int p0, p1;
        printf("amMTour: %d\n", ammTour);
        for(p0 = 0; p0 < ammTour; ++ p0){
            printf("Tour %d: ", p0);
            for(p1 = 0; p1 < ammCitiesTour[p0]; ++ p1){
                printf("%d, ", tours[p0][p1]);
            }
            printf("\n\n");
        }
        printf("---------------------------\n");*/
    }
    
    /*
     * For every city that was not made part of a tour, create an individual route
     */
    for(i = 1; i < ncities; ++ i){
        if(isInTour[i])
            continue;
        tours[ammTour][0] = i;
        ammCitiesTour[ammTour] = 1;
        ++ ammTour;
    }
    /*
     * Now we write each tour and set the limits
     */
    for(i = 0; i < ammTour; ++ i){
        start = (i == 0 ? 0 : limits[i - 1]);
        for (j = 0; j < ammCitiesTour[i]; ++j){
            arr[start + j] = tours[i][j];
        }
        limits[i] = start + ammCitiesTour[i];
    }
    
    for(i = 0; i < ammTour; ++ i)
        free(tours[i]);
    
    for(i = 0; i < numPairs; ++ i)
        free(pairs[i]);
    
    free(pairs);
    free(savings);
    
    return ammTour;
}

void sortPairs(city_pair **arr, int **s, int start, int end){
    if(start == end - 1)
        return;
    
    sortPairs(arr, s, start, (start + end) / 2);
    sortPairs(arr, s, (start + end) / 2, end);
    
    int i = start, j = (end + start) / 2, k = 0;
    city_pair **tmp = malloc((end - start + 1) * sizeof(city_pair *));
    while((i < ((end + start) / 2)) || (j < end)){
        if(i >= (end + start) / 2){
            tmp[k] = arr[j];
            ++ j;
        }else if(j >= end){
            tmp[k] = arr[i];
            ++ i;
        }else{
            if(s[arr[i]->city0][arr[i]->city1] > s[arr[j]->city0][arr[j]->city1]){
                tmp[k] = arr[i];
                ++ i;
            }else{
                tmp[k] = arr[j];
                ++ j;
            }
        }
        ++ k;
    }
        
    for(k = 0; k < end - start; ++ k)
        arr[start + k] = tmp[k];
    
    free(tmp);
    return;
}

void printSol(int *arr, int *limits, int nTours, int **dist){
    int servTime, capacity, i, j;
    for(i = 0; i < nTours; ++ i){
        printf("Tour %d: ", i);
        for(j = (i == 0 ? 0 : limits[i - 1]); j < limits[i]; ++j){
            printf("%d ", arr[j]);
        }
        printf("\n");
        servTime = 0;
        capacity = 0;
        for(j = (i == 0 ? 0 : limits[i - 1]); j < limits[i] - 2; ++ j){
            servTime += dist[arr[j]][arr[j + 1]] + dt;
            capacity += caps[arr[j]];
        }

        int k = (i == 0 ? 0 : limits[i - 1]);
        servTime += dist[0][arr[k]]
                    + dist[0][arr[limits[i] - 1]]
                    + dt;
        capacity += caps[arr[limits[i] - 1]];
        printf("Service time: %d / %d\n", servTime, mst);
        printf("Capacity: %d / %d", capacity, mcap);
        printf("\n\n");
    }     
}

int compareTours(int *arr0, int *limits0, int nTours0, int *arr1, int *limits1, int nTours1, int **dist){
    int totDist0, totDist1;
    
    /*Calculate distance in tour 0*/
    totDist0 = evaluate_sol(arr0, limits0, nTours0, dist);
    
    /*Calculate distance in tour 1*/
    totDist1 = evaluate_sol(arr1, limits1, nTours1, dist);
    
    /*printf("totDist0: %d, totDist1: %d\n", totDist0, totDist1);*/
    if(totDist0 < totDist1){
        dBest = totDist0;
        return 1;
    }
    
    dBest = totDist1;
    return 0;
}

cvrp_sol **generate_initial_pop(int n, int **dist, int nc){
    int i;
    cvrp_sol *aux;
    cvrp_sol **ret = malloc(n * sizeof(cvrp_sol *));
    
    for(i = 0; i < n; ++ i)
        ret[i] = NULL;
    
    aux = (cvrp_sol *)malloc(sizeof(cvrp_sol));
    aux->arr = malloc((nc - 1) * sizeof(int));
    aux->limits = malloc((nc - 1) * sizeof(int));
    
    aux->nTours = initSol(aux->arr, aux->limits, dist, nc);
    ret[0] = aux;
    
    aux = (cvrp_sol *)malloc(sizeof(cvrp_sol));
    aux->arr = malloc((nc - 1) * sizeof(int));
    aux->limits = malloc((nc - 1) * sizeof(int));
    
    ncities = nc - 1;
    seed = 1;
    aux->arr = generate_random_vector();
    
    /*for(i = 0; i < nc - 1; ++ i)
        printf("%d, ", aux->arr[i]);
    printf("\n");*/
    
    for(i = 0; i < nc - 1; ++ i)
        ++ aux->arr[i];
    aux->nTours = delimit(aux->arr, aux->limits, dist, nc - 1);
    ret[1] = aux;
    
    return ret;
}

int delimit(int *arr, int *limits, int **dist, int ncities){
    int ammT = 0, totCap, totTime, i;
    
    totCap = caps[arr[0]];
    totTime = 2 * dist[0][arr[0]] + dt;
    for(i = 0; i < ncities - 1; ++ i){
        if(((totTime - dist[0][arr[i]]) + (dist[arr[i]][arr[i + 1]] + dist[arr[i + 1]][0] + dt) < mst)
            &&
            totCap + caps[arr[i + 1]] < mcap){
            totTime = (totTime - dist[0][arr[i]]) + (dist[arr[i]][arr[i + 1]] + dist[arr[i + 1]][0] + dt);
            totCap += caps[arr[i + 1]];
        }else{
            limits[ammT] = i + 1;
            /*
            printf("Limit for tour %d: %d\n", ammT, limits[ammT]);
            */
            ++ ammT;
            totCap = caps[arr[i + 1]];
            totTime = 2 * dist[0][arr[i + 1]] + dt;
        }
    }
    
    if(limits[ammT - 1] != (ncities - 1)){
        limits[ammT] = ncities - 1;
        ++ ammT;
    }
    
    return ammT;    
}

void optimize(cvrp_sol **pop, int spop, int **dist){
    int i, j, low, up;
    for(i = 0; i < spop; ++ i){
        if(pop[i] != NULL){
            for(j = 0; j < pop[i]->nTours; ++ j){
                low = (j == 0 ? 0 : pop[i]->limits[j - 1]);
                up = pop[i]->limits[j];
                /*printf("Iteration %d: low = %d, up = %d\n", j, low, up);*/
                boxTSP(pop[i]->arr, low, up, dist);
            }
        }
    }
}

int evaluate_sol(int *arr, int *limits, int nTours, int **dist){
    int totDist = 0, i, j, start;
    
    for(i = 0; i < nTours; ++ i){
        start = (i == 0 ? 0 : limits[i - 1]);
        totDist += dist[0][arr[start]];
        for(j = start; j < limits[i] - 2; ++ j)
            totDist += dist[arr[j]][arr[j + 1]];
        totDist += dist[0][arr[limits[i] - 1]];
    }
       
    return totDist;   
}

cvrp_sol *evaluate_pop(cvrp_sol **pop, int spop, int **dist){
    int i, lbest = mst, lsol;
    cvrp_sol *best;
    
    for(i = 0; i < spop; ++ i){
        if(pop[i] != NULL){
            lsol = evaluate_sol(pop[i]->arr, pop[i]->limits, pop[i]->nTours, dist);
            if(lsol <= lbest){
                lbest = lsol;
                best = pop[i];
            }
        }
    }
    
    return best;
}

cvrp_sol **cross_pop(cvrp_sol **pop, int spop, int **dist){
    int i, j, k;
    cvrp_sol **offs = (cvrp_sol **)malloc((spop - 1) * sizeof(cvrp_sol *));
    
    for(i = 0; i < spop - 1; ++ i)
        offs[i] = NULL;
    
    j = 0;
    
    for(i = 0; i < spop - 1; ++ i){
        if((pop[i] == NULL) || (pop[i + 1] == NULL))
            continue;
        /*printf("Crossing %d - %d\n", i, i + 1);*/
        offs[j] = OX(pop[i], pop[i + 1]);
        /*printf("Out\n");*/
        offs[j]->nTours = delimit(offs[j]->arr, offs[j]->limits, dist, pop[i]->limits[pop[i]->nTours - 1]);
        /*printSol(offs[j]->arr, offs[j]->limits, offs[j]->nTours, dist);*/
        ++ j;
    }
    
    return offs;
}

cvrp_sol *OX(cvrp_sol *p0, cvrp_sol *p1){
    int nc = p0->limits[p0->nTours - 1], i, j, size_sub, init_sub;
    int *isUsed = (int *)malloc(nc * sizeof(int));
    long seed = (long)clock();
    cvrp_sol *off = (cvrp_sol *)malloc(sizeof(cvrp_sol));
    
    size_sub = nc / 3;
    init_sub = ran01(&seed) * (nc - size_sub);
    
    /*printf("Starting at: %d\nWith length: %d\nnc = %d\n", init_sub, size_sub, nc);*/
    off->arr = (int *)malloc(nc * sizeof(int));
    off->limits = (int *)malloc(nc * sizeof(int));
    
    memset(isUsed, 0, nc * sizeof(int));
        
    for(i = init_sub; i < init_sub + size_sub; ++ i){
        off->arr[i] = p0->arr[i];
        isUsed[p0->arr[i] - 1] = 1;
    }

    i = (init_sub == 0 ? init_sub + size_sub : 0);
    for(j = 0; j < nc; ++ j){
        /*printf("i = %d, j = %d, value = %d ", i, j, p1->arr[j]);*/
        if(!isUsed[p1->arr[j] - 1]){
            off->arr[i] = p1->arr[j];
            isUsed[p1->arr[j] - 1] = 1;
            ++ i;
            if(i == init_sub)
                i = init_sub + size_sub;
            /*printf("Assigned");*/
        }
        /*printf("\n");*/
    }
    
    /*printf("Offspring: ");
    for(i = 0; i < nc; ++ i)
        printf("%d, ", off->arr[i]);
    printf("\n");*/
    return off;
}