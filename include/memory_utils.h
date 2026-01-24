// Header file for memory-related utility functions.

#ifndef MEMORY_UTILS_H
#define MEMORY_UTILS_H

/*
Free all allocated inner pointers of a 2D double array before freeing the outer pointer.
Array and `num_allocated` inner pointers must be dynamically allocated and the outer 
array must be of length at least `num_allocated`.
*/
void free_2d_darr(double **arr, const unsigned int num_allocated);

/*
Free all allocated inner pointers of a 2D double array from GPU memory before freeing the outer pointer from GPU memory.
Array and `num_allocated` inner pointers must be dynamically allocated and the outer 
array must be of length at least `num_allocated`.
*/
void cudafree_2d_darr(double **arr, const unsigned int num_allocated);

/*
Free all allocated inner pointers of a 2D char array before freeing the outer pointer.
Array and `num_allocated` inner pointers must be dynamically allocated and the outer 
array must be of length at least `num_allocated`.
*/
void free_2d_carr(char **arr, const unsigned int num_allocated);

#endif