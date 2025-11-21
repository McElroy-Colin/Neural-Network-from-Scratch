// Header file for memory-related utility functions.

/*
Free all allocated inner pointers of a 2D array before freeing the outer pointer.
Array and `num_allocated` inner pointers must be dynamically allocated and the outer 
array must be of length at least `num_allocated`.
*/
void free_2d_darr(double **arr, const unsigned int num_allocated);
void free_2d_carr(char **arr, const unsigned int num_allocated);