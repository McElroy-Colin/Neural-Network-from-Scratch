#ifndef MEMORY_UTILS_CUH
#define MEMORY_UTILS_CUH


/*
Free all allocated inner pointers of a 2D double array from device memory before freeing the outer pointer from device memory.
Array and `num_allocated` inner pointers must be dynamically allocated and the outer 
array must be of length at least `num_allocated`.
*/
void cudafree_2d_darr(double **arr, const unsigned int num_allocated);

/*
Free each given pointer from device memory.
This function assumes the given pointers are not null except the final argument.
*/
void cudafree_ptrs(void *p1, ...);

#endif