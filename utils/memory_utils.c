
#include <stdlib.h>

/*
Free all inner pointers of a 2D array before freeing the outer pointer.
Array and inner pointers must be dynamically allocated and the outer 
array must be of length `size`.
*/
void free_2d_arr(void **arr, const int size) {
    for (int i = 0; i < size; i++) {
        free(arr[i]);
    }
    free(arr);
}