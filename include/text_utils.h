// Header file for text-related utility functions.

#ifndef TEXT_UTILS_H
#define TEXT_UTILS_H

/*
Return the given character's position in the given string if it is in the string, 
otherwise return a negative value. `str_size` should be the length of the given string.
*/
int char_in_str(const char* str, const unsigned int str_size, const char c);

/*
Split the given string according to given delimeters and store in `output`.
Dynamically allocate memory for `output` buffer. This function assumes that sizes are 
accurate to given arrays.

Parameters:
    `str`: string to split into substrings (input)
    `str_size`: length of `str` (input)
    `default_size`: default number of substrings expected in `str` (input)
    `delimeters`: array of characters that act as delimeters to split `str` along (input)
    `delimeter_count`: number of delimeters; length of `delimeters` (input)
    `output`: pointer to array of strings to store substrings in (output) 

Returns -1 on error, otherwise returns the number of substrings found (length of `*output`).
*/
int str_split(
    const char *str, 
    const unsigned int str_size, 
    const unsigned int default_size, 
    const char *delimeters, 
    const unsigned int delimeter_count, 
    char ***output
);

/*
Stores the first line from `stream` in the `output` buffer. Dynamically resizes buffers 
to fit the line at runtime. Inital buffer size is determined by `default_size`.
The `output` buffer does NOT include an ending newline, but the `stream` pointer advances 
to the next line.

Return the size of the stored line (excluding null byte) or -1 on an error.
*/
int get_next_line(char **input, const unsigned int default_size, FILE *stream);

/*
Convert the given CSV file into a matrix of doubles, where each row represents one line in the CSV.
This function expects a CSV file consisting ONLY of floating point numbers. Error on any invalid characters/numbers.
An empty line in the given CSV will halt processing at that line.

Parameters:
    `filename`: path to CSV file (input)
    `default_line_count`: expected number of lines in the CSV (input)
    `default_line_size`: expected average length of each line (input)
    `default_line_value_count`: expected average number of values per line (input)
    `value_output`: pointer to a matrix of doubles to store CSV data (output)
    `output_lengths`: pointer to an array of lengths, each representing the length of its repsective CSV line (output)
                        e.g. if `*output_lengths[2]` is 6, then `*value_output[2]` is length 6.

Return -1 on error, otherwise return the number of lines read (rows in `*output`).
*/
int csv_to_matrix(
    const char *filename, 
    const unsigned int default_line_count,
    const unsigned int default_line_size,
    const unsigned int default_line_value_count,
    double ***value_output,
    unsigned int **output_lengths
);

/*
Convert the given string into an unsigned integer and store result in `output`.
Errors if any characters in `str` are non-digit characters.
Returns -1 on error, otherwise return the length of the given string.
*/
int str_to_uint(const char *str, unsigned int *output);

#endif