#include <stdio.h>
#include <stdlib.h>

void main(int argc, char* argv[])
{
	const char* file_name = argv[1];
	int threads_number = atoi(argv[2]);

	FILE* file = fopen (file_name, "r");

	float counter = 0.0;
	float number = 0.0;
	float mean = 0.0;

	fscanf (file, "%f", &number);
	
	while (!feof (file))
	{
		//printf ("%f\n", number);

		mean = mean + number;
		
		fscanf (file, "%f", &number);
	
		counter++;	    
	}

	//printf("total %f\n", mean);
	//printf("counter %f\n", counter);
	printf("%d %f\n", threads_number, mean/counter);

	fclose (file);        
}
