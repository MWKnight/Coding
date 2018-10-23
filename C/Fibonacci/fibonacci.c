#include <stdio.h>

int main()
{
	int n = 50;
	long unsigned int num[n];
	char res[10];
	char resnum[10];
	
	if (n > 0)
	{
		num[0] = 1;
	}
	if (n > 1)
	{
		num[1] = 1;
	}

	if (n > 2)
	{
		for (int i = 2; i < n; i++)
		{
			num[i] = num[i - 1] + num[i - 2];
		}
	}


	for (int j = 0; j < n; j++)
	{
		printf("%i", j+1);
		printf(" : ");
		printf("%lu\n", num[j]);
	}

	return 0;
}
