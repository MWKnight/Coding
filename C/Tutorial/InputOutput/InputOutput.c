#include <stdio.h>

int main() {

	FILE *fp;
	FILE *fpr;
	
	char buff[255];

	fp = fopen("test.txt", "w+");
	fprintf(fp, "Testing to write to a file using fprintf\n");
	fputs("Testing to write to a file using fputs\n", fp);
	fclose(fp);


	fpr = fopen("test.txt", "r");
	fscanf(fpr, "%s", buff);
	printf("1 : %s\n", buff);

	fgets(buff, 255, (FILE*)fpr);
	printf("2 : %s\n", buff);

	fgets(buff, 255, (FILE*)fpr);
	printf("3 : %s\n", buff);
	fclose(fpr);

	return 0;
}